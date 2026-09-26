"""Scalable DuckDB-backed pipeline for Amazon ML Challenge 2026.

This module replaces the previous all-pairs/in-memory training and inference path.
S2/S3 stay on disk in DuckDB; S1 is processed in bounded batches; candidates,
features, predictions and submission rows are never accumulated for the full test
set in Python memory.
"""
from __future__ import annotations
import csv, json, math, os, time, pickle, subprocess, sys
from pathlib import Path
from typing import Dict, List, Tuple

import duckdb
import numpy as np
import polars as pl
import lightgbm as lgb
from rapidfuzz import fuzz

from .io_utils import resolve_dataset
from .utils import LOG, PeakMemory, ensure_dir, human_seconds, write_json
from .evaluation import entity_f05
from .submission import internal_checks

BLOCKS = {
    "exact_name": 1,
    "exact_name_heavy": 2,
    "core_name": 4,
    "exact_address": 8,
    "name_house": 16,
    "postal_name": 32,
    "name_prefix_fuzzy": 64,
    "address_prefix_fuzzy": 128,
}
FEATURE_NAMES = [
    "name_exact","name_heavy_exact","name_core_exact","name_jw","name_jaro",
    "name_lev_ratio","name_token_jaccard","name_partial","name_len_ratio",
    "addr_exact","addr_jw","addr_lev_ratio","addr_token_jaccard","addr_len_ratio",
    "house_match","postal_match","country_equal","country_conflict",
    "block_count","block_mask","high_name_high_addr","name_addr_product",
    "name_present","addr_present","source_is_s3",
]
N_FEATURES = len(FEATURE_NAMES)

def _q(s: str) -> str:
    return s.replace("'", "''")

def _norm_expr(col: str) -> str:
    return (
        "regexp_replace("
        "lower(strip_accents(coalesce(" + col + ",'')::VARCHAR)),"
        "'[^\\\\p{L}\\\\p{N}]+',' ','g')"
    )

def _heavy_expr(col: str) -> str:
    x = _norm_expr(col)
    return (
        "trim(regexp_replace(regexp_replace(regexp_replace(" + x +
        ",'\\\\bincorporated\\\\b','inc','g'),"
        "'\\\\bcorporation\\\\b','corp','g'),"
        "'\\\\blimited\\\\b','ltd','g'))"
    )

def _core_expr(col: str) -> str:
    x = _heavy_expr(col)
    return "trim(regexp_replace(" + x + ", '\\\\b(ltd|limited|llc|inc|corp|corporation|company|co|pvt|private|plc|llp)\\\\b',' ','g'))"

def _addr_expr(col: str) -> str:
    x = _norm_expr(col)
    return "trim(regexp_replace(" + x + ", '\\\\b(road|rd)\\\\b','rd','g'))"

def _prep_sql(src: str, path: str, limit: int | None = None) -> str:
    lim = f" LIMIT {int(limit)}" if limit else ""
    return f"""
    CREATE OR REPLACE TABLE {src}_raw AS
    SELECT * FROM read_csv_auto('{_q(path)}', delim='\\t', header=true,
                                all_varchar=true, ignore_errors=false);
    CREATE OR REPLACE TABLE {src} AS
    SELECT
      row_number() OVER () - 1 AS rid,
      entity_id::VARCHAR AS entity_id,
      business_name::VARCHAR AS raw_name,
      business_address::VARCHAR AS raw_addr,
      country::VARCHAR AS raw_country,
      {_norm_expr('business_name')} AS name,
      {_heavy_expr('business_name')} AS name_heavy,
      {_core_expr('business_name')} AS name_core,
      {_addr_expr('business_address')} AS addr,
      lower(trim(coalesce(country,''))) AS country,
      left({_core_expr('business_name')},3) AS name_prefix,
      left({_addr_expr('business_address')},4) AS addr_prefix,
      regexp_extract({_addr_expr('business_address')}, '(\\\\d{{1,6}})', 1) AS house,
      regexp_extract({_addr_expr('business_address')}, '\\\\b(\\\\d{{5,6}})\\\\b', 1) AS postal
    FROM {src}_raw{lim};
    DROP TABLE {src}_raw;
    """

class ScalableER:
    def __init__(self, cfg, data_root=None):
        self.cfg = cfg
        self.paths = resolve_dataset(data_root or cfg.raw.get("paths",{}).get("data_root"))
        self.db_path = Path(cfg.artifacts_dir) / "scalable_er.duckdb"
        ensure_dir(self.db_path.parent)
        self.con = duckdb.connect(str(self.db_path))
        stream = cfg.section("streaming")
        self.batch = int(stream.get("batch_size_s1", 5000))
        self.max_candidates = int(stream.get("max_candidates_per_s1", 300))
        self.train_entities = int(stream.get("training_entities", 100000))
        self.val_entities = int(stream.get("validation_entities", 20000))
        self.seed = int(cfg.seed)
        self.con.execute("SET threads TO ?", [max(1, min(os.cpu_count() or 4, 8))])
        self.con.execute("SET memory_limit='24GB'")
        self.con.execute("SET temp_directory=?", [str(self.db_path.parent / "duckdb_tmp")])
        ensure_dir(self.db_path.parent / "duckdb_tmp")

    def close(self):
        try: self.con.close()
        except Exception: pass

    def prepare(self, train: bool = True, test: bool = True):
        t=time.perf_counter()
        if train:
            self.con.execute(_prep_sql("tr_s1", str(self.paths.train["S1"])))
            self.con.execute(_prep_sql("tr_c", str(self.paths.train["S2"])))
            self.con.execute("INSERT INTO tr_c SELECT * FROM (SELECT * FROM tr_c) WHERE FALSE") if False else None
            # Rebuild candidate source with explicit source tag and S2/S3.
            self.con.execute("CREATE OR REPLACE TABLE tr_s2 AS SELECT row_number() OVER()-1 AS gid, 'S2' AS source, * FROM tr_c")
            self.con.execute(_prep_sql("tr_s3", str(self.paths.train["S3"])))
            self.con.execute("CREATE OR REPLACE TABLE tr_s3x AS SELECT row_number() OVER()-1 + (SELECT count(*) FROM tr_s2) AS gid, 'S3' AS source, * FROM tr_s3")
            self.con.execute("CREATE OR REPLACE TABLE tr_cand AS SELECT * FROM tr_s2 UNION ALL SELECT * FROM tr_s3x")
            self.con.execute("CREATE OR REPLACE TABLE tr_gt AS SELECT source1_entity_id::VARCHAR AS s1_id, matched_entity_ids::VARCHAR AS mids FROM read_csv_auto(?, delim='\\t', header=true, all_varchar=true)", [str(self.paths.train_gt)])
            self.con.execute("ANALYZE tr_cand")
            self.con.execute("ANALYZE tr_s1")
        if test:
            self.con.execute(_prep_sql("te_s1", str(self.paths.test["S1"])))
            self.con.execute(_prep_sql("te_c", str(self.paths.test["S2"])))
            self.con.execute("CREATE OR REPLACE TABLE te_s2 AS SELECT row_number() OVER()-1 AS gid, 'S2' AS source, * FROM te_c")
            self.con.execute(_prep_sql("te_s3", str(self.paths.test["S3"])))
            self.con.execute("CREATE OR REPLACE TABLE te_s3x AS SELECT row_number() OVER()-1 + (SELECT count(*) FROM te_s2) AS gid, 'S3' AS source, * FROM te_s3")
            self.con.execute("CREATE OR REPLACE TABLE te_cand AS SELECT * FROM te_s2 UNION ALL SELECT * FROM te_s3x")
            self.con.execute("ANALYZE te_cand")
            self.con.execute("ANALYZE te_s1")
        LOG.info("DuckDB preparation completed in %s", human_seconds(time.perf_counter()-t))

    def _candidate_sql(self, prefix: str) -> str:
        s1=f"{prefix}_s1"; c=f"{prefix}_cand"
        cap=self.max_candidates
        return f"""
        WITH blocks AS (
          SELECT s.rid, c.gid, {BLOCKS['exact_name']} AS bit
          FROM {s1} s JOIN {c} c ON s.country=c.country AND s.name<>'' AND s.name=c.name
          UNION ALL
          SELECT s.rid,c.gid,{BLOCKS['exact_name_heavy']} FROM {s1} s JOIN {c} c
            ON s.country=c.country AND s.name_heavy<>'' AND s.name_heavy=c.name_heavy
          UNION ALL
          SELECT s.rid,c.gid,{BLOCKS['core_name']} FROM {s1} s JOIN {c} c
            ON s.country=c.country AND s.name_core<>'' AND s.name_core=c.name_core
          UNION ALL
          SELECT s.rid,c.gid,{BLOCKS['exact_address']} FROM {s1} s JOIN {c} c
            ON s.addr<>'' AND s.addr=c.addr
          UNION ALL
          SELECT s.rid,c.gid,{BLOCKS['name_house']} FROM {s1} s JOIN {c} c
            ON s.country=c.country AND s.name<>'' AND s.house<>'' AND s.name=c.name AND s.house=c.house
          UNION ALL
          SELECT s.rid,c.gid,{BLOCKS['postal_name']} FROM {s1} s JOIN {c} c
            ON s.country=c.country AND s.postal<>'' AND s.postal=c.postal
           AND (s.name='' OR jaro_winkler_similarity(s.name,c.name)>=0.50)
          UNION ALL
          SELECT rid,gid,{BLOCKS['name_prefix_fuzzy']} FROM (
            SELECT s.rid,c.gid,jaro_winkler_similarity(s.name,c.name) sim,
                   row_number() OVER(PARTITION BY s.rid ORDER BY jaro_winkler_similarity(s.name,c.name) DESC) rn
            FROM {s1} s JOIN {c} c
              ON s.country=c.country AND s.name_prefix<>'' AND s.name_prefix=c.name_prefix
             AND jaro_winkler_similarity(s.name,c.name)>=0.68
          ) q WHERE rn<=100
          UNION ALL
          SELECT rid,gid,{BLOCKS['address_prefix_fuzzy']} FROM (
            SELECT s.rid,c.gid,jaro_winkler_similarity(s.addr,c.addr) sim,
                   row_number() OVER(PARTITION BY s.rid ORDER BY jaro_winkler_similarity(s.addr,c.addr) DESC) rn
            FROM {s1} s JOIN {c} c
              ON s.country=c.country AND s.addr_prefix<>'' AND s.addr_prefix=c.addr_prefix
             AND jaro_winkler_similarity(s.addr,c.addr)>=0.62
          ) q WHERE rn<=100
        ), merged AS (
          SELECT rid,gid,bit_or(bit) AS block_mask, count(*) AS block_count
          FROM blocks GROUP BY rid,gid
        ), ranked AS (
          SELECT *, row_number() OVER(PARTITION BY rid ORDER BY block_count DESC, block_mask DESC, gid) rn
          FROM merged
        )
        SELECT rid,gid,block_mask,block_count
        FROM ranked WHERE rn<={cap}
        ORDER BY rid,gid
        """

    def candidates(self, prefix: str, rid_lo: int, rid_hi: int) -> pl.DataFrame:
        self.con.execute("CREATE OR REPLACE TEMP TABLE s1_batch AS SELECT * FROM "+prefix+"_s1 WHERE rid>=? AND rid<?", [rid_lo,rid_hi])
        q=self._candidate_sql("s1_batch" if False else prefix).replace(f"{prefix}_s1 s","s1_batch s")
        return self.con.execute(q).pl()

    def _feature_sql(self, prefix: str) -> str:
        c=f"{prefix}_cand"
        return f"""
        SELECT
          b.rid,b.gid,c.entity_id,c.source,
          b.block_mask,b.block_count,
          (s.name<>'' AND s.name=c.name)::INT name_exact,
          (s.name_heavy<>'' AND s.name_heavy=c.name_heavy)::INT name_heavy_exact,
          (s.name_core<>'' AND s.name_core=c.name_core)::INT name_core_exact,
          jaro_winkler_similarity(s.name,c.name) name_jw,
          jaro_similarity(s.name,c.name) name_jaro,
          1.0-(levenshtein(s.name,c.name)::DOUBLE/GREATEST(length(s.name),length(c.name),1)) name_lev_ratio,
          CASE WHEN s.name='' OR c.name='' THEN 0.0
               ELSE length(list_intersect(string_split(s.name,' '),string_split(c.name,' ')))::DOUBLE/
                    GREATEST(length(list_unique(string_split(s.name,' '))),1) END name_token_jaccard,
          jaro_winkler_similarity(s.name,c.name) name_partial,
          LEAST(length(s.name),length(c.name))::DOUBLE/GREATEST(length(s.name),length(c.name),1) name_len_ratio,
          (s.addr<>'' AND s.addr=c.addr)::INT addr_exact,
          jaro_winkler_similarity(s.addr,c.addr) addr_jw,
          1.0-(levenshtein(s.addr,c.addr)::DOUBLE/GREATEST(length(s.addr),length(c.addr),1)) addr_lev_ratio,
          CASE WHEN s.addr='' OR c.addr='' THEN 0.0
               ELSE length(list_intersect(string_split(s.addr,' '),string_split(c.addr,' ')))::DOUBLE/
                    GREATEST(length(list_unique(string_split(s.addr,' '))),1) END addr_token_jaccard,
          LEAST(length(s.addr),length(c.addr))::DOUBLE/GREATEST(length(s.addr),length(c.addr),1) addr_len_ratio,
          (s.house<>'' AND s.house=c.house)::INT house_match,
          (s.postal<>'' AND s.postal=c.postal)::INT postal_match,
          (s.country<>'' AND s.country=c.country)::INT country_equal,
          (s.country<>'' AND c.country<>'' AND s.country<>c.country)::INT country_conflict,
          b.block_count,
          b.block_mask,
          ((jaro_winkler_similarity(s.name,c.name)>=0.90 AND jaro_winkler_similarity(s.addr,c.addr)>=0.85))::INT high_name_high_addr,
          jaro_winkler_similarity(s.name,c.name)*jaro_winkler_similarity(s.addr,c.addr) name_addr_product,
          (s.name<>' ' AND s.name<>'')::INT name_present,
          (s.addr<>' ' AND s.addr<>'')::INT addr_present,
          (c.source='S3')::INT source_is_s3
        FROM s1_batch s JOIN candidate_batch b ON s.rid=b.rid
        JOIN {c} c ON b.gid=c.gid
        """

    def feature_rows(self, prefix: str, cand: pl.DataFrame) -> pl.DataFrame:
        self.con.register("candidate_batch", cand.to_arrow())
        q=self._feature_sql(prefix)
        out=self.con.execute(q).pl()
        self.con.unregister("candidate_batch")
        return out

    def _gt_map(self, ids: List[str]) -> Dict[str,set]:
        if not ids: return {}
        self.con.execute("CREATE OR REPLACE TEMP TABLE wanted(id VARCHAR)")
        self.con.executemany("INSERT INTO wanted VALUES (?)", [(x,) for x in ids])
        rows=self.con.execute("""
          SELECT g.s1_id,g.mids FROM tr_gt g JOIN wanted w ON g.s1_id=w.id
        """).fetchall()
        ans={}
        for sid,mids in rows:
            if mids is None or not str(mids).strip(): ans[sid]=set()
            else: ans[sid]={x.strip() for x in str(mids).split(',') if x.strip()}
        return ans

    def _select_s1(self, table: str, n: int) -> List[Tuple[int,str]]:
        return self.con.execute(f"""
          SELECT rid,entity_id FROM {table}
          USING SAMPLE RESERVOIR ({int(n)} ROWS) REPEATABLE ({self.seed})
          ORDER BY rid
        """).fetchall()

    def _build_train_matrix(self, rows: List[pl.DataFrame], gt_by_id: Dict[str,set]):
        df=pl.concat(rows, how="vertical") if rows else pl.DataFrame()
        if df.is_empty(): raise RuntimeError("No training candidate rows were produced.")
        y=np.zeros(df.height,dtype=np.int8)
        mids=set().union(*gt_by_id.values()) if gt_by_id else set()
        y=np.fromiter((1 if eid in gt_by_id.get(sid,set()) else 0 for sid,eid in zip(df["s1_id"].to_list(),df["entity_id"].to_list())), dtype=np.int8, count=df.height)
        X=df.select(FEATURE_NAMES).to_numpy().astype(np.float32,copy=False)
        return X,y,df

    def train(self) -> dict:
        self.prepare(train=True,test=False)
        sample=self._select_s1("tr_s1", self.train_entities+self.val_entities)
        rng=np.random.default_rng(self.seed); rng.shuffle(sample)
        ntr=int(len(sample)*0.8)
        train_sample=sample[:ntr]; val_sample=sample[ntr:]
        selected=sample
        gt_by_id=self._gt_map([x[1] for x in selected])
        train_rows=[]; val_rows=[]; pos=neg=0
        raw_pos=0; raw_gt=0; raw_counts=[]
        for j in range(0,len(selected),self.batch):
            chunk=selected[j:j+self.batch]
            self.con.execute("DROP TABLE IF EXISTS selected_rids")
            self.con.execute("CREATE TEMP TABLE selected_rids(rid BIGINT)")
            self.con.executemany("INSERT INTO selected_rids VALUES (?)", [(int(x[0]),) for x in chunk])
            self.con.execute("CREATE OR REPLACE TEMP TABLE s1_batch AS SELECT * FROM tr_s1 WHERE rid IN (SELECT rid FROM selected_rids)")
            cand=self.con.execute(self._candidate_sql("tr_s1").replace("tr_s1 s","s1_batch s")).pl()
            # Guarantee known positives are in the training candidate set.
            ids=[x[1] for x in chunk]
            posdf=self.con.execute("""
              SELECT s.rid,c.gid,0::UBIGINT block_mask,0::BIGINT block_count
              FROM s1_batch s JOIN tr_gt g ON s.entity_id=g.s1_id
              CROSS JOIN LATERAL unnest(string_split(coalesce(g.mids,''),',')) m(mid)
              JOIN tr_cand c ON trim(m.mid)=c.entity_id
            """).pl()
            if posdf.height:
                cand=pl.concat([cand,posdf]).unique(["rid","gid"])
            raw_candidate_batch=cand.clone()
            if raw_candidate_batch.height:
                self.con.register("raw_candidate_batch", raw_candidate_batch.to_arrow())
                rawdf=self.con.execute("SELECT r.rid,c.entity_id FROM raw_candidate_batch r JOIN tr_cand c ON r.gid=c.gid").pl()
                self.con.unregister("raw_candidate_batch")
                rid_map={x[1]:x[0] for x in chunk}
                for sid in [x[1] for x in chunk]:
                    gtids=gt_by_id.get(sid,set())
                    raw_gt += len(gtids)
                    if gtids:
                        got=set(rawdf.filter(pl.col("rid")==rid_map[sid])["entity_id"].to_list())
                        raw_pos += len(gtids & got)
                raw_counts.extend([int(x) for x in rawdf.group_by("rid").len()["len"].to_list()])
            feats=self.feature_rows("tr",cand)
            train_ids={x[0] for x in train_sample}; mask=np.array([r in train_ids for r in feats["rid"].to_list()])
            tr=feats.filter(pl.Series(mask)); va=feats.filter(pl.Series(~mask))
            # Keep every positive but bound training negatives per entity.
            sampled_parts=[]
            for sid in tr["s1_id"].unique().to_list():
                part=tr.filter(pl.col("s1_id")==sid)
                gtset=gt_by_id.get(sid,set())
                pos_part=part.filter(pl.col("entity_id").is_in(list(gtset)))
                neg_part=part.filter(~pl.col("entity_id").is_in(list(gtset)))
                neg_cap=max(20*max(1,pos_part.height),20)
                if neg_part.height>neg_cap:
                    neg_part=neg_part.sample(n=neg_cap,seed=self.seed)
                sampled_parts.append(pl.concat([pos_part,neg_part],how="vertical"))
            tr_sampled=pl.concat(sampled_parts,how="vertical") if sampled_parts else tr
            if tr_sampled.height: train_rows.append(tr_sampled)
            if va.height: val_rows.append(va)
            LOG.info("training candidates batch %d/%d: %d",min(j+self.batch,len(selected)),len(selected),feats.height)
        gt_by_id=self._gt_map([x[1] for x in selected])
        Xtr,ytr,train_df=self._build_train_matrix(train_rows,gt_by_id)
        Xva,yva,val_df=self._build_train_matrix(val_rows,gt_by_id)
        # Hard-negative cap while preserving all positives.
        rng=np.random.default_rng(self.seed)
        pos_idx=np.flatnonzero(ytr==1); neg_idx=np.flatnonzero(ytr==0)
        cap=min(len(neg_idx),max(len(pos_idx)*20,10000))
        if len(neg_idx)>cap: neg_idx=rng.choice(neg_idx,size=cap,replace=False)
        keep=np.concatenate([pos_idx,neg_idx]); rng.shuffle(keep)
        model=lgb.LGBMClassifier(
            objective="binary",n_estimators=2500,num_leaves=96,learning_rate=0.05,
            min_child_samples=30,subsample=0.9,colsample_bytree=0.9,
            reg_alpha=0.1,reg_lambda=1.0,random_state=self.seed,n_jobs=-1
        )
        model.fit(Xtr[keep],ytr[keep],eval_set=[(Xva,yva)],callbacks=[lgb.early_stopping(150,verbose=False)])
        pva=model.predict_proba(Xva)[:,1]
        # entity-level threshold selection, including empty predictions.
        val_sid=val_df["s1_id"].to_list(); val_eid=val_df["entity_id"].to_list()
        best=(0.0,0.5)
        for t in np.linspace(0.50,0.995,100):
            scores=[]
            for sid in set(val_sid):
                gt=gt_by_id.get(sid,set())
                pred={eid for eid,p in zip([e for s,e in zip(val_sid,val_eid) if s==sid],[q for s,q in zip(val_sid,pva) if s==sid]) if p>=t}
                scores.append(entity_f05(gt,pred))
            sc=float(np.mean(scores)) if scores else 0.0
            if sc>best[0]: best=(sc,float(t))
        ensure_dir(self.cfg.models_dir)
        model.booster_.save_model(str(self.cfg.models_dir/"lgbm_pair_model.txt"))
        bundle={"feature_names":FEATURE_NAMES,"decision":{"threshold":best[1],"threshold_s2":None,"threshold_s3":None,"high_conf":0.999,"margin":0.0,"top_only":False,"max_matches_per_entity":0},"candidate_cap":self.max_candidates,"train_entities":len(train_sample),"validation_entities":len(val_sample),"metrics":{"macro_f05":best[0]},"engine":"duckdb_streaming_v1","model_license":"MIT"}
        (Path(self.cfg.models_dir)/"artifacts_bundle.json").write_text(json.dumps(bundle,indent=2))
        with open(Path(self.cfg.models_dir)/"model.pkl","wb") as f: pickle.dump(model,f)
        cand_recall=(raw_pos/raw_gt) if raw_gt else 0.0
        rec={"candidate_recall":cand_recall,"positive_pairs_found":raw_pos,"positive_pairs_total":raw_gt,
             "mean_candidates":float(np.mean(raw_counts)) if raw_counts else 0.0,
             "p95_candidates":float(np.percentile(raw_counts,95)) if raw_counts else 0.0,
             "p99_candidates":float(np.percentile(raw_counts,99)) if raw_counts else 0.0,
             "max_candidates":int(max(raw_counts)) if raw_counts else 0,
             "candidate_cap":self.max_candidates,"sample_entities":len(selected)}
        write_json(self.cfg.reports_dir/"candidate_recall.json",rec)
        validation={"eval":{"macro_f05":best[0]},"decision":bundle["decision"],
                    "candidate_recall":rec,"train_entities":len(train_sample),
                    "validation_entities":len(val_sample),"feature_count":len(FEATURE_NAMES)}
        write_json(self.cfg.reports_dir/"validation_results.json",validation)
        LOG.info("CANDIDATE recall(raw)=%.6f; VALIDATION macro F0.5=%.6f threshold=%.4f",cand_recall,best[0],best[1])
        return validation

    def predict(self) -> dict:
        self.prepare(train=False,test=True)
        bundle=json.loads((Path(self.cfg.models_dir)/"artifacts_bundle.json").read_text())
        with open(Path(self.cfg.models_dir)/"model.pkl","rb") as f: model=pickle.load(f)
        out=Path(self.cfg.output_dir); ensure_dir(out)
        match_path=out/"matching_results.tsv"; cand_path=out/"candidate_pairs.tsv"
        with open(match_path,"w",encoding="utf-8",newline="") as mf, open(cand_path,"w",encoding="utf-8",newline="") as cf:
            mw=csv.writer(mf,delimiter="\t",lineterminator="\n"); cw=csv.writer(cf,delimiter="\t",lineterminator="\n")
            mw.writerow(["source1_entity_id","matched_entity_ids"]); cw.writerow(["source1_entity_id","candidate_entity_ids"])
            n=int(self.con.execute("SELECT count(*) FROM te_s1").fetchone()[0]); total=0
            for lo in range(0,n,self.batch):
                hi=min(lo+self.batch,n)
                self.con.execute("CREATE OR REPLACE TEMP TABLE s1_batch AS SELECT * FROM te_s1 WHERE rid>=? AND rid<?",[lo,hi])
                cand=self.con.execute(self._candidate_sql("te_s1").replace("te_s1 s","s1_batch s")).pl()
                feats=self.feature_rows("te",cand)
                if feats.is_empty():
                    ids=self.con.execute("SELECT rid,entity_id FROM s1_batch ORDER BY rid").fetchall()
                    for _,sid in ids: mw.writerow([sid,""]); cw.writerow([sid,""])
                    continue
                X=feats.select(FEATURE_NAMES).to_numpy().astype(np.float32,copy=False)
                probs=model.predict_proba(X)[:,1]
                by={}
                cby={}
                for sid,eid,p in zip(feats["s1_id"].to_list(),feats["entity_id"].to_list(),probs):
                    cby.setdefault(sid,[]).append(eid)
                    if p>=bundle["decision"]["threshold"] or p>=bundle["decision"]["high_conf"]:
                        by.setdefault(sid,[]).append(eid)
                ids=self.con.execute("SELECT rid,entity_id FROM s1_batch ORDER BY rid").fetchall()
                for _,sid in ids:
                    cs=sorted(set(cby.get(sid,[]))); ms=sorted(set(by.get(sid,[])))
                    cw.writerow([sid,",".join(cs)]); mw.writerow([sid,",".join(ms)])
                total+=len(ids)
                if total%50000==0: LOG.info("test inference: %d/%d S1",total,n)
        LOG.info("test inference complete: %d S1",total)
        return {"rows":total,"matching_results":str(match_path),"candidate_pairs":str(cand_path)}

    def validate_submission(self) -> dict:
        validator=self.paths.validator
        if validator is None:
            raise FileNotFoundError("utils/validate_submission.py was not found")
        cmd=[sys.executable,str(validator),"--matching",str(self.cfg.output_dir/"matching_results.tsv"),
             "--candidate",str(self.cfg.output_dir/"candidate_pairs.tsv"),
             "--test-dir",str(self.paths.root/"dataset"/"test")]
        p=subprocess.run(cmd,capture_output=True,text=True)
        report={"ran":True,"pass_":p.returncode==0,"returncode":p.returncode,
                "stdout":p.stdout,"stderr":p.stderr}
        write_json(self.cfg.reports_dir/"submission_validation.json",report)
        if p.returncode!=0:
            raise RuntimeError("Official submission validator failed:\\n"+p.stdout+"\\n"+p.stderr)
        LOG.info("OFFICIAL VALIDATOR: PASS")
        return report

    def profile(self):
        from .data_profile import build_profile
        return build_profile(self.paths.root, str(self.cfg.reports_dir))

    def all(self, team_name=None, make_zip=False):
        train=self.train(); pred=self.predict()
        return {"train":train,"predict":pred}
