# -*- coding: utf-8 -*-
"""
构建 269 例可用患者的序列/分割映射清单 (manifest.json)。

配对策略（同 study 共定位 + 优先 CaPTk 处理序列）：
  UPenn-GBM 的 SEG 是在 CaPTk 配准空间生成的；每个 study 内含"原始/Processed_CaPTk 序列 + SEG"。
  若按 UID 配对到原始序列会与 SEG 不在同一空间 -> 错位。
  因此：以 study 为单位，把 SEG 与其同 study、描述含 Processed_CaPTk 的 MR 配对，
  保证 SEG<->MR 共处于配准空间，重采样对齐。
"""
import os, json, glob, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

from idc_index import IDCClient
import pandas as pd, numpy as np
import pydicom

from config import RAW_ROOT, MANIFEST

def classify(desc):
    d = (desc or "").lower()
    if "flair" in d:
        return "FLAIR"
    if "t1" in d and ("post" in d or "stealth" in d):
        return "T1CE"
    if "t1" in d:
        return "T1"
    if "t2" in d:
        return "T2"
    return None

def hdr(folder):
    fs = sorted(glob.glob(os.path.join(folder, "*.dcm")))
    if not fs:
        return None
    return pydicom.dcmread(fs[0], force=True)

def main():
    c = IDCClient(); c.fetch_index("clinical_index")
    t = c.get_clinical_table("upenn_gbm_clinical")
    sub = t[t["mgmt"].isin(["Methylated", "Unmethylated"])].copy()
    def ev(s):
        if s == "Deceased": return 1
        if s in ["Alive", "Lost to Follow-up", "Deceased - uncertain date of death"]: return 0
        return np.nan
    sub["event"] = sub["survival_status"].apply(ev)
    sub["days"] = pd.to_numeric(sub["survival_from_surgery_days_updated"], errors="coerce")
    n_raw = len(sub)
    sub = sub[sub["days"].notna() & sub["event"].notna()].copy()
    sub = sub.drop_duplicates(subset=["dicom_patient_id"], keep="first").reset_index(drop=True)
    print("可用(含重复):", n_raw, " 去重后唯一患者:", len(sub))
    sub["pid"] = sub["dicom_patient_id"]

    meta = {r.pid: {"mgmt": r.mgmt, "event": int(r.event), "days": float(r.days),
                    "age": float(r.age_at_scan_years), "gender": r.gender,
                    "gtr": (r.gtr_over90percent if isinstance(r.gtr_over90percent, str) else "NA")}
           for r in sub.itertuples()}

    base = Path(RAW_ROOT)
    manifest = []
    for pid in sub["pid"].tolist():
        pdir = base / pid
        if not pdir.exists():
            print("MISSING DIR", pid); continue
        # 按 study 分组
        studies = {}
        for study in [p for p in pdir.iterdir() if p.is_dir()]:
            mrs, segs = [], []
            for ser in [p for p in study.iterdir() if p.is_dir()]:
                ds = hdr(ser)
                if ds is None:
                    continue
                mod = (getattr(ds, "Modality", "") or "").upper()
                desc = getattr(ds, "SeriesDescription", "") or ""
                if mod == "SEG":
                    ref = None
                    try:
                        ref = ds.ReferencedSeriesSequence[0].SeriesInstanceUID
                    except Exception:
                        ref = None
                    segs.append((ser, ref))
                else:
                    seq = classify(desc)
                    if seq:
                        suid = getattr(ds, "SeriesInstanceUID", None)
                        n = len(glob.glob(str(ser / "*.dcm")))
                        mrs.append((seq, suid, ser, desc, n))
            if mrs or segs:
                studies[study] = {"mrs": mrs, "segs": segs}
        rec = {"pid": pid}; rec.update(meta[pid]); rec["series"] = {}
        all_segs = []
        for study, cont in studies.items():
            all_segs.extend(s[0] for s in cont["segs"])
            for segdir, ref in cont["segs"]:
                cands = cont["mrs"]
                if not cands:
                    continue
                # 关键：SEG 必须配对到其 ReferencedSeriesSequence 指向的 MR 系列，
                # 该系列的几何才是 SEG 生成所在空间，否则重采样错位（掩码落空）。
                matched = [m for m in cands if m[1] == ref]
                if matched:
                    pick = matched[0]
                else:
                    # 回退：同 study 的 CaPTk 处理序列，再回退任意序列
                    captk = [m for m in cands if "captk" in m[3].lower()]
                    pick = captk[0] if captk else cands[0]
                seq = pick[0]
                if seq in rec["series"]:
                    continue
                rec["series"][seq] = {"mr": str(pick[2]), "seg": str(segdir)}
        rec["segs_all"] = [str(s) for s in all_segs]
        manifest.append(rec)

    Path(MANIFEST).parent.mkdir(parents=True, exist_ok=True)
    manifest = list({r["pid"]: r for r in manifest}.values())
    json.dump(manifest, open(MANIFEST, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    n_all = len(manifest)
    n4 = sum(1 for r in manifest if len(r["series"]) == 4)
    noseg = sum(1 for r in manifest if not r["segs_all"])
    print(f"扫描完成: 总 {n_all} 例, 共定位配对得4序列 {n4} 例, 完全无SEG {noseg} 例")
    print("manifest ->", MANIFEST)

if __name__ == "__main__":
    main()
