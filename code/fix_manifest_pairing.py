# -*- coding: utf-8 -*-
"""用"SEG 引用 UID 精确匹配 MR 系列"重做配对，原地更新 manifest.json。
只重写 series / segs_all 字段，保留临床字段（不改样本量口径）。"""
import os, json, glob, warnings
warnings.filterwarnings("ignore")
import pydicom
from config import RAW_ROOT, MANIFEST

def classify(desc):
    d = (desc or "").lower()
    if "flair" in d: return "FLAIR"
    if "t1" in d and ("post" in d or "stealth" in d): return "T1CE"
    if "t1" in d: return "T1"
    if "t2" in d: return "T2"
    return None

def hdr(folder):
    fs = sorted(glob.glob(os.path.join(folder, "*.dcm")))
    if not fs: return None
    return pydicom.dcmread(fs[0], force=True)

m = json.load(open(MANIFEST, encoding="utf-8"))
base = RAW_ROOT
for r in m:
    pid = r["pid"]
    pdir = os.path.join(base, pid)
    if not os.path.isdir(pdir):
        continue
    # 收集该患者所有 study 下的 MR 系列 (uid -> (seq, folder, desc))
    uid_mr = {}
    # 收集所有 SEG (folder -> ref_uid)
    segs = []
    for study in [p for p in os.listdir(pdir) if os.path.isdir(os.path.join(pdir, p))]:
        sdir = os.path.join(pdir, study)
        for ser in [p for p in os.listdir(sdir) if os.path.isdir(os.path.join(sdir, p))]:
            fdir = os.path.join(sdir, ser)
            ds = hdr(fdir)
            if ds is None: continue
            mod = (getattr(ds, "Modality", "") or "").upper()
            desc = getattr(ds, "SeriesDescription", "") or ""
            if mod == "SEG":
                ref = None
                try:
                    ref = ds.ReferencedSeriesSequence[0].SeriesInstanceUID
                except Exception:
                    ref = None
                segs.append((fdir, ref))
            else:
                seq = classify(desc)
                if seq:
                    suid = getattr(ds, "SeriesInstanceUID", None)
                    uid_mr[suid] = (seq, fdir, desc)
    # 配对：SEG -> 引用 UID 对应的 MR（跨 study 查找）
    series = {}
    for segdir, ref in segs:
        pick = None
        if ref in uid_mr:
            pick = uid_mr[ref]
        else:
            # 回退：同目录所在 study 内任意 CaPTk / 任意 MR
            sdir = os.path.dirname(segdir)
            cand = [(seq, f, d) for (seq, f, d) in uid_mr.values()
                    if os.path.dirname(f) == sdir]
            captk = [c for c in cand if "captk" in c[2].lower()]
            pick = captk[0] if captk else (cand[0] if cand else None)
        if pick is None:
            continue
        seq, fdir, _ = pick
        if seq in series:
            continue
        series[seq] = {"mr": fdir, "seg": segdir}
    r["series"] = series
    r["segs_all"] = [s[0] for s in segs]

json.dump(m, open(MANIFEST, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

n_all = len(m)
n4 = sum(1 for r in m if len(r["series"]) == 4)
noseg = sum(1 for r in m if not r["segs_all"])
print(f"重配对完成: 总 {n_all} 例, 4序列齐全 {n4} 例, 完全无SEG {noseg} 例")
