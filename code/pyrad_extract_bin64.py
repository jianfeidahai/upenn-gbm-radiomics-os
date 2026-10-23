# -*- coding: utf-8 -*-
"""IBSI 固定箱数（binCount=64）方案的特征重提取（py3.9 环境）。

与 pyrad_extract.py 的唯一差别：离散化由"binWidth = ROI 灰度范围/64"
改为 PyRadiomics 的 binCount=64，即 IBSI 推荐的固定箱数（fixed bin number）
方案，保证每个感兴趣区都离散为恰好 64 个灰度级、规则在全队列一致。

输入：aligned_npz/{pid}_{seq}.npz, pyrad_clinical.csv
输出：features_wide_pyrad_bin64.csv（列结构与 features_wide_pyrad.csv 一致）
"""
import os
import csv
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import SimpleITK as sitk
import radiomics
from radiomics import featureextractor

HERE = os.path.dirname(os.path.abspath(__file__))
SEQ_ORDER = ["T1", "T1CE", "T2", "FLAIR"]
SUB_LAB = {1: "ET", 2: "NCR", 3: "ED"}
CLASSES = ["shape", "firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm"]
BIN_COUNT = 64


def make_extractor():
    ex = featureextractor.RadiomicsFeatureExtractor()
    ex.disableAllFeatures()
    for c in CLASSES:
        ex.enableFeatureClassByName(c)
    # IBSI 固定箱数：每个 ROI 恰好 BIN_COUNT 个灰度级
    ex.settings["binCount"] = BIN_COUNT
    ex.settings.pop("binWidth", None)
    return ex


def main():
    ex = make_extractor()
    npzdir = os.path.join(HERE, "aligned_npz")
    clin = list(csv.DictReader(open(os.path.join(HERE, "pyrad_clinical.csv"),
                                    encoding="utf-8-sig")))
    rows = []
    for n, c in enumerate(clin, 1):
        pid = c["pid"]
        wide = dict(c)
        for seq in SEQ_ORDER:
            fp = os.path.join(npzdir, f"{pid}_{seq}.npz")
            if not os.path.exists(fp):
                continue
            d = np.load(fp)
            img = d["img"].astype(np.float64)
            seg = np.rint(d["seg"]).astype(np.int32)
            if img.size == 0 or seg.size == 0:
                continue
            img_s = sitk.GetImageFromArray(img)
            img_s.SetSpacing((1.0, 1.0, 1.0))
            for lab, sub in SUB_LAB.items():
                m = (seg == lab).astype(np.uint8)
                if int(m.sum()) == 0:
                    continue
                mask_s = sitk.GetImageFromArray(m)
                mask_s.CopyInformation(img_s)
                try:
                    feats = ex.execute(img_s, mask_s)
                except Exception as e:
                    print(f"  ERR {pid} {seq} {sub}: {e}", flush=True)
                    continue
                for k, v in feats.items():
                    if not k.startswith("original"):
                        continue
                    try:
                        vv = float(v)
                        if np.isnan(vv) or np.isinf(vv):
                            vv = ""
                    except Exception:
                        vv = ""
                    wide[f"{seq}_{sub}_{k}"] = vv
        rows.append(wide)
        if n % 20 == 0:
            print(f"  patients {n}/{len(clin)}", flush=True)

    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(os.path.join(HERE, "features_wide_pyrad_bin64.csv"), "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"PYRAD_BIN64_DONE rows={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
