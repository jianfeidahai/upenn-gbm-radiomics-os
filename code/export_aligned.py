# -*- coding: utf-8 -*-
"""导出与自写提取器完全一致的"对齐 + 1mm 各向同性重采样"MR 与 SEG 数组，
供 PyRadiomics（py3.9 环境）做独立特征提取交叉验证。

输出：
  aligned_npz/{pid}_{seq}.npz : img(float32, 1mm 重采样后的肿瘤裁剪 MR),
                                seg(int16, 同网格的 SEG 标签 1/2/3=ET/NCR/ED)
  pyrad_clinical.csv           : 临床/分子列（供后续拼宽表）
"""
import json, os, sys, csv, warnings
warnings.filterwarnings("ignore")
import numpy as np
from scipy import ndimage
sys.path.insert(0, os.path.dirname(__file__))
from imaging_io import read_mr_series, align_seg_to_mr

SEQ_ORDER = ["T1", "T1CE", "T2", "FLAIR"]
HERE = os.path.dirname(os.path.abspath(__file__))


def spacing_of(aff):
    return tuple(float(np.linalg.norm(aff[:3, i])) for i in range(3))


def main():
    m = json.load(open("manifest.json", encoding="utf-8"))
    cohort = [r for r in m if r.get("img_aligned")]
    outdir = os.path.join(HERE, "aligned_npz")
    os.makedirs(outdir, exist_ok=True)
    clin = []
    for ri, r in enumerate(cohort):
        pid = r["pid"]
        if (ri + 1) % 20 == 0:
            print(f"  export {ri+1}/{len(cohort)}", flush=True)
        base = {"pid": pid, "mgmt": r.get("mgmt"), "event": r.get("event"),
                "days": r.get("days"), "age": r.get("age"),
                "gender": r.get("gender"), "gtr": r.get("gtr")}
        clin.append(base)
        for seq in SEQ_ORDER:
            s = r["series"][seq]
            try:
                vol, vaff = read_mr_series(s["mr"])
                res, aff, segs, method = align_seg_to_mr(s["mr"], s["seg"], mr_data=(vol, vaff))
                sp = spacing_of(vaff)
            except Exception as e:
                print(f"  {pid} {seq}: read fail {e}", flush=True)
                continue
            coords = np.argwhere(res > 0)
            if coords.size == 0:
                continue
            lo = coords.min(0); hi = coords.max(0); margin = 3
            lo = np.maximum(lo - margin, 0)
            hi = np.minimum(hi + margin + 1, np.array(res.shape))
            img_c = vol[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
            msk_c = res[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
            # 注意：radiomics_features.resample_iso 会把掩码二值化（丢亚区标签），
            # 这里对强度用 order=1、对 SEG 标签用 order=0 最近邻且**不**二值化，以保留 1/2/3 亚区。
            zoom = np.array(sp, float) / 1.0
            img_r = ndimage.zoom(img_c.astype(float), zoom, order=1)
            seg_r = ndimage.zoom(msk_c.astype(float), zoom, order=0)
            np.savez(os.path.join(outdir, f"{pid}_{seq}.npz"),
                     img=img_r.astype(np.float32), seg=seg_r.astype(np.int16))
    with open(os.path.join(HERE, "pyrad_clinical.csv"), "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(clin[0].keys()))
        w.writeheader()
        for c in clin:
            w.writerow(c)
    print("EXPORT_DONE", flush=True)


if __name__ == "__main__":
    main()
