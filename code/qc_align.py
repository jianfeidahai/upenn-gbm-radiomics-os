# -*- coding: utf-8 -*-
"""对齐 QC（最终版）：对成像队列逐 (患者,序列) 用 align_seg_to_mr 双方式自动对齐，
并以"肿瘤落于 MR 前景(脑组织)比例"判据标记是否对齐。
结果写回 manifest.json：每患者增加 align_ok={seq:bool}, align_method={seq:str}, img_aligned=bool。
"""
import os, json, warnings, sys
warnings.filterwarnings("ignore")
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from imaging_io import align_seg_to_mr

ON_TISSUE_THR = 0.5
MIN_VOX = 200

def main():
    MANIFEST = os.path.join(os.path.dirname(__file__), "manifest.json")
    m = json.load(open(MANIFEST, encoding="utf-8"))
    seqs = ["T1", "T1CE", "T2", "FLAIR"]
    n4 = [r for r in m if len(r["series"]) == 4 and r["segs_all"]]
    print(f"成像队列（4序列齐全）: {len(n4)} 例，开始对齐 QC ...")
    n_usable = 0
    excluded = []
    for r in n4:
        align_ok, align_method = {}, {}
        for seq in seqs:
            s = r["series"].get(seq)
            if not s:
                align_ok[seq] = False
                align_method[seq] = "missing"
                continue
            try:
                res, saff, segs, method = align_seg_to_mr(s["mr"], s["seg"],
                                                         min_vox=MIN_VOX, on_tissue_thr=ON_TISSUE_THR)
                tot = int((res > 0).sum())
                ok = (method != "lowconf") and (tot >= MIN_VOX)
                align_ok[seq] = ok
                align_method[seq] = method
                if not ok:
                    print(f"  {r['pid']} {seq}: FAIL (method={method}, vox={tot})")
            except Exception as e:
                align_ok[seq] = False
                align_method[seq] = f"ERR:{e}"
                print(f"  {r['pid']} {seq}: ERROR {e}")
        r["align_ok"] = align_ok
        r["align_method"] = align_method
        r["img_aligned"] = all(align_ok.get(q, False) for q in seqs)
        if r["img_aligned"]:
            n_usable += 1
        else:
            excluded.append(r["pid"])
    json.dump(m, open(MANIFEST, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n对齐 QC 完成：可用成像患者 {n_usable} / {len(n4)}")
    print(f"被排除（>=1序列未对齐）: {len(excluded)} 例")
    print("  ", excluded)

if __name__ == "__main__":
    main()
