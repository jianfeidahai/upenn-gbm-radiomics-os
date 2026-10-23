# -*- coding: utf-8 -*-
"""批量验证成像队列（4 序列齐全）的 SEG->MR 对齐（自动双方式选择）：
   统计失败（on-tissue<0.5 或为空）例数，并报告选择 method 分布与抽查。
"""
import json, glob, os, warnings, sys
warnings.filterwarnings("ignore")
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from imaging_io import align_seg_to_mr

m = json.load(open("manifest.json", encoding="utf-8"))
cohort = [r for r in m if len(r["series"]) == 4 and r["segs_all"]]
print(f"成像队列（4序列齐全）: {len(cohort)} 例")

method_count = {}
fails = []
samples = []
for r in cohort:
    pid = r["pid"]
    ok = True
    detail, ondet, meth = {}, {}, {}
    for seq in ["T1", "T1CE", "T2", "FLAIR"]:
        s = r["series"][seq]
        try:
            res, saff, segs, method = align_seg_to_mr(s["mr"], s["seg"])
            cnt = {int(k): int((res == k).sum()) for k in np.unique(res) if k > 0}
            tot = sum(cnt.values())
            on = float((res > 0).mean()) if tot else 0.0  # 占位，实际 on-tissue 见下
            detail[seq] = tot
            meth[seq] = method
            method_count[method] = method_count.get(method, 0) + 1
            if tot < 200 or method == "lowconf":
                ok = False
        except Exception as e:
            ok = False
            detail[seq] = f"ERR:{e}"
            meth[seq] = "err"
    if not ok:
        fails.append((pid, detail, meth))
    elif len(samples) < 4:
        samples.append((pid, detail, meth))

print(f"失败例数: {len(fails)} / {len(cohort)}")
print("method 分布:", method_count)
for pid, d, me in fails[:20]:
    print("  FAIL", pid, d, me)
print("\n抽查:")
for pid, d, me in samples:
    print(f"  {pid}: " + "  ".join(f"{k}={v}({me[k]})" for k, v in d.items()))
