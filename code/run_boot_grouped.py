# -*- coding: utf-8 -*-
"""单独重跑“副本同组”bootstrap（B=200）以量化其不稳定性，并写回主结果文件。"""
import json
import os

import numpy as np

import analysis_a as A
import model_utils as mu

HERE = os.path.dirname(os.path.abspath(__file__))
df = mu.load_wide("features_wide_pyrad.csv")

b = A.bootstrap_dr2(df, B=200, outer=5, k=10, seed=0, corr_threshold=0.85, mode="grouped")
lo, hi = np.percentile(b, [2.5, 97.5])
print("grouped B=200 mean=%.5f ci=[%.5f, %.5f] p>0=%.3f min=%.4f max=%.4f extreme=%.2f%%"
      % (b.mean(), lo, hi, (b > 0).mean(), b.min(), b.max(),
         100.0 * (np.abs(b) > 0.5).mean()))
print("quantiles 1/5/25/50/75/95/99:",
      np.round(np.percentile(b, [1, 5, 25, 50, 75, 95, 99]), 4))

path = os.path.join(HERE, "results_A_stability_main.json")
res = json.load(open(path, encoding="utf-8"))
res["bootstrap"] = {"B": 200, "mode": "grouped", "dR2_mean": float(b.mean()),
                    "dR2_ci95": [float(lo), float(hi)],
                    "p_dR2_gt_0": float((b > 0).mean()),
                    "n_extreme_abs_gt_0p5": int((np.abs(b) > 0.5).sum()),
                    "dR2_min": float(b.min()), "dR2_max": float(b.max()),
                    "note": "重抽样后在 bootstrap 样本内重新分组交叉验证的口径；测试折含同一患者的重复副本，"
                            "折内结局方差可趋近零而使 R² 失控，故不作为推断依据（正文报告袋外口径）。"}
with open(path, "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
np.save(os.path.join(HERE, "results_A_bootstrap_grouped_vals.npy"), b)
print("saved ->", path)
