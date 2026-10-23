# -*- coding: utf-8 -*-
"""审稿意见修订（2026-10-21）：一次性跑完全部敏感性分析。

对应审稿意见编号：
  S1  bootstrap 副本跨折泄漏 → 修正为副本同组(grouped)，并新增袋外(oob)评估
  S4  log(OS) 尺度分析 + RMSE/MAE
  S6  超参数敏感性（k、冗余阈值、alpha）
  C2  剔除术后变量 GTR 的临床基准模型 M0′
  C3  二分类 12/24 个月切点敏感性
  S5  校准按患者平均后计算（calibration_check.py 单独运行）

所有输出写入 results_*.json，主分析沿用 results_A_stability_main.json。
"""
import json
import os
import shutil
import time

import numpy as np

import analysis_a as A
import calibration_check as CAL
import model_utils as mu

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = "features_wide_pyrad.csv"
REPEAT = 10
BOOT = 200


def stamp(msg, t0):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}  (elapsed {time.time() - t0:.0f}s)", flush=True)


def _save(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print("saved ->", path, flush=True)


def main():
    t0 = time.time()
    df = mu.load_wide(CSV)
    n_feat = len(mu.feature_cols(df))
    print(f"[driver] n={len(df)} 特征={n_feat} repeat={REPEAT} boot={BOOT}", flush=True)

    # ---------- 0. 自校验：未加新参数前的点估计应与历史主结果一致 ----------
    r0, r1, nsel, yv, p0, p1, pidv, alphav = A.nested_cv(df, outer=5, k=10, seed=0,
                                                         corr_threshold=0.85)
    hist = json.load(open(os.path.join(HERE, "results_A_stability_main.json"), encoding="utf-8"))
    print(f"[selfcheck] seed0 fold-mean R2 M0={r0.mean():.6f} M1={r1.mean():.6f} "
          f"(历史主结果 repeat 均值 M0={hist['point']['R2_M0_mean']:.6f})", flush=True)

    # ---------- 1. 主分析（修正版 bootstrap，grouped） ----------
    stamp("主分析：修正版 bootstrap grouped", t0)
    res_main = A.main_cli(df, repeat=REPEAT, boot=BOOT, boot_mode="grouped",
                          corr_threshold=0.85, csv_name=CSV)
    _save(os.path.join(HERE, "results_A_stability_main.json"), res_main)

    # ---------- 2. 主分析（bootstrap 袋外口径，作为对照） ----------
    stamp("主分析：bootstrap oob", t0)
    res_oob = A.main_cli(df, repeat=REPEAT, boot=BOOT, boot_mode="oob",
                         corr_threshold=0.85, csv_name=CSV)
    _save(os.path.join(HERE, "results_A_bootstrap_oob.json"), res_oob)

    # ---------- 3. log(OS) 尺度 ----------
    stamp("log(OS) 尺度分析", t0)
    res_log = A.main_cli(df, repeat=REPEAT, boot=BOOT, boot_mode="grouped",
                         corr_threshold=0.85, log_os=True, csv_name=CSV)
    _save(os.path.join(HERE, "results_A_logos.json"), res_log)

    # ---------- 4. 剔除术后 GTR（M0′） ----------
    stamp("剔除 GTR 的 M0′ 敏感性分析", t0)
    res_nogtr = A.main_cli(df, repeat=REPEAT, boot=0, include_gtr=False,
                           corr_threshold=0.85, csv_name=CSV)
    _save(os.path.join(HERE, "results_A_nogtr.json"), res_nogtr)

    # ---------- 5. alpha 由内层 RidgeCV 选择 ----------
    stamp("alpha 内层选择敏感性分析", t0)
    res_alpha = A.main_cli(df, repeat=REPEAT, boot=0, tune_alpha=True,
                           corr_threshold=0.85, csv_name=CSV)
    _save(os.path.join(HERE, "results_A_tunealpha.json"), res_alpha)

    # ---------- 6. k 与冗余阈值网格 ----------
    grid = []
    for k in (5, 20):
        for th in (0.80, 0.90):
            stamp(f"网格 k={k} 阈值={th}", t0)
            r = A.main_cli(df, repeat=REPEAT, boot=0, k=k, corr_threshold=th, csv_name=CSV)
            grid.append({"k": k, "corr_threshold": th,
                         "dR2_mean": r["repeat"]["dR2_mean"],
                         "dR2_sd": r["repeat"]["dR2_sd"],
                         "dR2_ci95": r["repeat"]["dR2_ci95"],
                         "R2_M0_mean": r["point"]["R2_M0_mean"],
                         "R2_M1_mean": r["point"]["R2_M1_mean"],
                         "pooled_dR2": r["pooled_oof"]["dR2"],
                         "rmse_M0": r["pooled_oof"]["rmse_M0"],
                         "rmse_M1": r["pooled_oof"]["rmse_M1"]})
    _save(os.path.join(HERE, "results_A_sens_grid.json"), {"grid": grid, "repeats": REPEAT})

    # ---------- 7. 二分类 12 / 24 个月切点 ----------
    for months, th in ((12, 365.0), (24, 730.0)):
        stamp(f"二分类 {months} 个月切点", t0)
        rb = A.main_cli_binary(df, repeat=REPEAT, thresh=th, csv_name=CSV)
        _save(os.path.join(HERE, f"results_A_binary_{months}m.json"), rb)

    stamp("全部完成", t0)


if __name__ == "__main__":
    main()
