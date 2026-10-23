# -*- coding: utf-8 -*-
"""Pooled out-of-fold calibration assessment for M0 and M1 (primary endpoint).

按审稿意见 S5 修正：同一患者在 10 次重复划分中各出现一次折外预测，直接合并会形成
伪重复。此处同时给出两种口径：
  1) per_patient：先把同一患者的 10 次折外预测取平均（n=207），再回归 → 无伪重复，
     作为正文报告值；
  2) pooled：10 次重复共 2070 条记录的合并口径，仅作对照，不附带区间估计。
"""
import json
import os

import numpy as np

import analysis_a as A
import model_utils as mu

HERE = os.path.dirname(os.path.abspath(__file__))


def calibration(y, pred):
    y = np.asarray(y, float)
    pred = np.asarray(pred, float)
    design = np.column_stack([pred, np.ones(len(pred))])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    intercept = float(coef[1])
    slope = float(coef[0])
    pred_mean = float(pred.mean())
    obs_mean = float(y.mean())
    return {
        "slope": slope,
        "intercept": intercept,
        "mean_observed": obs_mean,
        "mean_predicted": pred_mean,
        "calibration_in_the_large": obs_mean - pred_mean,
    }


def per_patient_calibration(y_all, p_all, pid_all):
    """把同一患者的重复折外预测折叠为 1 个值后再回归，消除伪重复。"""
    pid_all = np.concatenate(pid_all)
    y_all = np.concatenate(y_all)
    p_all = np.concatenate(p_all)
    pids, inv = np.unique(pid_all, return_inverse=True)
    n = len(pids)
    p_mean = np.zeros(n)
    y_first = np.zeros(n)
    counts = np.zeros(n)
    for j in range(len(p_all)):
        p_mean[inv[j]] += p_all[j]
        counts[inv[j]] += 1
        y_first[inv[j]] = y_all[j]
    p_mean /= counts
    return calibration(y_first, p_mean), int(n)


def main(repeats=10, outer=5, k=10, seed=0, corr_threshold=0.85, csv="features_wide_pyrad.csv",
         out="results_calibration_main.json"):
    df = mu.load_wide(csv)
    y_all, p0_all, p1_all, pid_all = [], [], [], []
    for r in range(repeats):
        _, _, _, yv, p0v, p1v, pidv, _ = A.nested_cv(
            df, outer=outer, k=k, seed=seed + r, corr_threshold=corr_threshold)
        y_all.append(yv)
        p0_all.append(p0v)
        p1_all.append(p1v)
        pid_all.append(pidv)

    y = np.concatenate(y_all)
    p0 = np.concatenate(p0_all)
    p1 = np.concatenate(p1_all)

    m0_pool = calibration(y, p0)
    m1_pool = calibration(y, p1)
    m0_pp, n_pp = per_patient_calibration(y_all, p0_all, pid_all)
    m1_pp, _ = per_patient_calibration(y_all, p1_all, pid_all)

    res = {
        "endpoint": "continuous OS (days)",
        "repeats": repeats,
        "outer_folds": outer,
        "k": k,
        "corr_threshold": corr_threshold,
        "note": "per_patient 口径按患者平均折外预测后回归(n=207)，无伪重复，为正文报告值；"
                "pooled 口径含 10 次重复的 2070 条记录，仅作对照。",
        "n_pooled_predictions": int(len(y)),
        "n_per_patient": int(n_pp),
        "M0": {"per_patient": m0_pp, "pooled": m0_pool},
        "M1": {"per_patient": m1_pp, "pooled": m1_pool},
    }
    path = os.path.join(HERE, out)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    print("saved ->", path)


if __name__ == "__main__":
    main()
