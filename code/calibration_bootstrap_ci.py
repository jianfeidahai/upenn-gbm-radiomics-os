# -*- coding: utf-8 -*-
"""校准斜率/截距的患者级 bootstrap 95% 区间。

思路（与 run_boot_grouped.py 的患者级重抽样口径一致）：
  1. 复用已有的折外预测：以 20 次重复五折（按患者分组）交叉验证得到每例的折外预测，
     同一患者的 20 次预测取平均 → 每例 1 条记录（n=207），无伪重复；
  2. 以患者为单位有放回重抽样 B=200 次，M0 与 M1 使用同一重抽样样本（配对），
     每次重新拟合"观测 OS ~ 预测 OS"的线性回归，取 2.5/97.5 百分位作为 95% 区间，
     并给出两模型斜率之差（M1−M0）的区间。

注意：该区间刻画的是"患者抽样"带来的变异，不含重新拟合影像组学标签带来的变异；
      标签随训练折变化已由重复划分与特征稳定性分析另行描述。

输入：features_wide_pyrad_bin64.csv
输出：results_calibration_boot_bin64.json

用法：python calibration_bootstrap_ci.py
"""
import json
import os

import numpy as np

import analysis_a as A
import calibration_check as CAL
import model_utils as mu

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = "features_wide_pyrad_bin64.csv"


def per_patient_arrays(y_all, p_all, pid_all):
    """同一患者的重复折外预测折叠为 1 个值，返回 (y, pred, pids)。"""
    pid_all = np.concatenate([np.asarray(x, dtype=str) for x in pid_all])
    y_all = np.concatenate([np.asarray(x, float) for x in y_all])
    p_all = np.concatenate([np.asarray(x, float) for x in p_all])
    pids, inv = np.unique(pid_all, return_inverse=True)
    n = len(pids)
    psum = np.zeros(n)
    yv = np.zeros(n)
    cnt = np.zeros(n)
    for j in range(len(p_all)):
        psum[inv[j]] += p_all[j]
        yv[inv[j]] = y_all[j]
        cnt[inv[j]] += 1
    return yv, psum / cnt, pids


def _pack(slopes, inters, citls, B, seed):
    def ci(v, nd=3):
        return [round(float(np.percentile(v, 2.5)), nd),
                round(float(np.percentile(v, 97.5)), nd)]

    return {
        "B": B,
        "seed": seed,
        "slope_boot_mean": round(float(slopes.mean()), 3),
        "slope_boot_sd": round(float(slopes.std(ddof=1)), 3),
        "slope_ci95": ci(slopes),
        "p_slope_lt_1": round(float((slopes < 1).mean()), 3),
        "intercept_ci95": ci(inters, 1),
        "calibration_in_the_large_ci95": ci(citls, 2),
    }


def boot_ci_paired(y, p0, p1, B=200, seed=0):
    """M0 与 M1 共用同一重抽样样本，并给出斜率之差（M1−M0）的区间。"""
    rng = np.random.default_rng(seed)
    n = len(y)
    s0 = np.empty(B)
    i0 = np.empty(B)
    c0 = np.empty(B)
    s1 = np.empty(B)
    i1 = np.empty(B)
    c1 = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        r0 = CAL.calibration(y[idx], p0[idx])
        r1 = CAL.calibration(y[idx], p1[idx])
        s0[b] = r0["slope"]
        i0[b] = r0["intercept"]
        c0[b] = r0["calibration_in_the_large"]
        s1[b] = r1["slope"]
        i1[b] = r1["intercept"]
        c1[b] = r1["calibration_in_the_large"]
    d = s1 - s0
    return {
        "M0": _pack(s0, i0, c0, B, seed),
        "M1": _pack(s1, i1, c1, B, seed),
        "difference_M1_minus_M0": {
            "B": B,
            "seed": seed,
            "slope_diff_ci95": [round(float(np.percentile(d, 2.5)), 3),
                                round(float(np.percentile(d, 97.5)), 3)],
            "p_diff_lt_0": round(float((d < 0).mean()), 3),
        },
    }


def main(repeats=20, outer=5, k=10, seed=0, corr_threshold=0.85, csv=CSV,
         out="results_calibration_boot_bin64.json", B=200):
    df = mu.load_wide(csv)
    y_all, p0_all, p1_all, pid_all = [], [], [], []
    for r in range(repeats):
        _, _, _, yv, p0v, p1v, pidv, _ = A.nested_cv(
            df, outer=outer, k=k, seed=seed + r, corr_threshold=corr_threshold)
        y_all.append(yv)
        p0_all.append(p0v)
        p1_all.append(p1v)
        pid_all.append(pidv)

    y0, pp0, pids = per_patient_arrays(y_all, p0_all, pid_all)
    y1, pp1, _ = per_patient_arrays(y_all, p1_all, pid_all)
    if not np.array_equal(y0, y1):
        raise RuntimeError("M0 与 M1 的患者折外观测不一致")

    boot = boot_ci_paired(y0, pp0, pp1, B=B, seed=seed)
    res = {
        "endpoint": "continuous OS (days)",
        "basis": "per-patient averaged out-of-fold predictions "
                 "(mean over %d repeated 5-fold patient-grouped CV)" % repeats,
        "n_per_patient": int(len(pids)),
        "repeats": repeats,
        "outer_folds": outer,
        "k": k,
        "corr_threshold": corr_threshold,
        "resampling": "patient-level bootstrap with replacement "
                      "(same scheme as run_boot_grouped.py, B=%d)" % B,
        "note": "该区间刻画患者抽样变异，不含重新拟合标签带来的变异；"
                "标签随训练折变化由重复划分与特征稳定性分析另行描述。",
    }
    for name, pp in (("M0", pp0), ("M1", pp1)):
        point = CAL.calibration(y0, pp)
        res[name] = {
            "slope": round(point["slope"], 3),
            "intercept": round(point["intercept"], 1),
            "mean_observed": round(point["mean_observed"], 1),
            "mean_predicted": round(point["mean_predicted"], 1),
            "calibration_in_the_large": round(point["calibration_in_the_large"], 1),
            "bootstrap": boot[name],
        }

    p0 = CAL.calibration(y0, pp0)["slope"]
    p1 = CAL.calibration(y0, pp1)["slope"]
    res["difference_M1_minus_M0"] = dict(
        boot["difference_M1_minus_M0"],
        slope_diff_point=round(float(p1 - p0), 3))

    path = os.path.join(HERE, out)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    print("saved ->", path)


if __name__ == "__main__":
    main()
