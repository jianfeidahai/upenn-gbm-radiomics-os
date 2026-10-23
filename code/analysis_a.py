# -*- coding: utf-8 -*-
"""主分析 A：影像组学相对 MGMT+临床 的 OS 增量预测价值。

终点（两种口径，任选其一）：
  连续 OS（默认）：线性回归 R²。
  二分类 OS（--binary）：logistic AUC + 决策曲线 DCA。

对比：M0 = MGMT + age + sex + resection（临床+分子基线）
      M1 = M0 + 影像签名

指标：
  连续：ΔR²（M1-M0），嵌套 CV 点估计（可 --repeat 多次稳定）+ bootstrap 百分位 CI + pooled 部分 F。
  二分类：ΔAUC（M1-M0），嵌套 CV（可 --repeat）+ DCA net benefit（M0 vs M1 vs treat-all/none）。

修订（2026-10-21，回应模拟审稿意见）：
  S1 bootstrap_dr2 修正：有放回抽患者后，同一患者的副本共享同一分组标识，
     副本不再被拆到训练/测试两折（原实现把副本当作独立患者，形成"测试患者在
     训练集中"的泄漏，对参数更多的 M1 造成系统乐观偏倚）。新增 mode="oob"
     袋外评估口径。
  S4 新增 log_os=True（对 OS 取自然对数）；pooled_oof_metrics 增加 MAE。
  S6 新增 tune_alpha=True（岭回归 alpha 由训练折内 RidgeCV 选择）。
  C2 新增 include_gtr=False（剔除术后变量 GTR 的临床基准模型 M0′）。
  nested_cv 返回患者标识，用于把同一患者跨重复的折外预测折叠为一个值（S5）。
"""
import os, json, argparse, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
import model_utils as mu


def clinical_matrix(df, include_gtr=True, include_idh=False):
    clin = mu.clinical(df)
    cols = ["mgmt", "age", "sex"]
    if include_gtr:
        cols += ["resect", "resect_missing"]
    if include_idh and "idh1" in clin.columns:
        cols += ["idh1"]
    return clin[cols].values.astype(float)


def _shuffle_groups(groups, seed):
    groups = np.asarray(groups).astype(str)
    unique = np.unique(groups)
    shuffled = np.random.default_rng(seed).permutation(unique)
    rank = {group: i for i, group in enumerate(shuffled)}
    return np.asarray([rank[group] for group in groups])


def nested_cv(df, outer=5, k=10, seed=0, corr_threshold=0.85, log_os=False,
              include_gtr=True, include_idh=False, tune_alpha=False, alpha=1.0):
    """按患者分组的外层 5 折嵌套交叉验证。

    返回 8 元组 (r2_0, r2_1, nsel, y_oof, pred0_oof, pred1_oof, pid_oof, alpha_oof)。
    pid_oof 用于把同一患者跨重复的折外预测折叠为一个值，避免伪重复；
    log_os=True 时对 OS 取自然对数建模；include_gtr=False 时剔除术后变量 GTR。
    """
    R0 = df[mu.feature_cols(df)].values.astype(float)
    X0 = clinical_matrix(df, include_gtr=include_gtr, include_idh=include_idh)
    y = pd.to_numeric(df["days"]).values.astype(float)
    if log_os:
        y = np.log(np.maximum(y, 1.0))
    group_col = "_boot_group" if "_boot_group" in df.columns else "pid"
    groups = df[group_col].astype(str).values
    pids = df["pid"].astype(str).values
    kf = GroupKFold(n_splits=outer)
    split_groups = _shuffle_groups(groups, seed)
    r2_0, r2_1, nsel, alphas = [], [], [], []
    y_oof, pred0_oof, pred1_oof, pid_oof = [], [], [], []
    for fold, (tr, te) in enumerate(kf.split(R0, y, split_groups)):
        Rtr, stats = mu.scale_train(R0[tr])
        Rte = mu.scale_test(R0[te], stats)
        sel, final_model = mu.fit_signature(
            Rtr, y[tr], k=k, outcome="continuous", corr_threshold=corr_threshold,
            alpha=alpha, tune_alpha=tune_alpha)
        sig_tr = mu.crossfit_signature_train(
            R0[tr], y[tr], groups[tr], k=k, splits=5, seed=seed + fold,
            outcome="continuous", corr_threshold=corr_threshold, alpha=alpha,
            tune_alpha=tune_alpha)
        sig_te = final_model.predict(Rte[:, sel])
        valid_tr = np.isfinite(sig_tr)
        Xtr = np.c_[X0[tr][valid_tr], np.ones(valid_tr.sum())]
        Xte = np.c_[X0[te], np.ones(len(te))]
        ytr = y[tr][valid_tr]
        m0 = LinearRegression().fit(Xtr, ytr)
        m1 = LinearRegression().fit(np.c_[Xtr, sig_tr[valid_tr]], ytr)
        pred0 = m0.predict(Xte)
        pred1 = m1.predict(np.c_[Xte, sig_te])
        r2_0.append(m0.score(Xte, y[te]))
        r2_1.append(m1.score(np.c_[Xte, sig_te], y[te]))
        y_oof.extend(y[te].tolist())
        pred0_oof.extend(pred0.tolist())
        pred1_oof.extend(pred1.tolist())
        pid_oof.extend(pids[te].tolist())
        nsel.append(len(sel))
        alphas.append(float(getattr(final_model, "alpha_",
                                    getattr(final_model, "alpha", alpha))))
    return (np.array(r2_0), np.array(r2_1), np.array(nsel),
            np.asarray(y_oof), np.asarray(pred0_oof), np.asarray(pred1_oof),
            np.asarray(pid_oof), np.asarray(alphas))


def _oob_dr2(d_in, d_out, outer, k, seed, corr_threshold, log_os, include_gtr,
             include_idh, tune_alpha, alpha):
    """单个 bootstrap 重抽样的袋外(out-of-bag) ΔR²：仅在袋内拟合，仅在袋外评估。"""
    R_in = d_in[mu.feature_cols(d_in)].values.astype(float)
    R_out = d_out[mu.feature_cols(d_out)].values.astype(float)
    X_in = clinical_matrix(d_in, include_gtr=include_gtr, include_idh=include_idh)
    X_out = clinical_matrix(d_out, include_gtr=include_gtr, include_idh=include_idh)
    y_in = pd.to_numeric(d_in["days"]).values.astype(float)
    y_out = pd.to_numeric(d_out["days"]).values.astype(float)
    if log_os:
        y_in = np.log(np.maximum(y_in, 1.0))
        y_out = np.log(np.maximum(y_out, 1.0))
    Rtr, stats = mu.scale_train(R_in)
    Rte = mu.scale_test(R_out, stats)
    sel, final_model = mu.fit_signature(
        Rtr, y_in, k=k, outcome="continuous", corr_threshold=corr_threshold,
        alpha=alpha, tune_alpha=tune_alpha)
    sig_in = mu.crossfit_signature_train(
        R_in, y_in, d_in["pid"].astype(str).values, k=k, splits=outer, seed=seed,
        outcome="continuous", corr_threshold=corr_threshold, alpha=alpha,
        tune_alpha=tune_alpha)
    sig_out = final_model.predict(Rte[:, sel])
    m0 = LinearRegression().fit(np.c_[X_in, np.ones(len(X_in))], y_in)
    m1 = LinearRegression().fit(np.c_[X_in, sig_in], y_in)
    p0 = m0.predict(np.c_[X_out, np.ones(len(X_out))])
    p1 = m1.predict(np.c_[X_out, sig_out])
    sst = float(((y_out - y_out.mean()) ** 2).sum())
    if sst <= 0:
        return np.nan
    r2_0 = 1 - float(((y_out - p0) ** 2).sum()) / sst
    r2_1 = 1 - float(((y_out - p1) ** 2).sum()) / sst
    return r2_1 - r2_0


def bootstrap_dr2(df, B=200, outer=5, k=10, seed=0, corr_threshold=0.85, mode="grouped",
                  log_os=False, include_gtr=True, include_idh=False,
                  tune_alpha=False, alpha=1.0):
    """患者级 bootstrap 的 ΔR² 分布（B 次重抽样）。

    mode="grouped"（默认，修正版）：有放回抽患者后，同一患者的全部副本共享同一个
      分组标识，副本不会被拆到训练/测试两折，杜绝"测试患者恰在训练集中"的泄漏。
    mode="oob"：每次重抽样后在未被抽中的患者（袋外）上评估。
    两种模式下 M1（参数更多）都不再获得系统乐观偏倚，因此 ΔR² 与交叉验证可比。
    """
    rng = np.random.default_rng(seed)
    ids = df["pid"].astype(str).to_numpy()
    unique_ids = np.unique(ids)
    vals = []
    for b in range(B):
        sampled_ids = rng.choice(unique_ids, size=len(unique_ids), replace=True)
        if mode == "oob":
            in_bag = np.unique(sampled_ids)
            keep = np.isin(ids, in_bag)
            d_in = df.loc[keep].copy()
            d_out = df.loc[~keep].copy()
            if len(d_out) < outer or len(d_in) < outer:
                continue
            vals.append(float(_oob_dr2(d_in, d_out, outer, k, seed + b, corr_threshold,
                                       log_os, include_gtr, include_idh, tune_alpha, alpha)))
        else:
            parts = []
            for pid in sampled_ids:
                part = df.loc[ids == pid].copy()
                part["_boot_group"] = pid  # 副本与原始患者同组，不得跨折
                parts.append(part)
            d = pd.concat(parts, ignore_index=True)
            r = nested_cv(d, outer=outer, k=k, seed=seed + b,
                          corr_threshold=corr_threshold, log_os=log_os,
                          include_gtr=include_gtr, include_idh=include_idh,
                          tune_alpha=tune_alpha, alpha=alpha)
            vals.append(float((r[1] - r[0]).mean()))
    return np.array(vals)


def nested_cv_binary(df, outer=5, k=10, seed=0, ybin=None, corr_threshold=0.85,
                     include_gtr=True, include_idh=False):
    R0 = df[mu.feature_cols(df)].values.astype(float)
    X0 = clinical_matrix(df, include_gtr=include_gtr, include_idh=include_idh)
    groups = (df["_boot_group"] if "_boot_group" in df.columns else df["pid"]).astype(str).values
    kf = GroupKFold(n_splits=outer)
    split_groups = _shuffle_groups(groups, seed)
    auc0, auc1, P0, P1, Y = [], [], [], [], []
    for fold, (tr, te) in enumerate(kf.split(R0, ybin, split_groups)):
        Rtr, stats = mu.scale_train(R0[tr])
        Rte = mu.scale_test(R0[te], stats)
        sel, final_model = mu.fit_signature(
            Rtr, ybin[tr], k=k, outcome="binary", corr_threshold=corr_threshold)
        sig_tr = mu.crossfit_signature_train(
            R0[tr], ybin[tr], groups[tr], k=k, splits=5, seed=seed + fold,
            outcome="binary", corr_threshold=corr_threshold)
        sig_te = final_model.predict_proba(Rte[:, sel])[:, 1]
        valid_tr = np.isfinite(sig_tr)
        Xtr = np.c_[X0[tr][valid_tr], np.ones(valid_tr.sum())]
        Xte = np.c_[X0[te], np.ones(len(te))]
        ytr = ybin[tr][valid_tr].astype(int)
        m0 = LogisticRegression(max_iter=5000).fit(Xtr, ytr)
        m1 = LogisticRegression(max_iter=5000).fit(np.c_[Xtr, sig_tr[valid_tr]], ytr)
        p0 = m0.predict_proba(Xte)[:, 1]
        p1 = m1.predict_proba(np.c_[Xte, sig_te])[:, 1]
        auc0.append(roc_auc_score(ybin[te], p0))
        auc1.append(roc_auc_score(ybin[te], p1))
        P0.append(p0); P1.append(p1); Y.append(ybin[te])
    return (np.array(auc0), np.array(auc1),
            np.concatenate(P0), np.concatenate(P1), np.concatenate(Y))


def dca(y, p, pts):
    """决策曲线：给定阈值概率 pts，返回 net benefit（按总样本归一）。"""
    y = np.asarray(y, float); p = np.asarray(p, float)
    prev = y.mean()
    nb = []
    for t in pts:
        if t <= 0 or t >= 1:
            nb.append(0.0); continue
        pred = (p >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        nb.append((tp - fp * (t / (1 - t))) / len(y))
    nb_all = prev - (1 - prev) * (pts / (1 - pts))
    nb_none = np.zeros_like(pts)
    return np.array(nb), nb_all, nb_none


def pooled_oof_metrics(y, pred0, pred1):
    y = np.asarray(y, float)
    pred0 = np.asarray(pred0, float)
    pred1 = np.asarray(pred1, float)
    sst = float(((y - y.mean()) ** 2).sum())
    sse0 = float(((y - pred0) ** 2).sum())
    sse1 = float(((y - pred1) ** 2).sum())
    return {"R2_M0": 1 - sse0 / sst,
            "R2_M1": 1 - sse1 / sst,
            "dR2": (sse0 - sse1) / sst,
            "rmse_M0": float(np.sqrt(np.mean((y - pred0) ** 2))),
            "rmse_M1": float(np.sqrt(np.mean((y - pred1) ** 2))),
            "mae_M0": float(np.mean(np.abs(y - pred0))),
            "mae_M1": float(np.mean(np.abs(y - pred1)))}


def main_cli(df, repeat=1, boot=0, boot_mode="grouped", k=10, corr_threshold=0.85,
             seed=0, outer=5, log_os=False, include_gtr=True, include_idh=False,
             tune_alpha=False, csv_name=""):
    """连续 OS 主分析（可直接调用的函数形式）。"""
    r0_all, r1_all, dr_all, nsel_all, alpha_all = [], [], [], [], []
    y_oof, pred0_oof, pred1_oof = [], [], []
    for r in range(repeat):
        r0, r1, nsel, yv, p0v, p1v, pidv, alphav = nested_cv(
            df, outer, k, seed + r, corr_threshold=corr_threshold, log_os=log_os,
            include_gtr=include_gtr, include_idh=include_idh, tune_alpha=tune_alpha)
        r0_all.append(r0.mean()); r1_all.append(r1.mean())
        dr_all.append((r1 - r0).mean()); nsel_all.append(nsel.mean())
        alpha_all.append(alphav.mean())
        y_oof.append(yv); pred0_oof.append(p0v); pred1_oof.append(p1v)
    dr_all = np.array(dr_all)
    res = {"endpoint": "log(OS) (days)" if log_os else "continuous OS (days)",
           "signature_config": {"k": k, "corr_threshold": corr_threshold,
                                "alpha": 1.0, "tune_alpha": bool(tune_alpha),
                                "include_gtr": bool(include_gtr),
                                "include_idh": bool(include_idh),
                                "log_os": bool(log_os),
                                "data_file": csv_name},
           "alpha_mean_selected": float(np.mean(alpha_all)),
           "point": {"R2_M0_mean": float(np.mean(r0_all)), "R2_M1_mean": float(np.mean(r1_all)),
                     "dR2_mean": float(dr_all.mean()),
                     "n_selected_mean": float(np.mean(nsel_all))},
           "repeat": {"R": repeat, "dR2_mean": float(dr_all.mean()),
                      "dR2_sd": float(np.std(dr_all)),
                      "dR2_ci95": [float(np.percentile(dr_all, 2.5)),
                                   float(np.percentile(dr_all, 97.5))]}}
    pooled_results = [pooled_oof_metrics(yv, p0, p1)
                      for yv, p0, p1 in zip(y_oof, pred0_oof, pred1_oof)]
    res["pooled_oof"] = {key: float(np.mean([item[key] for item in pooled_results]))
                         for key in pooled_results[0]}
    if boot > 0:
        b = bootstrap_dr2(df, boot, outer, k, seed, corr_threshold, mode=boot_mode,
                          log_os=log_os, include_gtr=include_gtr, include_idh=include_idh,
                          tune_alpha=tune_alpha)
        lo, hi = np.percentile(b, [2.5, 97.5])
        res["bootstrap"] = {"B": boot, "mode": boot_mode,
                            "dR2_mean": float(b.mean()),
                            "dR2_ci95": [float(lo), float(hi)],
                            "p_dR2_gt_0": float((b > 0).mean()),
                            "n_extreme_abs_gt_0p5": int((np.abs(b) > 0.5).sum()),
                            "dR2_min": float(b.min()), "dR2_max": float(b.max())}
    return res


def main_cli_binary(df, repeat=1, thresh=456.0, k=10, corr_threshold=0.85, seed=0,
                    outer=5, include_gtr=True, include_idh=False, csv_name=""):
    """二分类 OS 分析（可直接调用的函数形式）。"""
    days = pd.to_numeric(df["days"]).values.astype(float)
    ybin = (days <= thresh).astype(int)
    auc0s, auc1s, dr = [], [], []
    P0 = P1 = Y = None
    for r in range(repeat):
        a0, a1, p0, p1, yv = nested_cv_binary(
            df, outer, k, seed + r, ybin=ybin, corr_threshold=corr_threshold,
            include_gtr=include_gtr, include_idh=include_idh)
        auc0s.append(a0.mean()); auc1s.append(a1.mean()); dr.append(a1.mean() - a0.mean())
        if r == 0:
            P0, P1, Y = p0, p1, yv
    dr = np.array(dr)
    res = {"endpoint": "binary OS", "threshold_days": float(thresh),
           "event_rate": float(ybin.mean()),
           "signature_config": {"k": k, "corr_threshold": corr_threshold,
                                "alpha": 1.0, "include_gtr": bool(include_gtr),
                                "include_idh": bool(include_idh),
                                "data_file": csv_name},
           "repeat": repeat,
           "AUC_M0_mean": float(np.mean(auc0s)), "AUC_M1_mean": float(np.mean(auc1s)),
           "dAUC_mean": float(np.mean(dr)),
           "AUC_M0_sd": float(np.std(auc0s)), "AUC_M1_sd": float(np.std(auc1s)),
           "dAUC_sd": float(np.std(dr)),
           "dAUC_ci95": [float(np.percentile(dr, 2.5)), float(np.percentile(dr, 97.5))]}
    pts = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50])
    nb0, nb0_all, _ = dca(Y, P0, pts)
    nb1, nb1_all, _ = dca(Y, P1, pts)
    res["dca"] = {"thresholds": pts.tolist(), "NB_M0": nb0.tolist(),
                  "NB_M1": nb1.tolist(), "NB_treat_all": nb0_all.tolist(),
                  "NB_treat_none": [0.0] * len(pts)}
    res["dca_summary"] = {f"NB@0.10_M0": float(nb0[1]), f"NB@0.10_M1": float(nb1[1]),
                          f"NB@0.20_M0": float(nb0[3]), f"NB@0.20_M1": float(nb1[3])}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--outer", type=int, default=5)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--corr-threshold", type=float, default=0.85,
                    help="训练折内特征相关性去冗余阈值（主分析预设 0.85）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=1, help="重复嵌套CV次数（不同种子），稳定点估计")
    ap.add_argument("--binary", action="store_true", help="改用二分类 OS 终点(AUC+DCA)")
    ap.add_argument("--thresh", type=float, default=None, help="二分类阈值(天)；优先于 --thresh-mode")
    ap.add_argument("--thresh-mode", choices=["median", "months15"], default="months15")
    ap.add_argument("--no-gtr", action="store_true",
                    help="敏感性分析：剔除术后变量 GTR 后的临床基准模型 M0′")
    ap.add_argument("--log-os", action="store_true", help="对 OS 取自然对数建模")
    ap.add_argument("--include-idh", action="store_true",
                    help="敏感性分析：基准模型纳入 IDH1 突变状态（需含 idh1 列的数据文件）")
    ap.add_argument("--tune-alpha", action="store_true",
                    help="敏感性分析：岭回归 alpha 由训练折内 RidgeCV 选择")
    ap.add_argument("--boot-mode", choices=["grouped", "oob"], default="grouped",
                    help="bootstrap 评估方式：grouped=副本同组(默认修正版)；oob=袋外评估")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    df = mu.load_wide(args.csv)
    csv_name = os.path.basename(args.csv or mu.load_wide.__defaults__[0])
    print(f"[A] n={len(df)} 特征={len(mu.feature_cols(df))} k={args.k} "
          f"repeat={args.repeat} binary={args.binary} no_gtr={args.no_gtr} "
          f"log_os={args.log_os} tune_alpha={args.tune_alpha}")

    if args.binary:
        days = pd.to_numeric(df["days"]).values.astype(float)
        if args.thresh is not None:
            th = args.thresh
        elif args.thresh_mode == "median":
            th = float(np.median(days))
        else:
            th = 456.0  # 15 个月
        res = main_cli_binary(df, repeat=args.repeat, thresh=th, k=args.k,
                              corr_threshold=args.corr_threshold, seed=args.seed,
                              outer=args.outer, include_gtr=not args.no_gtr,
                              include_idh=args.include_idh, csv_name=csv_name)
    else:
        res = main_cli(df, repeat=args.repeat, boot=args.boot, boot_mode=args.boot_mode,
                       k=args.k, corr_threshold=args.corr_threshold, seed=args.seed,
                       outer=args.outer, log_os=args.log_os,
                       include_gtr=not args.no_gtr, include_idh=args.include_idh,
                       tune_alpha=args.tune_alpha, csv_name=csv_name)
    out = args.out or os.path.join(mu.HERE, "results_A_binary.json" if args.binary
                                   else "results_A.json")
    json.dump(res, open(out, "w"), indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    print("saved ->", out)


if __name__ == "__main__":
    main()
