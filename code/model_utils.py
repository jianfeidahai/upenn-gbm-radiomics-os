# -*- coding: utf-8 -*-
"""建模共用工具：数据加载、临床编码、缺失填补/标准化、LASSO 影像签名拟合。

特征选择原则：训练折内 LASSO 锁特征（论文 §4.2），杜绝信息泄漏。
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.model_selection import KFold, GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
PREFIXES = ("T1_", "T1CE_", "T2_", "FLAIR_")


def load_wide(path=None):
    if path is None:
        path = os.path.join(HERE, "features_wide.csv")
    return pd.read_csv(path, encoding="utf-8-sig")


def feature_cols(df):
    return [c for c in df.columns if c.startswith(PREFIXES)]


def clinical(df):
    mgmt_map = {"Methylated": 1.0, "Unmethylated": 0.0}
    out = pd.DataFrame(index=df.index)
    out["mgmt"] = df["mgmt"].map(mgmt_map)
    out["age"] = pd.to_numeric(df["age"], errors="coerce")
    out["sex"] = (df["gender"].astype(str).str.upper() == "M").astype(float)
    g = df["gtr"].astype(str).str.upper()
    out["resect"] = (g == "Y").astype(float)
    out["resect_missing"] = (~g.isin(["Y", "N"])).astype(float)
    if "idh1" in df.columns:  # 敏感性分析：IDH1 突变状态（源数据完整）
        out["idh1"] = (df["idh1"].astype(str) == "Mutated").astype(float)
    return out


def get_blocks(df):
    """按 '序列_亚区' 拆分特征列（4 序列 × 3 亚区 = 12 块）。"""
    cols = feature_cols(df)
    blocks = {}
    for c in cols:
        parts = c.split("_", 2)
        blk = parts[0] + "_" + parts[1] if len(parts) >= 2 else c
        blocks.setdefault(blk, []).append(c)
    return blocks


def scale_train(R, med=None, mu=None, sd=None):
    if med is None:
        med = np.nanmedian(R, axis=0)
    Rf = np.where(np.isnan(R), med, R)
    if mu is None:
        mu = Rf.mean(0)
        sd = Rf.std(0)
    sd = np.where(sd == 0, 1.0, sd)
    return (Rf - mu) / sd, (med, mu, sd)


def scale_test(R, stats):
    med, mu, sd = stats
    Rf = np.where(np.isnan(R), med, R)
    return (Rf - mu) / sd


def fit_signature(Rtr, ytr, k=10, outcome="continuous", corr_threshold=None,
                  alpha=1.0, tune_alpha=False):
    """训练折内单变量 |相关| 初筛 top-k，再用岭回归(连续)/逻辑回归(二分类) 拟合成签名。

    不用 LASSO-CV 直接选 alpha：弱信号下其交叉验证会把签名压缩为全零（alpha 过大），
    反而制造"无信号"假象。单变量初筛 + 岭回归是影像组学常用且稳健的做法。
    tune_alpha=True 时连续终点由内层 RidgeCV 选择 alpha（仍仅在训练折内）。
    返回 (selected_indices, model)。
    """
    ytr = np.asarray(ytr, float)
    corr = np.array([np.corrcoef(Rtr[:, j], ytr)[0, 1] for j in range(Rtr.shape[1])])
    corr = np.nan_to_num(corr)
    order = np.argsort(-np.abs(corr), kind="stable")
    if corr_threshold is None:
        sel = order[: min(k, Rtr.shape[1])]
    else:
        abs_corr = np.abs(np.corrcoef(Rtr, rowvar=False))
        abs_corr = np.nan_to_num(abs_corr, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(abs_corr, 1.0)
        picked = []
        for idx in order:
            if all(abs_corr[idx, prev] < corr_threshold for prev in picked):
                picked.append(int(idx))
                if len(picked) >= min(k, Rtr.shape[1]):
                    break
        sel = np.asarray(picked, dtype=int)
    if outcome == "continuous":
        from sklearn.linear_model import Ridge, RidgeCV
        if tune_alpha:
            m = RidgeCV(alphas=np.logspace(-1, 3, 9)).fit(Rtr[:, sel], ytr)
        else:
            m = Ridge(alpha=alpha).fit(Rtr[:, sel], ytr)
    else:
        from sklearn.linear_model import LogisticRegression
        m = LogisticRegression(max_iter=5000).fit(Rtr[:, sel], ytr.astype(int))
    return sel, m


def cv_signature(df, cols, target, k=10, outer=5, seed=0, group_col="pid",
                 corr_threshold=None, alpha=1.0, tune_alpha=False):
    """返回 out-of-fold 影像签名；同一患者的重复行按 group_col 保持在同一折。"""
    R0 = df[cols].values.astype(float)
    if target == "days":
        y = pd.to_numeric(df[target]).values.astype(float)
        outcome = "continuous"
    else:
        y = clinical(df)["mgmt"].values.astype(float)
        outcome = "binary"
    n = len(df)
    sig = np.full(n, np.nan)
    groups = df[group_col].astype(str).values if group_col in df else np.arange(n)
    n_splits = min(outer, len(np.unique(groups)))
    if n_splits < 2:
        raise ValueError("At least two unique groups are required for grouped CV")
    kf = GroupKFold(n_splits=n_splits)
    for tr, te in kf.split(R0, y, groups):
        Rtr, stats = scale_train(R0[tr])
        Rte = scale_test(R0[te], stats)
        sel, m = fit_signature(Rtr, y[tr], k=k, outcome=outcome,
                               corr_threshold=corr_threshold, alpha=alpha,
                               tune_alpha=tune_alpha)
        sig[te] = m.predict(Rte[:, sel])
    return sig


def crossfit_signature_train(R, y, groups, k=10, splits=5, seed=0, outcome="continuous",
                             corr_threshold=None, alpha=1.0, tune_alpha=False):
    """Generate training predictions without fitting a row's model on that row."""
    R = np.asarray(R, float)
    y = np.asarray(y, float)
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)
    n_splits = min(splits, len(unique_groups))
    if n_splits < 2:
        raise ValueError("At least two unique groups are required for cross-fitting")
    splitter = GroupKFold(n_splits=n_splits)
    group_order = np.unique(groups)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(group_order)
    rank = {group: i for i, group in enumerate(shuffled)}
    split_groups = np.asarray([rank[group] for group in groups])
    sig = np.full(len(y), np.nan)
    for tr, te in splitter.split(R, y, split_groups):
        Rtr, stats = scale_train(R[tr])
        Rte = scale_test(R[te], stats)
        sel, model = fit_signature(Rtr, y[tr], k=k, outcome=outcome,
                                   corr_threshold=corr_threshold, alpha=alpha,
                                   tune_alpha=tune_alpha)
        if outcome == "binary":
            sig[te] = model.predict_proba(Rte[:, sel])[:, 1]
        else:
            sig[te] = model.predict(Rte[:, sel])
    return sig
