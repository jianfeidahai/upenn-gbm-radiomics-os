# -*- coding: utf-8 -*-
"""辅分析 B：影像–生存关联的 MGMT 轴分解（关联分解，非因果）+ 亚区解析。

注意：影像与 MGMT 同属肿瘤生物学下游效应，"影像→MGMT→OS"不是合法因果链；
本分析仅做**关联分解**——量化影像对 OS 的关联中有多大比例与 MGMT 信号"重叠/共享"。

方法（差异法 / 关联比例中介 PM）：
  β_t  : OS ~ 影像签名  （影像→OS 总关联）
  β_d  : OS ~ 影像签名 + MGMT （控制 MGMT 后影像→OS 直接关联）
  PM   = (β_t - β_d)/β_t  （与 MGMT 信号重叠的关联占比）
  AUC  : MGMT ~ 影像签名 （影像预测 MGMT 的分类效能，α 路径）
  同时报告 MGMT→OS 系数及其在加入影像后的变化。
亚区解析：对 12 个 (序列×亚区) 块 + 全特征(ALL) 分别分解，定位贡献 MGMT 相关信号的亚区。
bootstrap 百分位 CI。
"""
import os, json, argparse, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
import model_utils as mu


def mediation(df, cols, k=10, outer=5, seed=0, B=200, corr_threshold=0.85):
    clin = mu.clinical(df)
    mgmt_all = clin["mgmt"].values.astype(float)
    y_all = pd.to_numeric(df["days"]).values.astype(float)
    group_col = "_boot_group" if "_boot_group" in df else "pid"

    def signatures(sample_df, run_seed):
        sig_os = mu.cv_signature(sample_df, cols, "days", k=k, outer=outer,
                                 seed=run_seed, group_col=group_col,
                                 corr_threshold=corr_threshold)
        sig_mgmt = mu.cv_signature(sample_df, cols, "mgmt", k=k, outer=outer,
                                   seed=run_seed + 10000, group_col=group_col,
                                   corr_threshold=corr_threshold)
        return sig_os, sig_mgmt

    def estimate(s_os, s_mgmt, mgmt, yy):
        good = np.isfinite(s_os) & np.isfinite(s_mgmt) & np.isfinite(mgmt) & np.isfinite(yy)
        s, m, y = s_os[good], mgmt[good], yy[good]
        Xt = np.c_[s, np.ones(len(s))]
        bt = np.linalg.lstsq(Xt, y, rcond=None)[0][0]
        Xd = np.c_[s, m, np.ones(len(s))]
        bd = np.linalg.lstsq(Xd, y, rcond=None)[0][0]
        Xm = np.c_[m, np.ones(len(s))]
        bm = np.linalg.lstsq(Xm, y, rcond=None)[0][0]
        try:
            auc = roc_auc_score(m, LogisticRegression(max_iter=5000).fit(s.reshape(-1, 1), m)
                                .predict_proba(s.reshape(-1, 1))[:, 1])
        except Exception:
            auc = np.nan
        pm = (bt - bd) / bt if abs(bt) > 1e-9 else np.nan
        return {"n": int(len(y)), "beta_total": float(bt), "beta_direct": float(bd),
                "beta_MGMT": float(bm), "AUC_img2MGMT": float(auc), "PM": float(pm),
                "sig_os_std": float(np.std(s)),
                "corr_sigOS_days": float(np.corrcoef(s, y)[0, 1]),
                "corr_sigMGMT_mgmt": float(np.corrcoef(s_mgmt[good], m)[0, 1])}

    sig_os, sig_mgmt = signatures(df, seed)
    res = estimate(sig_os, sig_mgmt, mgmt_all, y_all)
    group_ids = df[group_col].astype(str).to_numpy() if group_col in df else np.arange(len(df)).astype(str)
    unique_groups = np.unique(group_ids)

    if B > 0:
        rng = np.random.default_rng(seed)
        samples = {name: [] for name in ["beta_total", "beta_direct", "beta_MGMT", "AUC_img2MGMT", "PM"]}
        for b in range(B):
            draw = rng.choice(unique_groups, size=len(unique_groups), replace=True)
            parts = []
            for j, original_id in enumerate(draw):
                part = df.loc[group_ids == original_id].copy()
                part["_boot_group"] = f"boot_{j}"
                parts.append(part)
            boot_df = pd.concat(parts, ignore_index=True)
            boot_clin = mu.clinical(boot_df)
            boot_mgmt = boot_clin["mgmt"].values.astype(float)
            boot_y = pd.to_numeric(boot_df["days"]).values.astype(float)
            boot_os, boot_mgmt_sig = signatures(boot_df, seed + b + 1)
            est = estimate(boot_os, boot_mgmt_sig, boot_mgmt, boot_y)
            for name in samples:
                samples[name].append(est[name])
        for name, values in samples.items():
            arr = np.asarray(values, float)
            lo, hi = np.nanpercentile(arr, [2.5, 97.5])
            res[name + "_ci95"] = [float(lo), float(hi)]
        bt_lo, bt_hi = res["beta_total_ci95"]
        pm_valid = np.isfinite(res["PM"]) and (bt_lo * bt_hi > 0)
        if not pm_valid:
            res["PM"] = None
            res["PM_ci95"] = None
            res["PM_valid"] = False
        else:
            res["PM_valid"] = True
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--outer", type=int, default=5)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--corr-threshold", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    df = mu.load_wide(args.csv)
    blocks = mu.get_blocks(df)
    blocks["ALL"] = mu.feature_cols(df)
    print(f"[B] n={len(df)} blocks={len(blocks)} k={args.k} boot={args.boot}")

    out = {"note": "associational decomposition (NOT causal); PM indexes overlap of imaging and MGMT "
                   "signals w.r.t. OS", "signature_config": {"k": args.k,
                   "corr_threshold": args.corr_threshold, "alpha": 1.0}, "blocks": {}}
    for name, cols in blocks.items():
        print("  block", name, "nfeat", len(cols))
        out["blocks"][name] = mediation(df, cols, k=args.k, outer=args.outer,
                                        seed=args.seed, B=args.boot,
                                        corr_threshold=args.corr_threshold)

    path = args.out or os.path.join(mu.HERE, "results_B.json")
    json.dump(out, open(path, "w"), indent=2, ensure_ascii=False)
    # 简要打印 PM 排序
    rows = []
    for name, r in out["blocks"].items():
        pm = r.get("PM", None)
        rows.append((name, (pm if pm is not None else float("nan")),
                     r.get("AUC_img2MGMT", float("nan")), r.get("beta_total", float("nan"))))
    rows.sort(key=lambda x: (np.isnan(x[1]), -x[1] if not np.isnan(x[1]) else 0))
    print("\nPM (影像→OS 关联中与 MGMT 重叠占比，β_t 不可靠时记 None) 排序：")
    for name, pm, auc, bt in rows:
        pmstr = "None" if np.isnan(pm) else f"{pm:.3f}"
        print(f"  {name:10s} PM={pmstr:>6s}  AUC_img2MGMT={auc:.3f}  beta_total={bt:.3f}")
    print("saved ->", path)


if __name__ == "__main__":
    main()
