# -*- coding: utf-8 -*-
"""Patient-grouped repeated-CV stability audit for the top-k radiomics signature."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import model_utils as mu


SCRIPT_SHA256 = None
MODEL_UTILS_SHA256 = None


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def shuffled_group_labels(groups, seed):
    unique = np.unique(groups.astype(str))
    permuted = np.random.default_rng(seed).permutation(unique)
    rank = {group: i for i, group in enumerate(permuted)}
    return np.asarray([rank[group] for group in groups.astype(str)])


def run(data_path, out_path, repeats=10, outer=5, k=10, seed=0, corr_threshold=0.85):
    df = mu.load_wide(data_path)
    cols = mu.feature_cols(df)
    X = df[cols].to_numpy(dtype=float)
    y = pd.to_numeric(df["days"], errors="coerce").to_numpy(dtype=float)
    groups = df["pid"].astype(str).to_numpy()
    n_splits = min(outer, len(np.unique(groups)))
    if n_splits < 2:
        raise ValueError("At least two unique patient groups are required")

    selected_count = np.zeros(len(cols), dtype=int)
    positive_count = np.zeros(len(cols), dtype=int)
    negative_count = np.zeros(len(cols), dtype=int)
    coefficient_values = [[] for _ in cols]
    fold_records = []

    for repeat in range(repeats):
        splitter = GroupKFold(n_splits=n_splits)
        split_groups = shuffled_group_labels(groups, seed + repeat)
        for fold, (train, test) in enumerate(splitter.split(X, y, split_groups)):
            X_train, stats = mu.scale_train(X[train])
            selected, model = mu.fit_signature(
                X_train, y[train], k=k, outcome="continuous",
                corr_threshold=corr_threshold,
            )
            coefficients = np.asarray(model.coef_, dtype=float).reshape(-1)
            for local_idx, feature_idx in enumerate(selected):
                selected_count[feature_idx] += 1
                coefficient = float(coefficients[local_idx])
                coefficient_values[feature_idx].append(coefficient)
                if coefficient > 0:
                    positive_count[feature_idx] += 1
                elif coefficient < 0:
                    negative_count[feature_idx] += 1
            fold_records.append({
                "repeat": repeat + 1,
                "fold": fold + 1,
                "n_train_rows": int(len(train)),
                "n_test_rows": int(len(test)),
                "n_train_patients": int(len(np.unique(groups[train]))),
                "n_test_patients": int(len(np.unique(groups[test]))),
                "selected_features": [cols[i] for i in selected],
            })

    total_fits = len(fold_records)
    rows = []
    for i, feature in enumerate(cols):
        n_selected = int(selected_count[i])
        if n_selected == 0:
            continue
        coefs = np.asarray(coefficient_values[i], dtype=float)
        pos = int(positive_count[i])
        neg = int(negative_count[i])
        rows.append({
            "feature": feature,
            "selected_count": n_selected,
            "selection_frequency": n_selected / total_fits,
            "positive_fraction_among_selected": pos / n_selected,
            "negative_fraction_among_selected": neg / n_selected,
            "median_coefficient_when_selected": float(np.median(coefs)),
            "median_absolute_coefficient_when_selected": float(np.median(np.abs(coefs))),
            "coefficient_q25": float(np.percentile(coefs, 25)),
            "coefficient_q75": float(np.percentile(coefs, 75)),
        })
    rows.sort(key=lambda row: (-row["selection_frequency"], row["feature"]))

    result = {
        "analysis": "patient-grouped repeated outer CV feature-selection stability",
        "data_file": os.path.basename(data_path),
        "data_sha256": file_sha256(data_path),
        "script_sha256": SCRIPT_SHA256,
        "model_utils_sha256": MODEL_UTILS_SHA256,
        "runtime": {
            "python": sys.version.split()[0],
            "numpy": package_version("numpy"),
            "pandas": package_version("pandas"),
            "scikit_learn": package_version("scikit-learn"),
        },
        "n_rows": int(len(df)),
        "n_unique_patients": int(len(np.unique(groups))),
        "n_features": int(len(cols)),
        "outer_folds": int(n_splits),
        "repeats": int(repeats),
        "total_training_fits": int(total_fits),
        "k": int(k),
        "seed": int(seed),
        "corr_threshold": corr_threshold,
        "selection_rule": "top-k by absolute univariate Pearson correlation within each training fold",
        "estimator": "Ridge(alpha=1.0) on training-fold standardized features",
        "interpretation": "Selection frequency measures sensitivity to patient-grouped training partitions, not external or scanner-level reproducibility.",
        "all_fold_records": fold_records,
        "selected_feature_summary": rows,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    csv_path = os.path.splitext(out_path)[0] + ".csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(json.dumps({
        "output_json": out_path,
        "output_csv": csv_path,
        "n_patients": result["n_unique_patients"],
        "n_features": result["n_features"],
        "training_fits": total_fits,
        "unique_selected_features": len(rows),
        "selected_at_least_50pct": sum(r["selection_frequency"] >= 0.5 for r in rows),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.abspath(__file__))
    SCRIPT_SHA256 = file_sha256(__file__)
    MODEL_UTILS_SHA256 = file_sha256(os.path.join(project_dir, "model_utils.py"))
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="features_wide_pyrad.csv")
    parser.add_argument("--out", default="feature_stability_pyrad.json")
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--outer", type=int, default=5)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--corr-threshold", type=float, default=0.85)
    args = parser.parse_args()
    run(args.csv, args.out, args.repeat, args.outer, args.k, args.seed,
        args.corr_threshold)
