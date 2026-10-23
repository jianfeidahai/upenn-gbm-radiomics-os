# -*- coding: utf-8 -*-
"""在 IBSI 固定箱数（binCount=64）特征矩阵上重跑全套分析。

用法：python run_all_bin64.py [core|all]
  core：主分析（连续+二分类+MGMT分解+特征稳定性+校准）
  all ：core + 全部敏感性分析
所有结果写入 *_bin64.json，不覆盖既有 main 结果文件。
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
CSV = "features_wide_pyrad_bin64.csv"
LOG = os.path.join(HERE, "run_all_bin64.log")


def sh(cmd):
    t0 = time.time()
    with open(LOG, "a", encoding="utf-8") as f:
        f.write("\n[%s] %s\n" % (time.strftime("%H:%M:%S"), cmd))
        f.flush()
        p = subprocess.run(cmd, shell=True, cwd=HERE, stdout=f,
                           stderr=subprocess.STDOUT)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write("[%s] done rc=%d (%.0fs)\n" % (time.strftime("%H:%M:%S"),
                                               p.returncode, time.time() - t0))
    return p.returncode


def build_idh():
    """由既有 features_wide_pyrad_idh.csv 取 idh1/kps 列，并入 bin64 矩阵。"""
    import pandas as pd

    base = pd.read_csv(os.path.join(HERE, CSV), encoding="utf-8-sig")
    idh = pd.read_csv(os.path.join(HERE, "features_wide_pyrad_idh.csv"),
                      encoding="utf-8-sig")[["pid", "idh1", "kps"]]
    m = base.merge(idh, on="pid", how="left", validate="one_to_one")
    out = "features_wide_pyrad_idh_bin64.csv"
    m.to_csv(os.path.join(HERE, out), index=False, encoding="utf-8-sig")
    print("saved ->", out, m.shape, "idh1 非空", int(m["idh1"].notna().sum()),
          m["idh1"].astype(str).value_counts().to_dict())


CORE = [
    f'"{PY}" analysis_a.py --csv {CSV} --repeat 10 --boot 200 --boot-mode oob '
    f'--out results_A_bin64.json',
    f'"{PY}" analysis_a.py --csv {CSV} --binary --repeat 10 '
    f'--out results_A_binary_bin64.json',
    f'"{PY}" analysis_b.py --csv {CSV} --out results_B_bin64.json',
    f'"{PY}" feature_stability.py --csv {CSV} '
    f'--out feature_stability_pyrad_bin64.json',
]

SENS = [
    f'"{PY}" analysis_a.py --csv {CSV} --no-gtr --repeat 10 '
    f'--out results_A_nogtr_bin64.json',
    f'"{PY}" analysis_a.py --csv {CSV} --log-os --repeat 10 --boot 200 '
    f'--boot-mode oob --out results_A_logos_bin64.json',
    f'"{PY}" analysis_a.py --csv {CSV} --tune-alpha --repeat 10 '
    f'--out results_A_tunealpha_bin64.json',
    f'"{PY}" analysis_a.py --csv features_wide_pyrad_idh_bin64.csv --include-idh '
    f'--repeat 10 --out results_A_idh_bin64.json',
    f'"{PY}" analysis_a.py --csv {CSV} --k 5 --corr-threshold 0.80 --repeat 10 '
    f'--out results_A_grid_k5_t080_bin64.json',
    f'"{PY}" analysis_a.py --csv {CSV} --k 5 --corr-threshold 0.90 --repeat 10 '
    f'--out results_A_grid_k5_t090_bin64.json',
    f'"{PY}" analysis_a.py --csv {CSV} --k 20 --corr-threshold 0.80 --repeat 10 '
    f'--out results_A_grid_k20_t080_bin64.json',
    f'"{PY}" analysis_a.py --csv {CSV} --k 20 --corr-threshold 0.90 --repeat 10 '
    f'--out results_A_grid_k20_t090_bin64.json',
    f'"{PY}" analysis_a.py --csv {CSV} --binary --thresh 365 --repeat 10 '
    f'--out results_A_binary_12m_bin64.json',
    f'"{PY}" analysis_a.py --csv {CSV} --binary --thresh 730 --repeat 10 '
    f'--out results_A_binary_24m_bin64.json',
]


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "core"
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    build_idh()
    for cmd in CORE:
        sh(cmd, env)
    if mode == "all":
        for cmd in SENS:
            sh(cmd, env)
    sh(f'"{PY}" -X utf8 -c "import calibration_check as cc; '
       f'cc.main(csv=\'{CSV}\', out=\'results_calibration_bin64.json\')"', env)


def sh(cmd, env=None):  # noqa: F811
    t0 = time.time()
    with open(LOG, "a", encoding="utf-8") as f:
        f.write("\n[%s] %s\n" % (time.strftime("%H:%M:%S"), cmd))
        f.flush()
        p = subprocess.run(cmd, shell=True, cwd=HERE, stdout=f,
                           stderr=subprocess.STDOUT, env=env)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write("[%s] done rc=%d (%.0fs)\n" % (time.strftime("%H:%M:%S"),
                                               p.returncode, time.time() - t0))
    return p.returncode


if __name__ == "__main__":
    main()
