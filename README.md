# UPENN-GBM 影像组学对胶质母细胞瘤总生存期的增量价值 —— 复现代码包

对应稿件：*MRI 影像组学对胶质母细胞瘤总生存期的增量价值：UPENN-GBM 队列分析*

本包提供该稿件全部结果的复现代码与派生数据：数据清单与配准质控、影像组学特征提取、采集参数提取、嵌套交叉验证建模、患者级 bootstrap、校准及其 bootstrap 区间、MGMT 关联分解、特征稳定性与敏感性分析。

---

## 1. 本包包含 / 不包含

**包含**：`code/` 下 18 个脚本（见第 3 节）、`data/` 下由公开数据集派生的影像组学特征矩阵、本 README、依赖清单、代码许可与引用元数据（`CITATION.cff`、`.zenodo.json`）。

**不包含**（需自行获取）：

| 内容 | 说明 |
|---|---|
| 原始影像与分割（DICOM / DICOM-SEG） | 受数据使用条款约束，须从公开仓储自行获取，见第 2 节 |
| 派生中间文件 | `manifest.json`、`aligned_npz/{pid}_{seq}.npz`、`pyrad_clinical.csv` |
| 结果文件 | `results_*_bin64.json` 等（由本包脚本生成，可随包发布或邮件索取） |

结果文件可随本包在公开仓储一并发布；若仅发布代码与特征矩阵，可邮件向通信作者索取（联系方式见稿件首页脚注）。

## 2. 数据来源（不在本包内）

- 影像与分割：**UPENN-GBM 第 2 版**，DOI: [10.7937/TCIA.709X-DN49](https://doi.org/10.7937/TCIA.709X-DN49)，可从癌症影像档案馆（TCIA）或影像数据公共云（IDC）获取。
- 临床信息：数据集随附的临床表（MGMT 启动子甲基化状态、生存天数、切除程度、IDH1、KPS）。
- 队列筛选口径（稿件 2.2，见稿件图 1）：630 例 → MGMT 状态明确 279 例 → 生存天数完整（均为死亡病例）269 例 → 四序列与三亚区分割齐备 213 例 → 配准质控通过 **207 例**。筛选在清单生成与导出阶段完成；脚本内不含独立的筛选子命令，复现时请按图 1 逐阶段核对计数。

## 3. 目录结构

```
UPENN_GBM_radiomics_OS_code/
├─ README.md
├─ requirements.txt
├─ LICENSE                       # 代码许可（MIT）
├─ CITATION.cff                  # 引用元数据（提交前请补全作者与仓储地址）
├─ .zenodo.json                  # Zenodo 上传元数据（含与源数据集的关联标识）
├─ data/                         # 由公开数据集派生的特征矩阵（CC BY 4.0）
│  ├─ README.md
│  ├─ features_wide_pyrad_bin64.csv
│  └─ features_wide_pyrad_idh_bin64.csv
└─ code/
   ├─ 数据准备（需原始数据访问权限）
   │   config.py                 # 路径、序列与亚区配置（跨机运行前须修改 RAW_ROOT / WORK）
   │   patient_manifest.py       # 由 IDC 索引与 DICOM 生成序列—分割配对清单 manifest.json
   │   fix_manifest_pairing.py   # 修正异常配对
   │   imaging_io.py             # DICOM 序列读取与 SEG 对齐（被下列脚本调用）
   │   bulk_align.py             # 批量序列—分割对齐
   │   qc_align.py               # 配准质控，标记失败病例
   │   export_aligned.py         # 导出 aligned_npz/{pid}_{seq}.npz 与 pyrad_clinical.csv
   ├─ 特征提取
   │   pyrad_extract_bin64.py    # PyRadiomics 3.1.0，IBSI 固定箱数 binCount=64
   │   scanner_metadata.py       # 由 DICOM 头部提取厂商/机型/场强/TR/TE/TI 分布
   └─ 建模与分析
       model_utils.py            # 训练折内预处理、贪心筛选、岭回归/逻辑回归标签拟合
       analysis_a.py             # 主终点（连续 OS）、二分类终点、敏感性分析
       analysis_b.py             # MGMT 判别与逐序列×亚区块分析
       calibration_check.py      # 折外校准（斜率、截距、calibration-in-the-large）
       calibration_bootstrap_ci.py  # 校准斜率等统计量的患者级 bootstrap 95% 区间
       feature_stability.py      # 50 个训练折的入选特征频率
       run_boot_grouped.py       # 患者级分组 bootstrap（副本不跨折）
       run_sensitivity_2026.py   # 敏感性分析驱动（历史版本）
       run_all_bin64.py          # 全套驱动（core / all）
```

## 4. 运行环境

### 4.1 目录约定

脚本之间按平级目录 import（如 `import model_utils`），且多数脚本以自身所在目录为工作目录。因此请**在 `code/` 目录下运行所有命令**，并将输入文件放在该目录中。

### 4.2 版本

本稿实际运行环境分为两段（依赖 PyRadiomics 的版本约束）：

| 环节 | 环境 |
|---|---|
| 特征提取 | Python 3.9，PyRadiomics 3.1.0，SimpleITK，numpy |
| 建模、统计、bootstrap、校准 | Python 3.12.10，scikit-learn 1.9.1，SimpleITK 2.5.6 |

安装依赖：`pip install -r requirements.txt`（特征提取环节请在满足 PyRadiomics 3.1.0 的 Python 3.9 环境中单独安装）。

## 5. 复现流程

### 路径 A（推荐）：自特征矩阵起复现全部建模、图表与稿件

1. 将 `data/` 下的 `features_wide_pyrad_bin64.csv` 与 `features_wide_pyrad_idh_bin64.csv` 复制到
   `code/`（脚本以自身所在目录为工作目录；若二者位于同一目录，可直接在该目录运行）。
2. 进入目录：`cd code`。
3. 主终点（连续 OS，正文为 20 次随机划分 + 患者级袋外 bootstrap B=200）：

   ```bash
   python analysis_a.py --csv features_wide_pyrad_bin64.csv --repeat 20 --boot 200 --boot-mode oob --out results_A_bin64.json
   ```

4. 次要终点（456 d 二分类 OS）：

   ```bash
   python analysis_a.py --csv features_wide_pyrad_bin64.csv --binary --repeat 20 --out results_A_binary_bin64.json
   ```

5. MGMT 关联分解：

   ```bash
   python analysis_b.py --csv features_wide_pyrad_bin64.csv --out results_B_bin64.json
   ```

6. 特征选择稳定性：

   ```bash
   python feature_stability.py --csv features_wide_pyrad_bin64.csv --out feature_stability_pyrad_bin64.json
   ```

7. 校准（独立进程调用）：

   ```bash
   python -c "import calibration_check as cc; cc.main(csv='features_wide_pyrad_bin64.csv', out='results_calibration_bin64.json')"
   ```

8. 敏感性分析（10 次划分口径，各条单独输出）：

   ```bash
   python run_all_bin64.py all
   ```

9. 采集参数分布（仅读取 DICOM 头部，供稿件 2.3 节引用）：

   ```bash
   python scanner_metadata.py
   ```

10. 校准斜率、截距与平均差值的患者级 bootstrap 95% 区间：

   ```bash
   python calibration_bootstrap_ci.py
   ```

> **关于划分次数**：`run_all_bin64.py core` 中主、次要终点命令为 `--repeat 10`；正文报告的主要与次要终点为 **20 次随机划分**，故复现正文请按第 3、4 步显式指定 `--repeat 20`。敏感性分析沿用 10 次划分（脚本默认值），与稿件一致。

### 路径 B：自原始数据（耗时长，需 IDC 访问权限）

1. 在 `config.py` 中设置 `RAW_ROOT`（原始 DICOM 根目录）与 `WORK`。
2. 依次执行：

   ```bash
   python patient_manifest.py        # 生成序列—分割配对清单
   python fix_manifest_pairing.py    # 修正异常配对（按需）
   python bulk_align.py              # 序列与分割对齐
   python qc_align.py                # 配准质控
   python export_aligned.py          # 导出 npz 与临床列
   python pyrad_extract_bin64.py     # 生成 features_wide_pyrad_bin64.csv
   ```

3. 之后转入路径 A 第 3 步。

## 6. 关键预设参数（与稿件 2.3~2.6 一致）

| 项目 | 取值 |
|---|---|
| 特征类别 | shape / firstorder / GLCM / GLRLM / GLSZM / GLDM / NGTDM（仅 original 滤波器） |
| 候选特征数 | 4 序列 × 3 亚区（ET、NCR、ED） = **1 284** |
| 灰度离散化 | IBSI 固定箱数方案，PyRadiomics `binCount=64` |
| 特征筛选 | 训练折内中位数填补 + 标准化 → 按与 OS 的 \|Pearson r\| 降序贪心入选 → 与已入选特征 \|r\|≥0.85 者跳过 → 取满 k=10 |
| 标签模型 | 岭回归 alpha=1.0（二分类终点为逻辑回归） |
| 验证结构 | 外层 5 折（按患者分组）+ 训练折内交叉拟合；主/次要终点 20 个随机种子，敏感性分析 10 个 |
| 随机种子 | 重复划分按 0…R−1 依次使用（R 由 `--repeat` 指定），无需传入；bootstrap 种子在此基础上递推 |
| bootstrap | B=200，以患者为单位有放回抽样；同一患者的副本共享分组标识，不跨训练/测试折；袋外口径仅在未抽中患者上评估 |
| 次要终点阈值 | 456 d（15 个月）；敏感性阈值 365 d、730 d |

## 7. 主要输出文件

| 文件 | 内容 |
|---|---|
| `results_A_bin64.json` | 主终点：ΔR² 及其划分变异、逐折与汇总 R²、RMSE、MAE、bootstrap 结果 |
| `results_A_binary_bin64.json` | 次要终点：456 d 二分类 ΔAUC、决策曲线净获益 |
| `results_B_bin64.json` | MGMT 标签判别 AUC（全特征块与 12 个序列×亚区块）、βt/βd 与 PM |
| `feature_stability_pyrad_bin64.json/.csv` | 50 个训练折入选特征及频率 |
| `results_calibration_bin64.json` | 折外校准斜率、截距、calibration-in-the-large |
| `results_calibration_boot_bin64.json` | 上述校准统计量的患者级 bootstrap 95% 区间及斜率之差（M1−M0） |
| `scanner_metadata_summary.json`、`scanner_metadata_by_case.csv`、`scanner_metadata_by_series.csv` | 本队列设备厂商、机型、场强与 TR/TE/TI 分布（稿件 2.3 节） |
| `results_A_{nogtr,logos,tunealpha,idh,grid_k5/k20_t080/t090}_bin64.json` | 敏感性分析 |
| `results_A_binary_{12m,24m}_bin64.json` | 二分类阈值敏感性分析 |

## 8. 许可与引用

- **代码**：MIT（见 `LICENSE`）。
- **数据**：`data/` 下的特征矩阵是 UPENN-GBM（第 2 版，DOI: 10.7937/TCIA.709X-DN49，CC BY 4.0）的改编产物，沿用相同许可 **CC BY 4.0**；再发布时须标注数据集名称与版本、来源（TCIA / 影像数据公共云 IDC）、DOI 与原始数据集文献。
- **引用**：请同时引用本稿件（题目、作者、期刊）与数据集 DOI 10.7937/TCIA.709X-DN49，并注明使用 PyRadiomics 3.1.0 与 IBSI 标准；机器可读引用见 `CITATION.cff`，Zenodo 上传元数据见 `.zenodo.json`。

## 9. 发布到 GitHub / Zenodo（待作者执行，本地仓库已就绪）

**当前状态**：本目录已完成 `git init -b main` 与首次提交（`LICENSE`、`CITATION.cff`、
`.zenodo.json`、`data/README.md`、`code/`、`data/` 共 27 个文件）。本机未安装 `gh`
命令行、也没有保存 GitHub 凭据，因此**建库与推送需要作者本人在网页完成**。

**发布前先补两处占位符**：`CITATION.cff` 与 `.zenodo.json` 里的作者姓名和
`repository-code`；并把首次提交的作者信息改成自己的：

```bash
cd d:\学习\影像\UPENN_GBM_radiomics_OS_code
git config user.name "你的姓名"
git config user.email "你的邮箱"
git commit --amend --reset-author      # 用真实作者信息重写首次提交
```

**第一步：建库并推送**

```bash
# 在 https://github.com/new 建空仓库（建议名 upenn-gbm-radiomics-os，Public，不要勾选初始化 README）
git remote add origin https://github.com/<你的账号>/upenn-gbm-radiomics-os.git
git push -u origin main
```

**第二步：Zenodo 存档拿 DOI（投稿件里要填的就是这个 DOI）**

1. 登录 https://zenodo.org ，用 GitHub 账号授权；
2. 进入 GitHub 页面，找到该仓库并打开同步开关（ON）；
3. 在 GitHub 上发布 Release（标签如 `v1.0.0`），Zenodo 自动生成快照并分配 DOI；
4. 许可选 **MIT**，描述中注明：影像组学特征矩阵为 UPENN-GBM（CC BY 4.0）的改编产物，
   沿用 CC BY 4.0，来源 DOI 10.7937/TCIA.709X-DN49；
5. 把仓储地址与 DOI 回填到稿件的"数据与代码可用性"声明。

> 注：`data/` 下两个特征矩阵共约 9.4 MB，Zenodo 单仓储上限 50 GB，完全够用；
> 若编辑部/单位不要求公开特征矩阵，可只上传 `code/` 与文档（删掉 `data/` 后重新提交），
> 但稿件中的表述要同步改为"代码公开、特征矩阵邮件索取"。

## 10. 已知限制

1. 本研究结果为**阴性**（影像组学标签未带来增量）：本包用于复现与核查，不提供可用于临床的预测模型。
2. 填补、标准化、筛选与模型拟合均在训练折内完成，故每折入选特征与系数均随折变化，**标签并非固定特征组合**，本包不输出单一"最终模型"文件。
3. 路径常量写在 `config.py` 与各脚本顶部（本稿为原始工作路径），跨机运行前须修改。
4. 特征提取与建模分处两个 Python 环境（见 4.2），若合并环境，请确认 PyRadiomics 与 Python 版本匹配。
5. 本包未包含多序列配准的外部工具（如 dcm2niix）；数据准备脚本直接读取 DICOM 与 DICOM-SEG。
