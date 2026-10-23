# 数据说明：影像组学特征矩阵

本目录存放由公开数据集 **UPENN-GBM（第 2 版）** 派生的影像组学特征矩阵，用于复现稿件
《MRI 影像组学对胶质母细胞瘤总生存期的增量价值：UPENN-GBM 队列分析》的全部建模结果。

## 文件清单

| 文件 | 内容 | 大小 |
|---|---|---|
| `features_wide_pyrad_bin64.csv` | 207 例 × 1 284 个 PyRadiomics 特征 + 临床/分子列，宽表，1 行 1 例 | 约 4.7 MB |
| `features_wide_pyrad_idh_bin64.csv` | 同上，另并入 `idh1` 与 `kps` 两列，用于"纳入 IDH1"的敏感性分析 | 约 4.6 MB |

## 结构

- 影像组学特征列名：`{序列}_{亚区}_{PyRadiomics 特征名}`，例如 `T1CE_ET_original_glcm_Contrast`。
  - 序列 ∈ `{T1, T1CE, T2, FLAIR}`
  - 亚区 ∈ `{ET, NCR, ED}`（enhancing tumor 强化肿瘤区 / necrotic core 坏死核心区 / peritumoral edema 瘤周水肿区）
  - 特征类：shape、firstorder、GLCM、GLRLM、GLSZM、GLDM、NGTDM（仅 original 滤波器）
- 临床/分子列：`pid`、`days`（生存天数）、`event`、`age`、`gender`、`mgmt`、`gtr`，以及（idh 版）`idh1`、`kps`。

## 生成方式

按 `../code/` 目录脚本顺序生成：

```
patient_manifest.py → bulk_align.py → qc_align.py → export_aligned.py → pyrad_extract_bin64.py
```

含 IDH1 的版本由 `run_all_bin64.py` 的 `build_idh()` 并入 `idh1`/`kps` 两列后得到；
本目录已提供该文件，可跳过该步。

## 许可与署名

- **原始数据集**：UPENN-GBM（第 2 版），DOI: 10.7937/TCIA.709X-DN49，遵循其使用条款（CC BY 4.0）。
- **本目录的特征矩阵**：为上述数据集的改编产物（derivative work），沿用相同许可 **CC BY 4.0**。
  使用、传播或再发布时须标注：数据集名称与版本（UPENN-GBM v2）、来源（TCIA / 影像数据公共云 IDC）、
  DOI 10.7937/TCIA.709X-DN49，并引用原始数据集文献。
- **本仓库的代码**：MIT 许可，见根目录 `LICENSE`。

## 不包含的内容

原始影像与分割（DICOM / DICOM-SEG）不包含在本仓库，须从公开仓储自行获取；
中间文件（`aligned_npz/*.npz`、`manifest.json`）同样不包含。
