# -*- coding: utf-8 -*-
"""全局配置：路径、序列与分割标签映射。"""
import os

# 原始 DICOM 根目录（IDC 下载）
RAW_ROOT = r"D:/医学影像数据/IDC/upenn_gbm"

# 处理后数据根目录（NIfTI + 特征 + 模型结果）
WORK = r"D:/学习/影像/pipeline"
NII_DIR = os.path.join(WORK, "nii")          # 转换后的 NIfTI
FEAT_DIR = os.path.join(WORK, "features")    # 特征矩阵
RES_DIR = os.path.join(WORK, "results")      # 结果表/图
for _d in (NII_DIR, FEAT_DIR, RES_DIR):
    os.makedirs(_d, exist_ok=True)

MANIFEST = os.path.join(WORK, "manifest.json")

# 目标 MRI 序列（按 DICOM SeriesDescription 关键字判定）
SEQUENCES = ["T1", "T1CE", "T2", "FLAIR"]

# DICOM-SEG 标签 -> 亚区名（与 Sci Data 2022 描述一致）
SEG_LABELS = {1: "NCR", 2: "ED", 3: "ET"}   # necrosis / edema / enhancing tumor
SUBREGIONS = ["ET", "NCR", "ED"]

# 灰度离散化（影像组学 IBSI 推荐：固定 binWidth 或固定 binCount）
BIN_WIDTH = 25.0        # 固定 BinWidth（Hounsfield/信号强度单位）
RESCALE_FOR_BIN = False # UPenn 已 CaPTk 预处理，直接用原始强度

# 随机种子（复现）
SEED = 20261015
