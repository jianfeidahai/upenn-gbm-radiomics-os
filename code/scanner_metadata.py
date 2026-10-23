# -*- coding: utf-8 -*-
"""提取本队列 207 例的 MRI 采集参数（厂商、机型、场强、TR/TE/TI、翻转角、层厚、像素间距）。

只读 DICOM 头部（stop_before_pixels），不载入像素数据；用于稿件 2.3 节的采集参数描述。

输入：manifest.json（序列—分割配对清单）、features_wide_pyrad_bin64.csv（本队列 pid 清单）
输出：scanner_metadata_by_series.csv（每例×序列一行）
      scanner_metadata_by_case.csv（每例一行）
      scanner_metadata_summary.json（分布汇总，供正文引用）

用法：python scanner_metadata.py
"""
import glob
import json
import os

import numpy as np
import pandas as pd
import pydicom

HERE = os.path.dirname(os.path.abspath(__file__))
SEQ_ORDER = ["T1", "T1CE", "T2", "FLAIR"]

TAGS = {
    "InstitutionName": "0008,0080",
    "StationName": "0008,1010",
    "Manufacturer": "0008,0070",
    "ManufacturerModelName": "0008,1090",
    "DeviceSerialNumber": "0018,1000",
    "MagneticFieldStrength": "0018,0087",
    "ScanningSequence": "0018,0020",
    "SequenceVariant": "0018,0021",
    "ScanOptions": "0018,0022",
    "RepetitionTime": "0018,0080",
    "EchoTime": "0018,0081",
    "InversionTime": "0018,0082",
    "FlipAngle": "0018,1314",
    "SliceThickness": "0018,0050",
    "PixelSpacing": "0028,0030",
    "ProtocolName": "0018,1030",
    "SeriesDescription": "0008,103E",
}


def read_header(mr_dir):
    """读取序列目录下第一个 DICOM 文件的头部。"""
    fs = sorted(glob.glob(os.path.join(mr_dir, "*")))
    fs = [f for f in fs if os.path.isfile(f)]
    if not fs:
        return None
    for f in fs:
        try:
            return pydicom.dcmread(f, stop_before_pixels=True)
        except Exception:
            continue
    return None


def get(ds, kw):
    if ds is None:
        return ""
    v = ds.get(kw, "")
    if v == "" or v is None:
        return ""
    try:
        if kw in ("PixelSpacing",):
            return float(list(v)[0])
        if kw in ("ScanningSequence", "SequenceVariant"):
            return "/".join(list(v))
        return v
    except Exception:
        return str(v)


def main():
    pids = pd.read_csv(os.path.join(HERE, "features_wide_pyrad_bin64.csv"),
                       encoding="utf-8-sig", usecols=["pid"])["pid"].astype(str).tolist()
    man = json.load(open(os.path.join(HERE, "manifest.json"), encoding="utf-8"))
    by_pid = {str(r["pid"]): r for r in man}

    rows = []
    for pid in pids:
        r = by_pid.get(pid)
        if r is None:
            continue
        for seq in SEQ_ORDER:
            mr_dir = r.get("series", {}).get(seq, {}).get("mr", "")
            ds = read_header(mr_dir)
            rec = {"pid": pid, "seq": seq}
            for kw in TAGS:
                rec[kw] = get(ds, kw)
            rows.append(rec)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "scanner_metadata_by_series.csv"),
              index=False, encoding="utf-8-sig")

    # 每例一行：以 T1 序列代表厂商/机型/场强
    case_rows = []
    for pid in pids:
        sub = df[df["pid"] == pid]
        if sub.empty:
            continue
        t1 = sub[sub["seq"] == "T1"]
        src = t1 if not t1.empty else sub
        rec = {"pid": pid,
               "Manufacturer": src.iloc[0].get("Manufacturer", ""),
               "Model": src.iloc[0].get("ManufacturerModelName", ""),
               "FieldStrength_T": src.iloc[0].get("MagneticFieldStrength", ""),
               "InstitutionName": src.iloc[0].get("InstitutionName", "")}
        for seq in SEQ_ORDER:
            s = sub[sub["seq"] == seq]
            rec["TR_%s" % seq] = s.iloc[0].get("RepetitionTime", "") if not s.empty else ""
            rec["TE_%s" % seq] = s.iloc[0].get("EchoTime", "") if not s.empty else ""
            rec["TI_%s" % seq] = s.iloc[0].get("InversionTime", "") if not s.empty else ""
            rec["FA_%s" % seq] = s.iloc[0].get("FlipAngle", "") if not s.empty else ""
            rec["ST_%s" % seq] = s.iloc[0].get("SliceThickness", "") if not s.empty else ""
            rec["PS_%s" % seq] = s.iloc[0].get("PixelSpacing", "") if not s.empty else ""
        case_rows.append(rec)
    dfc = pd.DataFrame(case_rows)
    dfc.to_csv(os.path.join(HERE, "scanner_metadata_by_case.csv"),
               index=False, encoding="utf-8-sig")

    def dist(s, name):
        ss = s.astype(str)
        vc = ss[ss.str.len() > 0].value_counts()
        tot = int(vc.sum())
        return {"variable": name, "n_available": tot,
                "categories": {str(k): {"n": int(v), "pct": round(100.0 * v / max(tot, 1), 1)}
                               for k, v in vc.items()}}

    def num(s, name, nd=2):
        v = pd.to_numeric(s, errors="coerce").dropna().values
        if len(v) == 0:
            return {"variable": name, "n": 0}
        return {"variable": name, "n": int(len(v)),
                "median": round(float(np.median(v)), nd),
                "q1": round(float(np.percentile(v, 25)), nd),
                "q3": round(float(np.percentile(v, 75)), nd),
                "min": round(float(v.min()), nd),
                "max": round(float(v.max()), nd)}

    per_seq = {}
    for seq in SEQ_ORDER:
        s = df[df["seq"] == seq]
        per_seq[seq] = {
            "n_series": int(len(s)),
            "TR_ms": num(s["RepetitionTime"], "TR_ms"),
            "TE_ms": num(s["EchoTime"], "TE_ms"),
            "TI_ms": num(s["InversionTime"], "TI_ms"),
            "flip_angle_deg": num(s["FlipAngle"], "flip_angle_deg", 1),
            "slice_thickness_mm": num(s["SliceThickness"], "slice_thickness_mm", 2),
            "pixel_spacing_mm": num(s["PixelSpacing"], "pixel_spacing_mm", 3),
        }

    summary = {
        "note": "由 manifest.json 中序列目录的 DICOM 头部提取（仅读头部，不载入像素）；"
                "机构与站点字段已由发布方去标识化，故逐例机构不可考。",
        "n_cases": int(dfc.shape[0]),
        "n_series": int(df.shape[0]),
        "manufacturer": dist(dfc["Manufacturer"], "Manufacturer"),
        "model": dist(dfc["Model"], "ManufacturerModelName"),
        "field_strength_T": dist(dfc["FieldStrength_T"], "MagneticFieldStrength"),
        "per_sequence": per_seq,
    }
    path = os.path.join(HERE, "scanner_metadata_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("saved ->", path)


if __name__ == "__main__":
    main()
