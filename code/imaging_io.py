# -*- coding: utf-8 -*-
"""
影像 I/O：
  - read_mr_series(folder)       : DICOM 序列 -> (volume[x,y,z], affine), 右手系校正
  - read_dicom_seg(path)         : DICOM-SEG -> (label[x,y,z], affine, {label:name})
  - resample_label(...)          : 将标签体素最近邻重采样到目标网格（保标签）

几何约定（与 nibabel 一致，affine 为 index->world，index 顺序 (x=col, y=row, z=slice)）：
  world = ipp + col_idx*(col_dir*col_spacing) + row_idx*(row_dir*row_spacing) + slice_idx*(normal*slice_gap)
其中 normal = cross(row_dir, col_dir)。

关键修正：帧/层的排序一律按"沿切片法向的投影"而非原始 z 坐标，以兼容法向为 -z 的序列
（T2/FLAIR 的 SEG 法向为 [0,0,-1]，按 z 排序会整段反转导致层索引偏移 n-1）。
"""
import numpy as np, pydicom, glob, os
import nibabel as nib
from scipy.ndimage import affine_transform


def _affine(iop, ipp, ps, sz):
    """由 IOP / 原点 IPP / PixelSpacing / 层间距 sz 构造 index->world 仿射。

    index 约定 (x=col, y=row, z=slice)：列方向对应 col_dir*col_spacing，行方向对应
    row_dir*row_spacing，层方向对应 normal*sz。
    """
    iop = np.array([float(x) for x in iop], dtype=float)
    ipp = np.array([float(x) for x in ipp], dtype=float)
    ps = [float(x) for x in ps]
    row_c, col_c = iop[0:3], iop[3:6]
    sx, sy = ps[1], ps[0]
    normal = np.cross(row_c, col_c)
    nrm = np.linalg.norm(normal)
    if nrm > 0:
        normal = normal / nrm
    aff = np.eye(4)
    aff[:3, 0] = col_c * sx
    aff[:3, 1] = row_c * sy
    aff[:3, 2] = normal * sz
    aff[:3, 3] = ipp
    return aff


def _force_ras(vol, aff):
    """若仿射行列式为负（左手系），沿 x 轴翻转体素并修正仿射，使世界映射保持不变。"""
    if np.linalg.det(aff[:3, :3]) < 0:
        vol = np.flip(vol, axis=0)
        n0 = vol.shape[0]
        aff[:3, 3] = aff[:3, 3] + aff[:3, 0] * (n0 - 1)
        aff[:3, 0] = -aff[:3, 0]
    return vol, aff


def _slice_gap(ipps):
    """层间距 = 首末帧实际欧氏距离 / (层数-1)，对任意（含 -z）法向都正确。"""
    if len(ipps) > 1:
        d = float(np.linalg.norm(np.array(ipps[-1]) - np.array(ipps[0])))
        return d / (len(ipps) - 1)
    return 1.0


def _normal(iop):
    row_c, col_c = np.array(iop[0:3], float), np.array(iop[3:6], float)
    n = np.cross(row_c, col_c)
    nrm = np.linalg.norm(n)
    return (n / nrm) if nrm > 0 else np.array([0.0, 0.0, 1.0])


def read_mr_series(folder):
    """DICOM MR 序列 -> (volume[x,y,z], affine)。按沿法向投影排序层，端点推导层间距。"""
    import glob, os
    files = sorted(glob.glob(os.path.join(folder, "*.dcm")))
    ds_list = [pydicom.dcmread(f, force=True) for f in files]
    ds_list = [d for d in ds_list if d is not None]
    iop = [float(x) for x in ds_list[0].ImageOrientationPatient]
    nrm = _normal(iop)
    # 排序键：沿法向投影（兼容 -z 法向）
    def key(d):
        ipp = np.array([float(x) for x in d.ImagePositionPatient])
        return float(np.dot(ipp, nrm))
    ds_list.sort(key=key)
    ps = [float(x) for x in ds_list[0].PixelSpacing]
    arrs = []
    ipps = []
    for d in ds_list:
        a = d.pixel_array.astype(np.float32)
        if getattr(d, "PhotometricInterpretation", "") == "MONOCHROME1":
            a = a.max() - a
        if ("RescaleSlope" in d) or ("RescaleIntercept" in d):
            a = a * float(getattr(d, "RescaleSlope", 1) or 1) + float(getattr(d, "RescaleIntercept", 0) or 0)
        arrs.append(a)
        ipps.append(np.array([float(x) for x in d.ImagePositionPatient]))
    vol = np.stack(arrs, axis=-1)            # (row, col, slice) = (y, x, z)
    vol = np.transpose(vol, (1, 0, 2))        # (x, y, z)
    sz = _slice_gap(ipps)
    aff = _affine(iop, ipps[0], ps, sz)
    vol, aff = _force_ras(vol, aff)
    return vol, aff


def _seg_frame_meta(ds):
    """返回 (frames, iop, ps, segs)：frames = [(ipp[3], ref, frame_idx), ...]。"""
    sh = ds.SharedFunctionalGroupsSequence[0]
    try:
        iop = [float(x) for x in sh.PlaneOrientationSequence[0].ImageOrientationPatient]
    except Exception:
        iop = [float(x) for x in getattr(ds, "ImageOrientationPatient", [1, 0, 0, 0, 1, 0])]
    try:
        ps = [float(x) for x in sh.PixelMeasuresSequence[0].PixelSpacing]
    except Exception:
        ps = [float(x) for x in getattr(ds, "PixelSpacing", [1.0, 1.0])]
    segs = {}
    for s in ds.SegmentSequence:
        segs[int(s.SegmentNumber)] = getattr(s, "SegmentDescription",
                                             getattr(s, "SegmentLabel", "seg%d" % int(s.SegmentNumber)))
    frames = []
    for i, fg in enumerate(ds.PerFrameFunctionalGroupsSequence):
        pls = getattr(fg, "PlanePositionSequence", None)
        if pls is None:
            continue
        ipp = np.array([float(x) for x in pls[0].ImagePositionPatient])
        ref = None
        if hasattr(fg, "SegmentIdentificationSequence"):
            ref = int(fg.SegmentIdentificationSequence[0].ReferencedSegmentNumber)
        frames.append((ipp, ref, i))
    return frames, iop, ps, segs


def read_dicom_seg(path, ref_aff=None):
    """DICOM-SEG -> (label[x,y,z], affine, {label:name})。

    按"沿法向投影"对帧去重并排序，端点推导层间距，从根本上消除 -z 法向导致的层序反转。

    ref_aff: 若提供（配对 MR 的 affine），则仅借用 MR 的"平面内"几何（原点 x/y 与列/行方向），
    保留 SEG 自身可靠的 z（层方向与层原点）。UPenn-GBM 的 SEG 像素矩阵与 MR 同网格，
    但其 in-plane ImagePositionPatient/IOP 元数据常存于不同坐标系，故以 MR 平面内几何为基准。
    """
    ds = pydicom.dcmread(path, force=True)
    rows, cols = int(ds.Rows), int(ds.Columns)
    frames, iop, ps, segs = _seg_frame_meta(ds)
    nrm = _normal(iop)
    # 按位置（四舍五入到 0.01mm）去重；用代表位置（平均）记录
    groups = {}          # key -> (repr_ipp, [(ref, frame_idx), ...])
    for ipp, ref, i in frames:
        key = tuple(np.round(ipp, 2))
        if key not in groups:
            groups[key] = (ipp.copy(), [])
        groups[key][1].append((ref, i))
    # 排序键：沿法向投影
    def proj(k):
        return float(np.dot(groups[k][0], nrm))
    ordered = sorted(groups.keys(), key=proj)
    idx = {k: s for s, k in enumerate(ordered)}
    n_slices = len(ordered)
    arr = ds.pixel_array.astype(np.uint32)
    label = np.zeros((n_slices, rows, cols), dtype=np.int16)
    for key, (repr_ipp, lst) in groups.items():
        s = idx[key]
        for ref, i in lst:
            fr = arr[i]
            if ref is not None:                       # 每帧对应单个 segment
                mask = (fr > 0).astype(np.int16) * ref
                label[s] = np.maximum(label[s], mask)
            else:                                    # 位打包：一帧含所有 segment
                for k in segs:
                    mask = (((fr >> (k - 1)) & 1).astype(np.int16)) * k
                    label[s] = np.maximum(label[s], mask)
    ipps = [groups[k][0] for k in ordered]
    sz = _slice_gap(ipps)
    aff = _affine(iop, ipps[0], ps, sz)
    # 转为 (x,y,z) 布局以与 affine 约定一致
    label = np.transpose(label, (2, 1, 0))
    label, aff = _force_ras(label, aff)
    if ref_aff is not None:
        # 平面内（x,y）采用 MR 几何；z（层方向与层原点）保留 SEG 自身（已验证可靠）
        aff = np.array(aff, copy=True)
        aff[:3, 0] = ref_aff[:3, 0]
        aff[:3, 1] = ref_aff[:3, 1]
        aff[:3, 3][0] = ref_aff[:3, 3][0]
        aff[:3, 3][1] = ref_aff[:3, 3][1]
    return label, aff, segs


def resample_label(src_vol, src_aff, tgt_shape, tgt_aff):
    """最近邻将 src 标签体素重采样到 tgt 网格（保标签值）。"""
    M4 = np.linalg.inv(src_aff) @ tgt_aff          # 目标体素 -> 源体素 (齐次)
    lin = M4[:3, :3]
    off = M4[:3, 3]
    out = affine_transform(src_vol.astype(np.float32), lin, offset=off,
                           output_shape=tuple(tgt_shape), order=0,
                           mode="constant", cval=0.0)
    return out.astype(np.int16)


def align_seg_to_mr(mr_dir, seg_dir, min_vox=200, on_tissue_thr=0.5, mr_data=None):
    """将 SEG 重采样到 MR 网格，自动选择对齐方式。

    UPenn-GBM 的 DICOM-SEG in-plane 元数据对部分患者不可靠（存于不同坐标系），对另一部分可靠。
    策略：分别以"SEG 自身几何"与"MR in-plane 几何覆盖"两种方式重采样，取肿瘤落于 MR 前景
    （脑组织，强度>0，针对 CaPTk 颅骨剥离结果）比例更高者。
    返回 (resampled_label[x,y,z], chosen_affine, segs, method)。
    mr_data: 可选 (vol, vaff)，避免重复读 MR。
    """
    if mr_data is not None:
        vol, vaff = mr_data
    else:
        vol, vaff = read_mr_series(mr_dir)
    sd = sorted(glob.glob(os.path.join(seg_dir, "*.dcm")))
    if not sd:
        raise FileNotFoundError(seg_dir)
    sd = sd[0]
    # 方式 A：SEG 自身几何
    label_o, saff_o, segs = read_dicom_seg(sd)
    res_o = resample_label(label_o, saff_o, vol.shape, vaff)
    # 方式 B：以 MR in-plane 为基准
    label_r, saff_r, _ = read_dicom_seg(sd, ref_aff=vaff)
    res_r = resample_label(label_r, saff_r, vol.shape, vaff)
    fg = vol > 0
    def on_tissue(res):
        n = int((res > 0).sum())
        if n < min_vox:
            return 0.0, n
        ot = float((fg & (res > 0)).sum()) / n
        return ot, n
    on_o, n_o = on_tissue(res_o)
    on_r, n_r = on_tissue(res_r)
    if (on_r >= on_o) and (on_r >= on_tissue_thr) and (n_r >= min_vox):
        return res_r, saff_r, segs, "mr_inplane"
    if (on_o >= on_tissue_thr) and (n_o >= min_vox):
        return res_o, saff_o, segs, "seg_own"
    # 两者都低于阈值：取 on-tissue 较高者（标注 lowconf 供 QC 复查）
    if on_r >= on_o:
        return res_r, saff_r, segs, "lowconf"
    return res_o, saff_o, segs, "lowconf"


def decode_seg_array(ds):
    """从 DICOM-SEG 数据集解析标签体素，返回 (z,y,x) 数组与 {label:name}（与 read_dicom_seg 同口径）。"""
    rows, cols = int(ds.Rows), int(ds.Columns)
    frames, iop, ps, segs = _seg_frame_meta(ds)
    nrm = _normal(iop)
    groups = {}
    for ipp, ref, i in frames:
        key = tuple(np.round(ipp, 2))
        if key not in groups:
            groups[key] = (ipp.copy(), [])
        groups[key][1].append((ref, i))
    def proj(k):
        return float(np.dot(groups[k][0], nrm))
    ordered = sorted(groups.keys(), key=proj)
    idx = {k: s for s, k in enumerate(ordered)}
    n = len(ordered)
    arr = ds.pixel_array.astype(np.uint32)
    label = np.zeros((n, rows, cols), dtype=np.int16)
    for key, (repr_ipp, lst) in groups.items():
        s = idx[key]
        for ref, i in lst:
            fr = arr[i]
            if ref is not None:
                mask = (fr > 0).astype(np.int16) * ref
            else:
                mask = np.zeros_like(fr, dtype=np.int16)
                for k in segs:
                    mask |= (((fr >> (k - 1)) & 1).astype(np.int16)) * k
            label[s] = np.maximum(label[s], mask)
    return label, segs


def mr_shape_and_affine(folder):
    """仅读几何（不解码像素），返回 (shape[x,y,z], affine)。用于对齐 QC，速度远快于 read_mr_series。"""
    import glob, os
    files = sorted(glob.glob(os.path.join(folder, "*.dcm")))
    ds_list = [pydicom.dcmread(f, force=True) for f in files]
    ds_list = [d for d in ds_list if d is not None]
    iop = [float(x) for x in ds_list[0].ImageOrientationPatient]
    nrm = _normal(iop)
    def key(d):
        ipp = np.array([float(x) for x in d.ImagePositionPatient])
        return float(np.dot(ipp, nrm))
    ds_list.sort(key=key)
    ps = [float(x) for x in ds_list[0].PixelSpacing]
    ipps = [np.array([float(x) for x in d.ImagePositionPatient]) for d in ds_list]
    # 行列数取自首帧
    rows = int(ds_list[0].Rows); cols = int(ds_list[0].Columns)
    shape = (cols, rows, len(ds_list))          # (x, y, z)
    sz = _slice_gap(ipps)
    aff = _affine(iop, ipps[0], ps, sz)
    # 与 read_mr_series 一致的 RAS 校正：翻转 x 轴并修正 origin，使世界映射/范围一致
    if np.linalg.det(aff[:3, :3]) < 0:
        n0 = shape[0]
        aff[:3, 3] = aff[:3, 3] + aff[:3, 0] * (n0 - 1)
        aff[:3, 0] = -aff[:3, 0]
    return shape, aff


def read_mr_sitk(folder):
    """用 SimpleITK 正确读取 DICOM 序列（自动排序+几何），返回 sitk.Image。"""
    import SimpleITK as sitk
    reader = sitk.ImageSeriesReader()
    files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(folder)
    reader.SetFileNames(files)
    return reader.Execute()


def resample_seg_to_mr_sitk(seg_ds, mr_img):
    """将 DICOM-SEG 标签重采样到 mr_img 几何（SITK 负责对齐）。返回 (z,y,x) 标签数组。"""
    import SimpleITK as sitk
    label, _ = decode_seg_array(seg_ds)
    seg_img = sitk.ReadImage(seg_file_of(seg_ds))  # 仅取几何
    label_img = sitk.GetImageFromArray(label)
    label_img.SetOrigin(seg_img.GetOrigin())
    label_img.SetSpacing(seg_img.GetSpacing())
    label_img.SetDirection(seg_img.GetDirection())
    res = sitk.Resample(label_img, mr_img, sitk.Transform(), sitk.sitkNearestNeighbor, 0, label_img.GetPixelID())
    return sitk.GetArrayFromImage(res)


def seg_file_of(ds):
    return ds.filename
