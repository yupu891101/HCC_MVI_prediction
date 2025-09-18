from typing import Literal
import numpy as np
from scipy import ndimage
from scipy.signal import fftconvolve

INTERPOLATION_MAPPING = {
    "Nearest": 0,
    "Bilinear": 1,
    "BiQuadratic": 2,
    "Cubic": 3
}

def normalized_cross_correlation_fft(array1: np.ndarray, array2: np.ndarray) -> np.ndarray:
    """
    使用 FFT 快速計算兩個陣列的標準化互相關。
    """
    array1 = array1.astype(np.float64)
    array2 = array2.astype(np.float64)

    # 減去均值
    array1_mean = array1.mean()
    array2_mean = array2.mean()
    array1_centered = array1 - array1_mean
    array2_centered = array2 - array2_mean

    # 互相關（使用 FFT 實現）
    cross_correlation = fftconvolve(array1_centered, np.flip(array2_centered), mode='same')

    # 標準化步驟與方法一相同
    array1_auto_correlation = np.sum(array1_centered**2)
    array2_auto_correlation = np.sum(array2_centered**2)

    denominator = np.sqrt(array1_auto_correlation * array2_auto_correlation)
    if denominator == 0:
        return np.zeros_like(cross_correlation)

    normalized_result = cross_correlation / denominator

    return normalized_result

def slice_resize(
        dicom_array: np.ndarray,
        target_size: int | tuple[int, int],
        interpolation: Literal["Nearest", "Bilinear", "BiQuadratic", "Cubic"] = "Bilinear"
    ) -> np.ndarray:
    """dicom_array shape: (slice_num, H, W)"""
    if isinstance(target_size, int):
        target_h = target_w = target_size
    else:
        target_h, target_w = target_size

    _, dicom_h, dicom_w = dicom_array.shape
    resize_factor = (target_h/dicom_h, target_w/dicom_w)
    resized_dicom = np.zeros((dicom_array.shape[0], target_h, target_w))
    order = INTERPOLATION_MAPPING[interpolation]
    for idx, dicom_slice in enumerate(dicom_array):
        resized_dicom[idx, :, :] = ndimage.zoom(dicom_slice, resize_factor, order=order, prefilter=False)
    return resized_dicom
