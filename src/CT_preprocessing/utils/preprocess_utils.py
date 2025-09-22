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
    array1 = array1 - array1.mean()
    array2 = array2 - array2.mean()
    cross_correlation = fftconvolve(array1, np.flip(array2), mode='same')

    array1_auto_correlation = np.sum(array1**2)
    ones_array1 = np.ones_like(array1)
    array2_auto_correlation = fftconvolve(array2**2, np.flip(ones_array1), mode='same')

    denominator = np.sqrt(array1_auto_correlation * array2_auto_correlation)
    denominator[denominator == 0] = 1e-12

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
