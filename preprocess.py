import SimpleITK as sitk
import numpy as np


def read_dicom_series(directory):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(directory))
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    return image


def n4_bias_correction(input_image):
    input_image = sitk.Cast(input_image, sitk.sitkFloat32)
    mask_image = sitk.OtsuThreshold(input_image, 0, 1, 200)
    mask_image.CopyInformation(input_image)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected_image = corrector.Execute(input_image, mask_image)
    array = sitk.GetArrayFromImage(corrected_image)
    return array


def get_bbox(keypoints, expand_ratio=1.5):
    y_min, x_min = keypoints.min(0)
    y_max, x_max = keypoints.max(0)
    
    H = y_max - y_min
    y_margin = H * (expand_ratio-1.0) / 2
    H_expanded = H * expand_ratio

    y_min = round(y_min-y_margin)
    y_max = y_min + round(H_expanded)

    W = x_max - x_min
    x_margin = W * (expand_ratio-1.0) / 2
    W_expanded = W * expand_ratio

    x_min = round(x_min-x_margin)
    x_max = x_min + round(W_expanded)

    bbox = y_min, x_min, y_max, x_max
    return bbox


def crop(array, bbox):
    y_min, x_min, y_max, x_max = bbox

    y_min = max(y_min, 0)
    x_min = max(x_min, 0)

    y_max = min(y_max, array.shape[0])
    x_max = min(x_max, array.shape[1])

    array_cropped = array[y_min:y_max, x_min:x_max]
    return array_cropped