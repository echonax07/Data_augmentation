import numpy as np
import glob
import nibabel as nib
import os
import SimpleITK as sitk
import random


def resample1(image, transform):
    reference_image = image
    interpolator = sitk.sitkNearestNeighbor
    default_value = 0.0
    return sitk.Resample(image, reference_image, transform, interpolator, default_value)


def resample2(image, transform):
    reference_image = image
    interpolator = sitk.sitkCosineWindowedSinc
    default_value = 0.0
    return sitk.Resample(image, reference_image, transform, interpolator, default_value)


def get_rotation_transform(dim, angles):
    if not isinstance(angles, list):
        angles = [angles]
    assert isinstance(angles, list), 'Angles parameter must be a list of floats, one for each dimension.'
    assert len(angles) in [1, 3], 'Angles must be a list of length 1 for 2D, or 3 for 3D.'

    t = sitk.AffineTransform(dim)

    if len(angles) == 1:
        # 2D
        t.Rotate(0, 1, angle=angles[0])

    elif len(angles) > 1:
        # 3D
        # rotate about x axis
        t.Rotate(1, 2, angle=angles[0])
        # rotate about y axis
        t.Rotate(0, 2, angle=angles[1])
        # rotate about z axis
        t.Rotate(0, 1, angle=angles[2])

    return t


def get_scale_transform(dim, scale):
    if isinstance(scale, list) or isinstance(scale, tuple):
        assert len(scale) == dim, 'Length of scale must be equal to dim.'

    s = sitk.AffineTransform(dim)
    s.Scale(scale)

    return s


def get_translation_transform(offset):
    dimension = 3
    translation = sitk.TranslationTransform(dimension)
    translation.SetOffset(offset)
    return translation
