# -*- coding: utf-8 -*-
# pylint: disable=W0621,C0301,C0303,C0103
"""
Created on Mon May 24 15:21:01 2021

@author: Muhammed Patel
email: muhammed.patel@philips.com

"""

import argparse
import copy
import datetime
import glob
from collections import OrderedDict
import pickle
from shutil import copyfile

import nibabel as nib
import numpy as np
import torch
import torchio as tio
from tqdm import tqdm


import os
import SimpleITK as sitk
import random
from utils import get_scale_transform, get_translation_transform, get_rotation_transform

torch.set_grad_enabled(False)
torch.manual_seed(20202021)
np.random.seed(0)

print('TorchIO version:', tio.__version__)
print('Last run:', datetime.date.today())


class Policy():

    def __init__(self, number):
        """
        A class containing various transformation policies

        Parameters
        ----------
        number : int or string
            The policy no
            For implementing all the policies, set --policies='all' in CLI.

        Returns
        -------
        instance of Policy class which contains all the policies

        """
        self.number = number

        control_point = np.random.randint(low=8, high=12, size=3).tolist()
        max_displacement = np.random.randint(low=8, high=12, size=3).tolist()

        self.non_spatial_transforms = {tio.RandomGhosting(num_ghosts=(2, 3)): 5,
                                       tio.transforms.RandomSpike(num_spikes=(1, 3), intensity=(0.1, 0.7)): 10,
                                       tio.RandomBiasField(coefficients=(0.1, 0.4), order=3): 40,
                                       tio.RandomNoise(): 35,
                                       tio.transforms.RandomMotion(degrees=(-10, 10), translation=(-10, 10),
                                                                   num_transforms=2,
                                                                   image_interpolation='linear'): 30,
                                       tio.transforms.RandomBlur(std=(0.7, 1.2)): 10,

                                       }
        self.spatial_transforms = {
            tio.transforms.RandomAffine(scales=(0.9, 1.1, 0.9, 1.1, 0.9, 1.1),
                                        degrees=(-10, 10), translation=(-5, 5)): 30,
            tio.transforms.RandomElasticDeformation(num_control_points=control_point,
                                                    max_displacement=max_displacement, locked_borders=2): 35,
            tio.transforms.RandomAnisotropy(axes=(0), downsampling=(1, 2)): 35
        }

        if number == 2:
            self.is_spatial_transforms = False
        else:
            self.is_spatial_transforms = True

    def get_transforms_operation(self):
        """


        Returns
        -------
        DA_operations : instance of torchio.transforms.Compose
            Returns the Transforms_operation function which can be applied to
            the images

        """
        if self.number == 1:
            DA = [
                tio.OneOf(self.spatial_transforms),
                tio.OneOf(self.non_spatial_transforms)
            ]

            DA_operations = tio.Compose(DA)
        elif self.number == 2:
            DA = [
                tio.OneOf(self.non_spatial_transforms)
            ]

            DA_operations = tio.Compose(DA)

        elif self.number == 3:
            DA = [
                tio.OneOf(self.spatial_transforms)
            ]

            DA_operations = tio.Compose(DA)
        elif self.number == 4:
            control_point = np.random.randint(low=8, high=12, size=3).tolist()
            max_displacement = np.random.randint(
                low=8, high=12, size=3).tolist()
            RED = {
                tio.transforms.RandomElasticDeformation(num_control_points=control_point,
                                                        max_displacement=max_displacement, locked_borders=2): 35,
            }
            DA = [
                tio.OneOf(RED)
            ]

            DA_operations = tio.Compose(DA)            
        else:
            raise Exception('Undefined Policy')
        return DA_operations


def get_ijk(fcsv_files, affine_mat):
    """
    Get the ijk cordinates from a .fcsv file

    Parameters
    ----------
    fcsv_files : str
        The file path to the fcsv file
    affine_mat : np.array
        The affine matrix

    Returns
    -------
    ijk_array : np.array
        Returns the landmarks in IJK  cordinates

    """
    landmark_path = open(fcsv_files, 'r')
    points = landmark_path.readlines()
    points = points[3:]

    IJK_points = []

    def rasMatrix(i, j, k): return np.array([[i], [j], [k], [1]])

    inverse_affine_matrix = np.linalg.inv(affine_mat)

    for k in points:
        temp_ras = k.split(',')[1: 4]
        temp_ras = list(map(float, temp_ras))

        temp_ijk = np.dot(inverse_affine_matrix, rasMatrix(
            temp_ras[0], temp_ras[1], temp_ras[2]))
        temp_ijk = list(map(int, temp_ijk))
        temp_ijk = np.transpose(temp_ijk).tolist()
        IJK_points.append(temp_ijk[:3])

    ijk_array = np.array(IJK_points)
    return ijk_array


def tio_To_nifty(tioImage, img_path, output_path, transforms, policy_no):
    """
    Saves the TIO image to Nifty file

    Parameters
    ----------
    tioImage : tio.Image
    img_path : The input path to the nifty image
    output_path : The output path to save the image
    transforms : The list of Transforms operations
    policy_no : int
        The policy no

    Returns
    -------
    None.

    """
    tioImage_np = tioImage.numpy().squeeze()
    tioImage_affine = tioImage.affine
    empty_header = nib.Nifti1Header()
    new_img = nib.Nifti1Image(tioImage_np, tioImage_affine, empty_header)
    list2str = ''.join(transforms)
    file_name =os.path.basename(img_path).split('.')[0]
    out_file=os.path.join(output_path, file_name +
             '_DA'+str(policy_no) + '_' + list2str + '.nii')
    nib.save(new_img,out_file)
    print('File Saved successfully at: ',
          out_file)


def get_RAS(ijk_cordinates, affine_matrix, img_path, transforms, policy_no, output_path, save_fcsv=True):
    """
    Get RAS cordinates from IJK cordinate and save them in an FCSV file

    Parameters
    ----------
    ijk_cordinates : np.array
    affine_matrix : np.array
    img_path : str
    transforms : the list of transforms applied
    policy_no : int
        The policy No
    output_path : str
        The path to save the fcsv file
    save_fcsv : Bool, optional
        Whether to save the fcsv or not. The default is True.

    Returns
    -------
    np.array
        The RAS cordinates

    """
    ijk_points = ijk_cordinates
    RAS_points = []
    slicer_coordinates = ['# Markups fiducial file version = 4.10\n',
                          '# CoordinateSystem = 0\n',
                          '# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n', ]

    def ijkMatric(i, j, k):
        return np.array([[i], [j], [k], [1]])

    Node_ID = 1
    file_name = os.path.basename(img_path).split('.')[0]
    for temp_point in ijk_points:
        temp_point = np.dot(affine_matrix, ijkMatric(
            temp_point[0], temp_point[1], temp_point[2]))
        temp_point = np.transpose(temp_point).tolist()
        RAS_points.append(temp_point[0][:3])
        revised = 'vtkMRMLMarkupsFiducialNode_{},{},{},{},0.000,0.000,0.000,1.000,1,1,0,F-{},vtkMRMLScalarVolumeNode1\n'.format(
            Node_ID, temp_point[0][0], temp_point[0][1], temp_point[0][2], Node_ID)
        slicer_coordinates.append(revised)
        Node_ID += 1

    list2str = ''.join(transforms)
    if save_fcsv:
        out_file=os.path.join(output_path,file_name+'_DA' +
                     str(policy_no)+'_' + list2str + '.fcsv')
        write = open(out_file, "w")
        write.writelines(slicer_coordinates)
        write.close()
    return RAS_points


def save_fcsv_non_spatial(fscv_path, output_path, policy_no, transforms, save_fcsv):
    """
    Copies fcsv file from the fcsv path to the output path

    Parameters
    ----------
    fscv_path : str
        The path to fcsv.
    output_path : str
        Output path.
    policy_no : int
        The policy no.
    transforms : list
        The list containing the transformation names.
    save_fcsv : Bool
        Whether to save fcsv or not.

    Returns
    -------
    None.

    """
    file_name = os.path.basename(fscv_path).split('.')[0]
    list2str = ''.join(transforms)
    if save_fcsv:
        out_file=os.path.join(output_path,file_name+'_DA' +
                 str(policy_no)+'_' + list2str + '.fcsv')
        copyfile(fscv_path,out_file)


def create_LMS_volume(ijk, img_np):
    """
    Creates Faux volume from the ijk cordinates of Ladmarks
    Parameters
    ----------
    ijk : np.array
        The ijk cordinates of the landmarks
    img_np : np.array
        The orignal nii file in numpy

    Returns
    -------
    dict_LM_vols : Dictionary
        The Faux volumes in dictionary form
        key: landmark_no
        value: the faux volume

    """
    keys = ['LM' + str(i + 1) for i in range(len(ijk))]
    dict_LM_vols = OrderedDict()
    vol_tempelate = np.full_like(img_np, 0)
    for key, cord in zip(keys, ijk):
        temp_LM = copy.deepcopy(vol_tempelate)
        temp_LM[cord[0] - 1:cord[0] + 2, cord[1] -
                1:cord[1] + 2, cord[2] - 1:cord[2] + 2] = 750
        temp_LM[cord[0], cord[1], cord[2]] = 1500
        dict_LM_vols[key] = temp_LM
    return dict_LM_vols


def get_non_zero_indices(subject, keys):
    """


    Parameters
    ----------
    subject : tio.Subject
    keys : list
        list which represents the landmarks

    Returns
    -------
    transformed_cords : List
        The non-zero cordinates of the faux voluumes after applying the transformation 
        operation on the faux voluumes 

    """
    transformed_cords = []
    max_val = []
    for key in keys:
        try:
            # cord=np.argmax(subject[key].numpy().squeeze(),axis=0)
            cord = np.unravel_index(subject[key].numpy().squeeze(
            ).argmax(), subject[key].numpy().squeeze().shape)
            transformed_cords.append(list(cord))
            max_val.append(subject[key].numpy().squeeze().max())
        except:
            transformed_cords.append(list(cord))
    return transformed_cords


def get_TorchIO_transform(img_path, fcsv_path, output_path, policy, save_NIFTY=True, save_fcsv=True):
    """


    Parameters
    ----------
    img_path : str
        The path to the nifty file.
    fcsv_path : str
        The path to the fcsv file.
    output_path : str
        The output path to save the outputs .
    policy : an instance of Policy class
    save_NIFTY : Bool, optional
        whether to save the nifty or not. The default is True.
    save_fcsv : Bool, optional
        whether to save the nifty or not. The default is True.

    Returns
    -------
    mri_np : np.array
        The transformed mri image in numpy format 

    """
    img = nib.load(img_path)
    img_np = img.get_fdata()
    is_spatial_transform = policy.is_spatial_transforms
    # mri = tio.ScalarImage(tensor=img_np, affine=img.affine)
    if is_spatial_transform:
        # print("Creating Faux Volume")
        ijk = get_ijk(fcsv_path, img.affine)
        dict_LM_vols = create_LMS_volume(ijk, img_np)
        keys = ['LM' + str(i + 1) for i in range(len(ijk))]
        # dict_LM_vols=create_LMS_volume(ijk,img_np.squeeze())

    # lm37_np=dict_LM_vols['LM37']

    img_np = np.expand_dims(img_np, axis=0)

    subject_dict = OrderedDict()
    subject_dict['mri'] = tio.ScalarImage(img_path)
    if is_spatial_transform:
        #print("Creating Subject Dict")
        for key in keys:
            subject_dict[key] = tio.LabelMap(
                tensor=np.expand_dims(dict_LM_vols[key], axis=0),
                affine=img.affine
            )
        del dict_LM_vols

    subject = tio.Subject(subject_dict)
    if is_spatial_transform:
        del subject_dict

    transform_operations = policy.get_transforms_operation()
    policy_no = policy.number
    print('Applying Transforms operation')
    subject = transform_operations(subject)
    transform_list = []

    for hist in subject.history:
        transform_list.append('Random' + hist.name)
    print("Applied The following transformation", transform_list)
    mri_np = subject.mri.numpy()
    if save_NIFTY:
        tio_To_nifty(subject.mri, img_path, output_path,
                     transform_list, policy_no)
    # landmark = transform_operations(landmark)

    if is_spatial_transform:
        print("Saving Landmark Volume")
        indices_transformed = get_non_zero_indices(subject, keys)
        _ = get_RAS(indices_transformed, img.affine, img_path,
                    transform_list, policy_no, output_path, save_fcsv)
    else:
        print("Saving fcsv file")
        save_fcsv_non_spatial(fcsv_path, output_path,
                              policy_no, transform_list, save_fcsv)

    return mri_np


def resample1(image, transform):
    """

    Parameters
    ----------
    image : Sitk Image
    transform : Sitk operation

    Returns
    -------
    Transformed Image with Nearest Neighbour Interpolator

    """
    reference_image = image
    interpolator = sitk.sitkNearestNeighbor
    default_value = 0.0
    return sitk.Resample(image, reference_image, transform, interpolator, default_value)

def resample2(image, transform):
    """

    Parameters
    ----------
    image : Sitk Image
    transform : Sitk Operation

    Returns
    -------
    Transformed Image with Cosine Window Interpolator

    """
    reference_image = image
    interpolator = sitk.sitkCosineWindowedSinc
    default_value = 0.0
    return sitk.Resample(image, reference_image, transform, interpolator, default_value)


def transform_points(image, pf, op):
    """

    Parameters
    ----------
    image : sitk Image
    pf : Point to transform
    op : sitk operation to perform

    Returns
    -------
    Transformed point

    """
    P = [int(x) for x in pf]
    point = sitk.Image(image.GetSize(), sitk.sitkUInt16)
    point.SetOrigin(image.GetOrigin())
    point.SetSpacing(image.GetSpacing())
    point.SetDirection(image.GetDirection())
    nda = sitk.GetArrayFromImage(image)
    mm = nda.max()
    point[P[0] - 1:P[0] + 2, P[1] - 1:P[1] + 2, P[2] - 1:P[2] + 2] = int(mm) // 2
    point[P[0], P[1], P[2]] = int(mm)
    transformed_point = resample1(point, op)
    a = sitk.GetArrayFromImage(transformed_point)
    ind = np.unravel_index(np.argmax(a, axis=None), a.shape)
    return [ind[2], ind[1], ind[0]]


def rasMatrix(i, j, k):
    A = np.array([[i], [j], [k], [1]])
    return A


def ijkMatric(i, j, k):
    A = np.array([[i], [j], [k], [1]])
    return A


def scale(image_path, write_path, Points, name, save_NIFTY):
    """

    Parameters
    ----------
    image_path : str
        Path to image file
    write_path : str
        Path to output folder
    Points : list
        List of points to transform
    name : str
        Basename of the image file
    save_NIFTY : bool
        Whether to save the transformed nifty image
    Returns
    -------
        Path to the saved nifty file, Path to the saved text file

    """
    im = sitk.ReadImage(image_path)
    scale = [scale_range[0] + random.random() * (scale_range[1] - scale_range[0]),
             scale_range[0] + random.random() * (scale_range[1] - scale_range[0]),
             scale_range[0] + random.random() * (scale_range[1] - scale_range[0])]
    op = get_scale_transform(3, scale)
    transformed = resample2(im, op)

    loc_w = os.path.join(write_path, name + "_scale.txt")

    fw = open(loc_w, 'w')
    for i in range(len(Points)):
        P = Points[i]
        P_t = transform_points(im, P, op)
        text = str(P_t[0]) + "," + str(P_t[1]) + "," + str(P_t[2]) + "\n"
        fw.write(text)

    fw.close()

    if save_NIFTY:
        writer = sitk.ImageFileWriter()
        writer.SetFileName(os.path.join(write_path, name + "_scale.nii"))
        writer.Execute(transformed)

    return os.path.join(write_path, name + "_scale.nii"), loc_w


def translate(image_path, write_path, Points, name, save_NIFTY):
    """

    Parameters
    ----------
    image_path : str
        Path to image file
    write_path : str
        Path to output folder
    Points : list
        List of points to be transformed
    name : str
        Basename of the image file
    save_NIFTY : bool
        Whether to save the transformed nifty image
    Returns
    -------
        Path to the saved nifty file, Path to the saved text file
    """
    im = sitk.ReadImage(image_path)
    trans = [random.randint(trans_range[0], trans_range[1]), random.randint(
        trans_range[0], trans_range[1]), random.randint(trans_range[0], trans_range[1])]
    op = get_translation_transform(trans)
    transformed = resample2(im, op)

    loc_w = os.path.join(write_path, name+"_trans.txt")

    fw = open(loc_w, 'w')
    for i in range(len(Points)):
        P = Points[i]
        P_t = transform_points(im, P, op)
        text = str(P_t[0])+","+str(P_t[1])+","+str(P_t[2])+"\n"
        fw.write(text)
    fw.close()

    if save_NIFTY:
        writer = sitk.ImageFileWriter()
        writer.SetFileName(os.path.join(write_path, name+"_trans.nii"))
        writer.Execute(transformed)

    return os.path.join(write_path, name+"_trans.nii"), loc_w


def rotate(image_path, write_path, Points, name, save_NIFTY):
    """

    Parameters
    ----------
    image_path : str
        Path to image file
    write_path : str
        Path to output folder
    Points : list
        List of points to transform
    name : str
        Basename of the image file
    save_NIFTY : bool
        Whether to save the nifty image
    Returns
    -------
        Path to the saved nifty file, Path to the saved text file
    """
    im = sitk.ReadImage(image_path)

    angle_min = angle_range[0]
    angle_max = angle_range[1]
    angle = [random.randint(angle_min, angle_max), random.randint(angle_min, angle_max),
             random.randint(angle_min, angle_max)]
    op = get_rotation_transform(
        3, [np.deg2rad(angle[0]), np.deg2rad(angle[1]), np.deg2rad(angle[2])])
    transformed = resample2(im, op)
    loc_w = os.path.join(write_path, name + "_rot.txt")
    fw = open(loc_w, 'w')
    Transformed_points = []
    for i in range(len(Points)):
        P = Points[i]
        P_t = transform_points(im, P, op)
        Transformed_points.append(P_t)
        text = str(P_t[0]) + "," + str(P_t[1]) + "," + str(P_t[2]) + "\n"
        fw.write(text)
    fw.close()

    if save_NIFTY:
        writer = sitk.ImageFileWriter()
        writer.SetFileName(os.path.join(write_path, name + "_rot.nii"))
        writer.Execute(transformed)
    return os.path.join(write_path, name+"_rot.nii"), loc_w


def get_SITK_transform(img_path, fcsv_path, output_path, save_NIFTY=True, save_fcsv=True):
    """

    Parameters
    ----------
    img_path : str
        Path to image file
    fcsv_path : str
        Path to fcsv file (slicer coordinate system)
    output_path : str
        Path to the output folder
    save_NIFTY : bool
        Whether to save the nifty file or not
    save_fcsv : bool
        Whether to save the fcsv file or not
    """
    name = os.path.basename(fcsv_path)[:-5]
    img = nib.load(img_path)
    affine_matrix = img.affine
    inverse_affine_matrix = np.linalg.inv(affine_matrix)

    landmark_path = open(fcsv_path, 'r')
    points = landmark_path.readlines()
    points_updated = points[3:]

    IJK_points = []
    save_points = []

    for k in points_updated:
        temp_ras = k.split(',')[1:4]
        temp_ras = list(map(float, temp_ras))

        temp_ijk = np.dot(inverse_affine_matrix, rasMatrix(
            temp_ras[0], temp_ras[1], temp_ras[2]))
        temp_ijk = list(map(int, temp_ijk))
        temp_ijk = np.transpose(temp_ijk).tolist()
        IJK_points.append(temp_ijk[:3])

        save_point = '{},{},{}\n'.format(temp_ijk[0], temp_ijk[1], temp_ijk[2])
        save_points.append(save_point)

    decider = random.random()
    if decider < rotation_prob:
        img_path, landmark_file = rotate(
            img_path, output_path, IJK_points, name, save_NIFTY)
    elif decider < rotation_prob + scale_prob:
        img_path, landmark_file = scale(
            img_path, output_path, IJK_points, name, save_NIFTY)
    else:
        img_path, landmark_file = translate(
            img_path, output_path, IJK_points, name, save_NIFTY)

    img = nib.load(img_path)
    affine_matrix = img.affine

    landmark_path = open(landmark_file, 'r')
    points = landmark_path.readlines()

    RAS_points = []
    slicer_coordinates = ['# Markups fiducial file version = 4.10\n',
                          '# CoordinateSystem = 0\n',
                          '# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n', ]
    Node_ID = 1
    for j in points:
        temp_point = j.replace('\n', '').split(',')
        temp_point = list(map(int, temp_point))
        temp_point = np.dot(affine_matrix, ijkMatric(
            temp_point[0], temp_point[1], temp_point[2]))
        temp_point = np.transpose(temp_point).tolist()
        RAS_points.append(temp_point[0][:3])
        revised = 'vtkMRMLMarkupsFiducialNode_{},{},{},{},0.000,0.000,0.000,1.000,1,1,0,F-{},vtkMRMLScalarVolumeNode1\n'.format(
            Node_ID, temp_point[0][0], temp_point[0][1], temp_point[0][2], Node_ID)
        slicer_coordinates.append(revised)
        Node_ID += 1

    fsv_path = landmark_file.replace('.txt', '.fcsv')
    fsv_path = os.path.join(output_path,os.path.basename(fsv_path))
    print(fsv_path)
    os.remove(landmark_file)
    if save_fcsv:
        write = open(fsv_path, "w")
        write.writelines(slicer_coordinates)
        write.close()
        print('File saved successfully')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nifty_input_path',
                        help='Input_path to Nifty File', required=True)
    parser.add_argument('--fcsv_input_path',
                        help='Input_path to FCSV File', required=True)
    parser.add_argument(
        '--output_path', help='Output_path to Nii and FCSV File', required=True)
    parser.add_argument('--policy', help='policy no', required=True)
    args = parser.parse_args()
    nifty_input_path = args.nifty_input_path
    fcsv_input_path = args.fcsv_input_path
    output_path = args.output_path
    policy = args.policy
    print("Polciy: ", policy)
    nii_files = sorted(glob.glob(nifty_input_path + '//' + "*.nii"))
    fcsv_files = sorted(glob.glob(fcsv_input_path + '//' + "*.fcsv"))
    print('Found {} Nifty Files'.format(len(nii_files)))

    count = 0
    skipped_files = []
    for nii, fcsv in tqdm(zip(nii_files, fcsv_files), total=len(nii_files)):
        if policy == str(5):
            print('Policy: ', policy)
            angle_range = [-15, 15]
            scale_range = [0.8, 1.2]
            trans_range = [-10, 10]
            rotation_prob = 0.6
            scale_prob = 0.2
            trans_prob = 0.2
            get_SITK_transform(nii, fcsv,
                               output_path, save_NIFTY=True, save_fcsv=True)

        else:
            print('Going here')
            if policy != 'all':
                # Implements the SITK transforms if policy=5 is given
                P = Policy(int(policy))
            else:
                rnd = np.random.random()
                if rnd <= 0.2:
                    P = Policy(1)
                    print('### Applying Policy:', 1)
                elif 0.2 < rnd <= 0.45:
                    P = Policy(2)
                    print('### Applying Policy:', 2)
                elif 0.45 < rnd <= 0.70:
                    P = Policy(3)
                    print('### Applying Policy:', 3)
                else:
                    P = Policy(4)
                    print('### Applying Policy:', 4)
            try:
                mri_np = get_TorchIO_transform(nii, fcsv, output_path,
                                               P, save_NIFTY=True, save_fcsv=True)
    
                print('Done File: {} {}/{}'.format(nii, count + 1, len(nii_files)))
                count += 1
            except:
                print("Skipped the file: ", nii)
                skipped_files.append(nii)
                count += 1
    with open('skipped_files.pkl', 'wb') as f:
        pickle.dump(skipped_files, f)
