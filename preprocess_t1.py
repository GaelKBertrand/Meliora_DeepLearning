#!/usr/bin/env python

import nibabel as nib
import numpy as np
import glob
import nipype
from nipype.interfaces import niftyreg
import os
import tempfile
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse 

def get_args():
    parser = argparse.ArgumentParser(description='performs 12-DOF affine registration of a T1 image to ICBM152 template')
    parser.add_argument('-a', '--anat', type=str, help='input T1w nifti image')
    parser.add_argument('-o', '--out_dir', type=str, help='output directory')
    return parser.parse_args()

def registration(in_file, out_file, reference):
    temp = tempfile.TemporaryDirectory()
    ala = niftyreg.RegAladin(gpuid_val=0, platform_val=1, omp_core_val=20)
    ala.inputs.ref_file = reference
    ala.inputs.flo_file = in_file
    ala.inputs.res_file = os.path.join(temp.name, 'res.nii.gz')
    ala.inputs.aff_file = os.path.join(temp.name, 'aff.txt')
    result0 = ala.run()
    res = niftyreg.RegResample(inter_val="SINC")
    res.inputs.ref_file = reference
    res.inputs.flo_file = in_file
    res.inputs.trans_file = result0.outputs.aff_file
    res.inputs.out_file = out_file
    result = res.run()
    temp.cleanup()

def normalization(scan):
    scan = (scan - np.mean(scan)) / np.std(scan)
    return scan

def clip(scan):
    return np.clip(scan, -1, 2.5)

def back_remove(file, temp, new_path):

    if not os.path.exists(new_path):
        os.mkdir(new_path)

    data = np.load(file)
    new_data = data[:,:,:]

    stack = [(0, 0, 0), (180, 0, 0), (0, 216, 0), (180, 216, 0)]
    visited = set([(0, 0, 0), (180, 0, 0), (0, 216, 0), (180, 216, 0)])

    def valid(x, y, z):
        if x < 0 or x >= 181:
            return False
        if y < 0 or y >= 217:
            return False
        if z < 0 or z >= 181:
            return False
        return True

    while stack:
        x, y, z = stack.pop()
        for dx, dy, dz in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
            new_x, new_y, new_z = x + dx, y + dy, z + dz
            if valid(new_x, new_y, new_z) and (new_x, new_y, new_z) not in visited \
            and data[new_x, new_y, new_z] < -0.6 and temp[new_x, new_y, new_z] < 0.8:
                visited.add((new_x, new_y, new_z))
                new_data[new_x, new_y, new_z] = -10
                stack.append((new_x, new_y, new_z))

    filename = file.split('/')[-1]
    plt.subplot(131)
    plt.imshow(new_data[100, :, :])
    plt.subplot(132)
    plt.imshow(new_data[:, 100, :])
    plt.subplot(133)
    plt.imshow(new_data[:, :, 100])
    plt.savefig(os.path.join(new_path, filename.replace('npy', 'jpg')))
    plt.close()
    
    new_data = np.where(new_data==-10, -np.ones((181, 217, 181)), new_data).astype(np.float32)
    np.save(os.path.join(new_path, filename), new_data)


if __name__ == "__main__":
    reference = '/Meliora_DeepLearning/mni_icbm152_t1_tal_nlin_sym_09c_1mm_181x217x181.nii.gz'
    tmp = tempfile.TemporaryDirectory()
    args = get_args()
    in_anat = args.anat
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    out_name_nifti = os.path.basename(in_anat).split('.', 1)[0] + '.nii.gz'
    out_anat_nifti = os.path.join(tmp.name, out_name_nifti)
    registration(in_anat, out_anat_nifti, reference)
    in_img = nib.load(out_anat_nifti)
    out_name_npy = os.path.basename(in_anat).split('.', 1)[0] + '.npy'
    out_npy = os.path.join(out_dir, out_name_npy)
    data = np.array(in_img.dataobj)
    data = normalization(data)
    data = clip(data)
    np.save(out_npy, data)
    tmp.cleanup()
    template = np.load('/home/medetax/Applications/brain2020/brain_region.npy')
    back_remove(out_npy, template, out_dir)
