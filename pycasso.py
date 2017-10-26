#!/usr/bin/env python

import os, sys
import numpy as np
import nrrd as nd # also known as pynrrd
import nibabel as nib
import argparse


class DiffusionData:
    """Contains the various attributes of the input diffusion data"""
    def __init__(self, filename, maskname):

        self.filename = filename
        self.maskname = maskname
        self.data = None
        self.mask = None
        self.header = None
        self.gradients = None
        self.b = None
        self.x = None
        self.y = None
        self.z = None
        self.n = None
        self.idx_b0 = None
        self.idx_mask = None
        self.filetype = None
        self.normalized = False

        # returns gradients, index of b0 volumes, index of mask,
        # data and mask in x,y,z,n orientation,
        if self.is_nrrd():
            self.import_nrrd()
        else:
            self.import_nifti()


    def is_nrrd(self):
        if '.nrrd' in os.path.splitext(self.filename)[1]:
            return True
        else:
            return False


    def import_nrrd(self):
        """imports nrrd data using pynrrd"""
        self.filetype = 'nrrd'
        self.data, self.header = nd.read(self.filename)
        self.mask, _ = nd.read(self.maskname)

        # import diffusion matrix, gradient matrix, bval
        self.n, self.x, self.y, self.z = self.data.shape

        # converts the gradients as formatted in the nrrd header as a n_gradient
        # by direction (xyz) matrix.
        gradient_keys = self.header['keyvaluepairs'].keys()
        gradient_keys.sort()
        gradient_keys = filter(lambda x: 'gradient' in x, gradient_keys)

        self.gradients = []
        for key in gradient_keys:
            # converts whitespace delimited list to numpy vector
            self.gradients.append(
                np.array(self.header['keyvaluepairs'][key].split()).astype(np.float))
        self.gradients = np.vstack(self.gradients)

        # find b0 volumes
        self.idx_b0 = np.zeros(len(self.gradients)).astype(np.bool)
        for i, gradient in enumerate(self.gradients):
           if np.sum(np.abs(i)) == 0:
               self.idx_b0[i] = 1
           else:
               self.idx_b0[i] = 0

        # get bvalue from header, divided by 1000 (not clear why)
        self.b = float(self.header['keyvaluepairs']['DWMRI_b-value'])/1000

        # orientation: voxels x directions
        self.data = np.reshape(self.data, (self.n, self.x*self.y*self.z)).T
        self.mask = np.reshape(self.mask, (1, self.x*self.y*self.z)).T
        self.idx_mask = np.where(self.mask)[0]


    def import_nifti(self):
        """imports nifti data using nibabel"""
        self.filetype = 'nifti'
        file_nib = nib.load(self.filename)
        mask_nib = nib.load(self.maskname)

        self.data = file_nib.get_data()
        self.header = file_nib.header
        self.mask = mask_nib.get_data()

        self.x, self.y, self.z, self.n = file_nib.shape

        # import gradients and bvals, assume only name difference is extension
        if self.filename.endswith('nii.gz'):
            stem = os.path.splitext(os.path.splitext(filename)[0])[0]
        else:
            stem = os.path.splitext(filename)[0]

        self.gradients = np.genfromtxt('{}.bvec'.format(stem))
        self.b = np.genfromtxt('{}.bval'.format(stem)) / 1000 # not clear why we divide
        self.idx_b0 = ~self.b.astype(np.bool)

        # orientation: voxels x directions
        self.data = np.reshape(self.data, (self.x*self.y*self.z, self.n))
        self.mask = np.reshape(self.mask, (self.x*self.y*self.z, 1))
        self.idx_mask = np.where(self.mask)[0]


    def normalize(self):
        """
        takes the mean value across all b0 volumes, and then normalizes all
        diffusion values by the b0 estimate. Removes b0 volumes and gradients.
        """
        if not self.normalized:
            b0_mean = np.mean(self.data[:, self.idx_b0], axis=1)
            self.data = self.data[:, ~self.idx_b0]
            self.gradients = self.gradients[~self.idx_b0, :]
            self.data = self.data / np.atleast_2d(b0_mean).T
            self.data[np.isinf(self.data)] = 0
            self.data[np.isnan(self.data)] = 0


def model_picaso(s, u, b):
    """
    accepts s (normalized diffusion signal from one voxel),
            u (gradient direction vectors), and
            b (a vector of b-values)
    returns ?
    """
    idx = np.where(b>=1)





filename = '/archive/data/SPINS/pipelines/dtiprep/SPN01_CMH_0114_01/SPN01_CMH_0114_01_01_DTI60-1000_15_Ax-DTI-60plus5-20iso_QCed.nii.gz'
maskname =  '/archive/data/SPINS/pipelines/dtiprep/SPN01_CMH_0114_01/SPN01_CMH_0114_01_01_DTI60-1000_15_Ax-DTI-60plus5-20iso_QCed_B0_threshold_masked.nii.gz'

diff_data = DiffusionData(filename, maskname)

# calculate normalized diffusion values and return their gradients
data, gradients = normalize(data, gradients, idx_b0)


# initalize outputs
disturb_per = np.zeros((x*y*z))
disturb_par = np.zeros((x*y*z))
diff_per = np.zeros((x*y*z))
diff_par = np.zeros((x*y*z))

# fit picaso model per voxel
for i in idx:
    diff_vox = data[i, :].astype(np.float)
    a, b, coef = model_picaso(diff_vox, gradients, b)



