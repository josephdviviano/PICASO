#!/usr/bin/env python

import os, sys
import numpy as np
import nrrd as nd # also known as pynrrd
import nibabel as nib
import argparse
import logging
from scipy.optimize import least_squares

logging.basicConfig(level=logging.WARN, format="[%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(os.path.basename(__file__))


class DiffusionData:
    """Contains the various attributes of the input diffusion data"""
    def __init__(self, filename, maskname, normalize=True):

        self.filename = filename
        self.maskname = maskname

        # images
        self.data = None
        self.mask = None
        self.clipmask = None

        # metadata
        self.header = None
        self.affine = None
        self.gradients = None
        self.b = None
        self.x = None
        self.y = None
        self.z = None
        self.n = None

        # masks
        self.idx_b0 = None

        # state variables
        self.filetype = None
        self.normalized = False
        self.clipped = False

        # returns gradients, index of b0 volumes, index of mask,
        # data and mask in x,y,z,n orientation,
        if self._is_nrrd():
            self._import_nrrd()
        else:
            self._import_nifti()

        self.data = self.data.astype(np.float)

        # normalizes data and removes impossible values (all data 0 < x < 1)
        # these impossible values typically caused by noise in the B0 image
        if normalize:
            self._normalize()
            self._clip()

    def _is_nrrd(self):
        if '.nrrd' in os.path.splitext(self.filename)[1]:
            return True
        else:
            return False


    def _import_nrrd(self):
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


    def _import_nifti(self):
        """imports nifti data using nibabel"""
        self.filetype = 'nifti'
        file_nib = nib.load(self.filename)
        mask_nib = nib.load(self.maskname)

        self.data = file_nib.get_data()
        self.header = file_nib.header
        self.affine = file_nib.affine
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


    def _normalize(self):
        """
        takes the mean value across all b0 volumes, and then normalizes all
        diffusion values by the b0 estimate. Removes b0 volumes and gradients.
        """
        if not self.normalized:
            # seperate b0 mean from gradient direction volumes
            b0_mean = np.mean(self.data[:, self.idx_b0], axis=1)
            self.data = self.data[:, ~self.idx_b0]
            self.gradients = self.gradients[~self.idx_b0, :]
            self.b = self.b[~self.idx_b0]

            # normalize gradient direction volumes by b0 mean
            self.data = self.data / np.atleast_2d(b0_mean).T

            # set all infs and nans to 0
            self.data[np.isinf(self.data)] = 0
            self.data[np.isnan(self.data)] = 0

            self.normalized = True


    def _clip(self):
        """clips all values to be 0 < x < 1"""
        if not self.clipped:
            self.clipmask = np.zeros((self.x*self.y*self.z, self.data.shape[1]))
            self.data[(self.mask == 0).flatten(), :] = 0 # mask non-brain regions

            idx = self.data > 1
            logger.debug('removing {} values > 1'.format(np.sum(idx)))
            self.data[idx] = 1
            self.clipmask[idx] = 1

            idx = self.data < 0
            logger.debug('removing {} values < 0'.format(np.sum(idx)))
            self.data[idx] = 0
            self.clipmask[idx] = 1

            self.clipmask = np.atleast_2d(np.sum(self.clipmask, axis=1)).T
            self.clipped = True


    def _write(self, data, filename):
        if self.filetype == 'nifti':
            try:
                data = np.reshape(data, (self.x, self.y, self.z, data.shape[1]))
            except:
                raise IndexError('input data does not have the same x,y,z dimensions as input DWI file')

            output = nib.nifti1.Nifti1Image(data, self.affine, self.header)
            output.update_header()
            output.header_class(extensions=())
            output.to_filename(filename)


def null(a, rtol=1e-5):
    """
    is an orthonormal basis for the null space of A obtained from the singular
    value decomposition. That is, a.dot(n) has negligible elements,
    np.shape(n, 2) is the nullity of a, and n.T.dot(n) = I.
    """
    u, s, v = np.linalg.svd(a)
    rank = (s > rtol*s[0]).sum()
    n = v[rank:].T.copy()

    return n


def estimate_tensor(x, g):
    """
    computes DTI tensor using q values and the normalized dMRI signals.
    details in: Introduction to diffusion tensor imaging mathematics: part III.
    Tensor calculation, noise, simulations, and optimization. Peter B. Kingsley.
    2005. Concepts in Magnetic Resonance Part A, Vol 28A(2)

    x = input diffusion weightings for a voxel
    g = normalized gradient components
    """
    idx = np.where(x > 1e-4)[0] # get rid of extremely small diff weighted vols
    x = x[idx]                  #
    g = g[idx, :]               #

    # apparent diffusion coefficients (ADCs), eqn 8
    x = -np.log(x)

    # represents tensor [Dxx, Dyy, Dzz, Dxy, Dxz, Dyz], eqn 10
    d = np.array([g[:, 0]**2,
                  g[:, 1]**2,
                  g[:, 2]**2,
                  g[:, 0]*g[:, 1],
                  g[:, 0]*g[:, 2],
                  g[:, 1]*g[:, 2]]).T

    # TODO: DOUBLE CHECK THIS LINE
    # calculate pseudoinverse of H, multiply with data s, eqn. 37
    # collapses X gradient directions down to 9 numbers that represent cardinal
    # directions x,y,z
    d = np.linalg.pinv(d.T.dot(d)).dot(d.T).dot(np.atleast_2d(x).T)
    d = np.real(d)

    # initial diffusion matrix
    D = np.array([[d[0], d[3], d[4]],
                  [d[3], d[1], d[5]],
                  [d[4], d[5], d[2]]])

    # find the eigenvalues, eigenvectors of d
    # NB: order of outputs from eig is flipped w.r.t. MATLAB
    e, U = np.linalg.eig(D[:, :, 0]) # I have numpy matrix issues, hence the 0
    e[e < 5e-6] = 5e-6 # ensure all values are positive (clip near zero)
    e = np.diag(e) # makes square with off-diagnonal of zero

    # final pseudoinverse of H, eqn. 47
    D = U.dot(e).dot(U.T)

    return D


def model_bloch_to_rrey(x, u, b, v):
    """TODO: get docstring from Lipeng"""
    v_perp = null(v.T)
    V = np.hstack((v, v_perp))
    U2D  = V.dot(np.diag([x[0], x[1], x[1]])).dot(V.T)
    diff = V.dot(np.diag([x[2], x[3], x[3]])).dot(V.T)
    signal = np.sum(u.dot(U2D)*u, axis=1) +
        (1-np.sum(u.dot(U2D)*u, axis=1)) * np.exp(-b*np.sum(u.dot(diff)*u, axis=1))

    return signal


def model_picaso(x, g, b):
    """
    accepts x (normalized diffusion signal from one voxel),
            g (gradient direction vectors), and
            b (a vector of b-values)
    returns ?

    Computes the 'structural disturbance' or 'axon density and volume' measure
    as well as the mean diffusivity in the directions parallel and perpendicular
    to the fiber orientation. (U2_parallel, U2_perp, D_parallel, D_perp).
    """
    g = np.tile(np.sqrt(b), (3, 1)).T * g # modulates gradients by sqrt of bval
    T = estimate_tensor(x, g)
    U, S, V = np.linalg.svd(T)
    u = U[:, 0] # first eig

    fit = least_squares() ## lost -- where does @fun come from?

    import IPython; IPython.embed()


logger.setLevel(logging.DEBUG)

filename = '/archive/data/SPINS/pipelines/dtiprep/SPN01_CMH_0114_01/SPN01_CMH_0114_01_01_DTI60-1000_15_Ax-DTI-60plus5-20iso_QCed.nii.gz'
maskname =  '/archive/data/SPINS/pipelines/dtiprep/SPN01_CMH_0114_01/SPN01_CMH_0114_01_01_DTI60-1000_15_Ax-DTI-60plus5-20iso_QCed_B0_threshold_masked.nii.gz'

# normalizes and clips input data by default
diff = DiffusionData(filename, maskname)

# initalize outputs
disturb_per = np.zeros((diff.x*diff.y*diff.z, 1))
disturb_par = np.zeros((diff.x*diff.y*diff.z, 1))
diff_per = np.zeros((diff.x*diff.y*diff.z, 1))
diff_par = np.zeros((diff.x*diff.y*diff.z, 1))

idx = np.where(diff.mask == 1)[0]

# fit picaso model per voxel
for i in idx:
    diff_vox = diff.data[i, :]
    a, b, coef = model_picaso(diff_vox, diff.gradients, diff.b)



