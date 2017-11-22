pycaso
------
A python port of the MATLAB implementation for computing the PICASO dMRI measure. Automatically uses all cores available.

If you find this tool useful in your research, please cite:
Precise Inference and Characterization of Structural Organization (PICASO) of tissue from molecular diffusion. Lipeng Ning, Evren Özarslan, Carl-Fredrik Westin, Yogesh Rathia. Neuroimage 2017.

**matlab/**

Contains a backup of the original code form Lipeng, which will be removed when this code is verified to work properly.

**test/**

Contains an example input subject (and mask), in both `.nii.gz` and `.nrrd` format, as well as example outputs run through Lipeng's official MATLAB code. Can be used to ensure this python package is calculating the correct values.

Please contact Lipeng Ning at lning at bwh.harvard.edu with questions.

Requires numpy, scipy, pynrrd, nibabel, and python 2.7.
