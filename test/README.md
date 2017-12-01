test subject
------------

input (.nii.gz or nrrd) and mask can be used as inputs to pycaso to test the code.

`dtiprep/` contains test data that has been run through unring and dtiprep before analysis.
`fsl/` contains test data that has been run through eddy_correct before analysis.

the outputs in `official_picaso_outputs/` are from the offical MATLAB distribution. the outputs of pycaso must match those stat maps (or be very close) to be considered correct.
