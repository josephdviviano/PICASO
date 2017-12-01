#!/bin/bash
# compares the results obtained with the unring+dtiprep pipeline vs the stock fsl pipeline
./pycasso.py test/dtiprep/input.nii.gz test/dtiprep/mask.nii.gz test/dtiprep/output.nii.gz -v
./pycasso.py test/fsl/input.nii.gz test/fsl/mask.nii.gz test/fsl/output.nii.gz -v
