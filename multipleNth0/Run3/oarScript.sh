#!/bin/bash

EXEC_DIR=/home/kamran.karimi1/Project/git/Swarm/multipleNth0/Run3

papermill --prepare-only /home/kamran.karimi1/Project/git/Swarm/multipleNth0/Run3/DifferentThresholds.ipynb ./output.ipynb -p	n_thresh	1e-05 -p SWARM_PATH '/home/kamran.karimi1/Project/git/Swarm/dataset/Oklahoma/TableS1.csv'
jupyter nbconvert --execute /home/kamran.karimi1/Project/git/Swarm/multipleNth0/Run3/output.ipynb --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.allow_errors=True
