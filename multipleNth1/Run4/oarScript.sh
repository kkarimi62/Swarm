#!/bin/bash

EXEC_DIR=/home/kamran.karimi1/Project/git/Swarm/multipleNth1/Run4

papermill --prepare-only /home/kamran.karimi1/Project/git/Swarm/multipleNth1/Run4/DifferentThresholds.ipynb ./output.ipynb -p	n_thresh	0.0001 -p SWARM_PATH '/home/kamran.karimi1/Project/git/Swarm/dataset/Oklahoma/TableS1.csv'
jupyter nbconvert --execute /home/kamran.karimi1/Project/git/Swarm/multipleNth1/Run4/output.ipynb --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.allow_errors=True
