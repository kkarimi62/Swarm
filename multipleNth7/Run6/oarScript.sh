#!/bin/bash

EXEC_DIR=/home/kamran.karimi1/Project/git/Swarm/multipleNth7/Run6

papermill --prepare-only /home/kamran.karimi1/Project/git/Swarm/multipleNth7/Run6/DifferentThresholds.ipynb ./output.ipynb -p	n_thresh	100.0 -p SWARM_PATH '/home/kamran.karimi1/Project/git/Swarm/dataset/Oklahoma/TableS1.csv'
jupyter nbconvert --execute /home/kamran.karimi1/Project/git/Swarm/multipleNth7/Run6/output.ipynb --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.allow_errors=True
