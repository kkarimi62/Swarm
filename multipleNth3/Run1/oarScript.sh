#!/bin/bash

EXEC_DIR=/home/kamran.karimi1/Project/git/Swarm/multipleNth3/Run1

papermill --prepare-only /home/kamran.karimi1/Project/git/Swarm/multipleNth3/Run1/DifferentThresholds.ipynb ./output.ipynb -p	bval	0.01 -p SWARM_PATH '/home/kamran.karimi1/Project/git/Swarm/dataset/Oklahoma/TableS1.csv'
jupyter nbconvert --execute /home/kamran.karimi1/Project/git/Swarm/multipleNth3/Run1/output.ipynb --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.allow_errors=True
