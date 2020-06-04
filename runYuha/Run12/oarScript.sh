#!/bin/bash

EXEC_DIR=/home/kamran.karimi1/Project/git/Swarm/runYuha/Run12

papermill --prepare-only /home/kamran.karimi1/Project/git/Swarm/runYuha/Run12/swarmYuhaDesert.ipynb ./output.ipynb -p quantile 0.05 -p SWARM_PATH '/home/kamran.karimi1/Project/git/Swarm/dataset/YuhaDesert/EMC.csv'
jupyter nbconvert --execute /home/kamran.karimi1/Project/git/Swarm/runYuha/Run12/output.ipynb --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.allow_errors=True