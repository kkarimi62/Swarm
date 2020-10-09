#!/bin/bash

EXEC_DIR=/home/kamran.karimi1/Project/git/Swarm/swarmYuhaDesertFirstPeriod1/Run0

papermill --prepare-only /home/kamran.karimi1/Project/git/Swarm/swarmYuhaDesertFirstPeriod1/Run0/swarmYuhaDesertFirstPeriod.ipynb ./output.ipynb -p BVALL 1.30 -p MCC 3.0 -p SWARM_PATH '/home/kamran.karimi1/Project/git/Swarm/dataset/YuhaDesert/EMC.csv'
jupyter nbconvert --execute /home/kamran.karimi1/Project/git/Swarm/swarmYuhaDesertFirstPeriod1/Run0/output.ipynb --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.allow_errors=True;ls output.html
