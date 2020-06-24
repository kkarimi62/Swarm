#!/bin/bash

EXEC_DIR=/newhome/kamran.karimi1/Project/git/Swarm/runRidgeCrest/Run0

papermill --prepare-only /newhome/kamran.karimi1/Project/git/Swarm/runRidgeCrest/Run0/RidgeCrest2019_recovered.ipynb ./output.ipynb -p quantile 0.05 -p SWARM_PATH '/newhome/kamran.karimi1/Project/git/Swarm/dataset/RidgeCrest/Dataset S2.txt' -p SWARM_PATH2 '/newhome/kamran.karimi1/Project/git/Swarm/dataset/RidgeCrest/momentTensor.csv'
jupyter nbconvert --execute /newhome/kamran.karimi1/Project/git/Swarm/runRidgeCrest/Run0/output.ipynb --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.allow_errors=True;ls output.html
