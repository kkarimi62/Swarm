#!/bin/bash

EXEC_DIR=/home/kamran.karimi1/Project/git/Swarm/runRidgeCrest0/Run0

papermill --prepare-only /home/kamran.karimi1/Project/git/Swarm/runRidgeCrest0/Run0/RidgeCrest2019_recovered.ipynb ./output.ipynb -p	n_thresh	0.001 -p SWARM_PATH3 '/home/kamran.karimi1/Project/git/Swarm/dataset/RidgeCrest/catalog_fmsearch.csv' -p SWARM_PATH4 '/home/kamran.karimi1/Project/git/Swarm/dataset/RidgeCrest/momentTensors.pkl'
jupyter nbconvert --execute /home/kamran.karimi1/Project/git/Swarm/runRidgeCrest0/Run0/output.ipynb --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.allow_errors=True;ls output.html
