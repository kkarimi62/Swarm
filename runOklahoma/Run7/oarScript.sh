#!/bin/bash

EXEC_DIR=/home/kamran.karimi1/Project/git/Swarm/runOklahoma/Run7

papermill --prepare-only /home/kamran.karimi1/Project/git/Swarm/runOklahoma/Run7/swarmOklahoma.ipynb ./output.ipynb -p quantile 0.05 -p SWARM_PATH '/home/kamran.karimi1/Project/git/Swarm/dataset/Oklahoma/TableS1.csv'
jupyter nbconvert --execute /home/kamran.karimi1/Project/git/Swarm/runOklahoma/Run7/output.ipynb --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.allow_errors=True;ls output.html
