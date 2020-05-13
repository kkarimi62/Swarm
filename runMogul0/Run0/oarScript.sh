#!/bin/bash

EXEC_DIR=/home/kamran.karimi1/Project/git/Swarm/runMogul0/Run0

papermill --prepare-only /home/kamran.karimi1/Project/git/Swarm/runMogul0/Run0/Mogul2008.ipynb ./output.ipynb -p	Df	1.9 -p SWARM_PATH '/home/kamran.karimi1/Project/git/Swarm/dataset/Mogul2008/catalog.csv'
jupyter nbconvert --execute /home/kamran.karimi1/Project/git/Swarm/runMogul0/Run0/output.ipynb --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.allow_errors=True
