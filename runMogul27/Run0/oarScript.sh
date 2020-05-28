#!/bin/bash

EXEC_DIR=/home/kamran.karimi1/Project/git/Swarm/runMogul27/Run0

papermill --prepare-only /home/kamran.karimi1/Project/git/Swarm/runMogul27/Run0/Mogul2008.ipynb ./output.ipynb -p	Df	2.9	-p	quantile	0.11 -p SWARM_PATH '/home/kamran.karimi1/Project/git/Swarm/dataset/Mogul2008/catsearch.14436'
jupyter nbconvert --execute /home/kamran.karimi1/Project/git/Swarm/runMogul27/Run0/output.ipynb --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.allow_errors=True;rm ./output.html