#!/bin/bash

EXEC_DIR=/home/kamran.karimi1/Project/git/Swarm/runMogul17/Run0

papermill --prepare-only /home/kamran.karimi1/Project/git/Swarm/runMogul17/Run0/Mogul2008.ipynb ./output.ipynb -p	Df	2.6999999999999997	-p	quantile	0.07 -p SWARM_PATH '/home/kamran.karimi1/Project/git/Swarm/dataset/Mogul2008/catsearch.14436'
jupyter nbconvert --execute /home/kamran.karimi1/Project/git/Swarm/runMogul17/Run0/output.ipynb --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.allow_errors=True;rm ./output.html
