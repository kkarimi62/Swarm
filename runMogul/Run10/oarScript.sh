#!/bin/bash

EXEC_DIR=/home/kamran.karimi1/Project/git/Swarm/runMogul/Run10

papermill --prepare-only /home/kamran.karimi1/Project/git/Swarm/runMogul/Run10/Mogul2008.ipynb ./output.ipynb -p quantile 0.05 -p SWARM_PATH '/home/kamran.karimi1/Project/git/Swarm/dataset/Mogul2008/catsearch.14436'
jupyter nbconvert --execute /home/kamran.karimi1/Project/git/Swarm/runMogul/Run10/output.ipynb --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.allow_errors=True;rm output.html