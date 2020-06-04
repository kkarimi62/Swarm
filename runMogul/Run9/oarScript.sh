#!/bin/bash

EXEC_DIR=/home/kamran.karimi1/Project/git/Swarm/runMogul/Run9

papermill --prepare-only /home/kamran.karimi1/Project/git/Swarm/runMogul/Run9/Mogul2008.ipynb ./output.ipynb -p quantile 0.05 -p SWARM_PATH '/home/kamran.karimi1/Project/git/Swarm/dataset/Mogul2008/hypodd.reloc.mag'
jupyter nbconvert --execute /home/kamran.karimi1/Project/git/Swarm/runMogul/Run9/output.ipynb --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.allow_errors=True
