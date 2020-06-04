#!/bin/bash

EXEC_DIR=/home/kamran.karimi1/Project/git/Swarm/runFillmore/Run1

papermill --prepare-only /home/kamran.karimi1/Project/git/Swarm/runFillmore/Run1/Fillmore2015.ipynb ./output.ipynb -p quantile 0.05 -p SWARM_PATH '/home/kamran.karimi1/Project/git/Swarm/dataset/Fillmore2015/SRL-2016020_esupp_Table_S2.txt'
jupyter nbconvert --execute /home/kamran.karimi1/Project/git/Swarm/runFillmore/Run1/output.ipynb --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.allow_errors=True;rm output.html
