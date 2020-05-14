#!/bin/bash

EXEC_DIR=/home/kamran.karimi1/Project/git/Swarm/runFillmore2015/Run3

papermill --prepare-only /home/kamran.karimi1/Project/git/Swarm/runFillmore2015/Run3/Fillmore2015.ipynb ./output.ipynb -p Df 2.5 -p SWARM_PATH '/home/kamran.karimi1/Project/git/Swarm/dataset/Fillmore2015/SRL-2016020_esupp_Table_S2.txt'
jupyter nbconvert --execute /home/kamran.karimi1/Project/git/Swarm/runFillmore2015/Run3/output.ipynb --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.allow_errors=True
