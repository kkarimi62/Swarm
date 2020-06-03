#!/bin/bash

EXEC_DIR=/home/kamran.karimi1/Project/git/Swarm/runCaldera/Run9

papermill --prepare-only /home/kamran.karimi1/Project/git/Swarm/runCaldera/Run9/swarmCaldera.ipynb ./output.ipynb -p Df 2.0 -p quantile 0.05 -p SWARM_PATH '/home/kamran.karimi1/Project/git/Swarm/dataset/LongValleyCaldera/catalog1st.csv'
jupyter nbconvert --execute /home/kamran.karimi1/Project/git/Swarm/runCaldera/Run9/output.ipynb --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.allow_errors=True;rm ./output.html
