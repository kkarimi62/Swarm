#!/bin/bash

EXEC_DIR=/home/kamran.karimi1/Project/git/Swarm/runCaldera2nd/Run3

papermill --prepare-only /home/kamran.karimi1/Project/git/Swarm/runCaldera2nd/Run3/swarmCaldera2nd.ipynb ./output.ipynb -p quantile 0.05 -p SWARM_PATH '/home/kamran.karimi1/Project/git/Swarm/dataset/LongValleyCaldera/catalog1st.csv'
jupyter nbconvert --execute /home/kamran.karimi1/Project/git/Swarm/runCaldera2nd/Run3/output.ipynb --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.allow_errors=True;rm output.html
