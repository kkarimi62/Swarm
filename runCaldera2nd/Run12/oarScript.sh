#!/bin/bash

EXEC_DIR=/home/kamran.karimi1/Project/git/Swarm/runCaldera2nd/Run12

papermill --prepare-only /home/kamran.karimi1/Project/git/Swarm/runCaldera2nd/Run12/swarmCaldera2nd.ipynb ./output.ipynb -p quantile 0.05 -p SWARM_PATH '/home/kamran.karimi1/Project/git/Swarm/dataset/LongValleyCaldera/catalog1st.csv'
jupyter nbconvert --execute /home/kamran.karimi1/Project/git/Swarm/runCaldera2nd/Run12/output.ipynb --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.allow_errors=True;rm output.html
