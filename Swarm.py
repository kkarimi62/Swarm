def makeOAR( EXEC_DIR, node, core, tpartitionime, PYFIL, argv):
	someFile = open( 'oarScript.sh', 'w' )
	print >> someFile, '#!/bin/bash\n'
	print >> someFile, 'EXEC_DIR=%s\n' %( EXEC_DIR )
	print >> someFile, 'papermill --prepare-only %s/%s ./output.ipynb %s'%(EXEC_DIR,PYFIL,argv) #--- write notebook with a list of passed params
	print >> someFile, 'jupyter nbconvert --execute %s/output.ipynb --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.allow_errors=True;ls output.html'%(EXEC_DIR)
	someFile.close()										  
#
if __name__ == '__main__':
	import os
#
	nruns	 = 1
	jobname  = 'swarmCalderaFinerMesh3rd' #'swarmYuhaDesertFirstPeriod1' #'yuhaDesertFit2nd' #'runRidgeCrest' #'runFillmore2015' 
	readPath = os.getcwd() # + '/CLUSTER' # --- source
	EXEC_DIR = '.'     #--- path for executable file
	durtn = '23:59:59'
	mem = '256gb' #'4gb' #'64gb'
	partition = 'bigmem' #'single' #'parallel' #'single'
#	argv = "-p BVALL 0.84 -p MCC 1.5"
	argv = "" #-p BVALL 1.30 -p MCC 3.0"
#	argv = "-p Df 2.0 -p quantile 0.05"
#	argv = "-p quantile 0.05"
#	argv = "-p n_thresh 0.001"
#	argv += " -p SWARM_PATH \'%s\'"%(readPath+'/dataset/Oklahoma/TableS1.csv') 
	argv += " -p SWARM_PATH \'%s\'"%(readPath+'/dataset/LongValleyCaldera/catalog1st.csv') 
#	argv += " -p SWARM_PATH \'%s\'"%(readPath+'/dataset/Mogul2008/hypodd.reloc.mag') 
#	argv += " -p SWARM_PATH \'%s\'"%(readPath+'/dataset/Fillmore2015/SRL-2016020_esupp_Table_S2.txt') 
#	argv += " -p SWARM_PATH \'%s\'"%(readPath+'/dataset/YuhaDesert/EMC.csv') 
#	argv += " -p SWARM_PATH \'%s\'"%(readPath+'/dataset/RidgeCrest/DataS1_noXYZ.txt') 
#	argv += " -p SWARM_PATH3 \'%s\'"%(readPath+'/dataset/RidgeCrest/catalog_fmsearch.csv') 
#	argv += " -p SWARM_PATH4 \'%s\'"%(readPath+'/dataset/RidgeCrest/momentTensors.pkl') 
#	argv += " -p SWARM_PATH2 \'%s\'"%(readPath+'/dataset/RidgeCrest/momentTensor.csv') 
	PYFILdic = { 
		0:'DifferentThresholds.ipynb',
		1:'Mogul2008.ipynb',
		2:'Fillmore2015.ipynb',
		3:'swarmOklahoma.ipynb',
		4:'swarmCaldera.ipynb',
		5:'swarmYuhaDesert.ipynb',
		6:'swarmCaldera2nd.ipynb',
		7:'RidgeCrest2019_recovered.ipynb',
		8:'swarmYuhaDesertFirstPeriod.ipynb'
		}
	keyno = 4
#---
#---
	PYFIL = PYFILdic[ keyno ] 
	#--- update argV
	#---
	os.system( 'rm -rf %s' % jobname ) # --- rm existing
	# --- loop for submitting multiple jobs
	counter = 0
	for irun in xrange( nruns ):
		print ' i = %s' % counter
		writPath = os.getcwd() + '/%s/Run%s' % ( jobname, counter ) # --- curr. dir
		os.system( 'mkdir -p %s' % ( writPath ) ) # --- create folder
#		argv = None #ARGS #eval( ARGS[ keyno ] )
		makeOAR( writPath, 1, 1, durtn, PYFIL, argv) # --- make oar script
		os.system( 'chmod +x oarScript.sh; mv oarScript.sh %s; cp %s/%s %s' % ( writPath, EXEC_DIR, PYFIL, writPath ) ) # --- create folder & mv oar scrip & cp executable
		os.system( 'sbatch --partition=%s --mem=%s --time=%s --job-name %s.%s --output %s.%s.out --error %s.%s.err \
						    --chdir %s -c %s -n %s %s/oarScript.sh'\
						   % ( partition, mem, durtn, jobname, counter, jobname, counter, jobname, counter \
						       , writPath, 1, 1, writPath ) ) # --- runs oarScript.sh!
		counter += 1
											 

