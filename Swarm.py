def makeOAR( EXEC_DIR, node, core, tpartitionime, PYFIL, argv):
	someFile = open( 'oarScript.sh', 'w' )
	print >> someFile, '#!/bin/bash\n'
	print >> someFile, 'EXEC_DIR=%s\n' %( EXEC_DIR )
	print >> someFile, 'papermill --prepare-only %s/%s ./output.ipynb %s'%(EXEC_DIR,PYFIL,argv) #--- write notebook with a list of passed params
	print >> someFile, 'jupyter nbconvert --execute %s/output.ipynb --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.allow_errors=True'%(EXEC_DIR)
	someFile.close()										  
#
if __name__ == '__main__':
	import os
#
	nruns	 = 8
	jobname  = 'multipleNth' 
	readPath = os.getcwd() # + '/CLUSTER' # --- source
	EXEC_DIR = '.'     #--- path for executable file
	durtn = '23:59:59'
	mem = '8gb'
	partition = 'single'
	argv = "-p n_thresh 1.0"
	argv += " -p SWARM_PATH \'%s\'"%(readPath+'/dataset/Oklahoma/TableS1.csv') 
	PYFILdic = { 
		0:'DifferentThresholds.ipynb',
		}
	keyno = 0
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
											 
