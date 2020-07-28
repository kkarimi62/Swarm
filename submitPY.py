if __name__ == '__main__':
	import sys
	import os
	import numpy as np
	#---
	lnums = [ 13, 21 ]
	string=open('Swarm.py').readlines() #--- python script
	#---
	nphi = 5
	PHI = np.logspace(-3.0,-1.0,nphi,endpoint=True)
#	PHI = np.linspace(2.3,2.9,nphi,endpoint=True)
#	PHI = np.linspace(0.05,0.45,nphi,endpoint=True)
	#---
#	nn = 4
#	NTHRESH = np.linspace(0.05,0.11,nn,endpoint=True)
	#---
	jobname = 'runRidgeCrest'

#	PHI = [[PHI[iphi],NTHRESH[inn]] for iphi in xrange( nphi ) for inn in xrange(nn)]
#	nphi = len(PHI)
	
	for iphi in xrange( nphi ):
	#---	
		inums = lnums[ 0 ] - 1
		string[ inums ] = "\tjobname  = '%s'\n" % ('%s%s'%(jobname,iphi)) #--- change job name
	#---	densities
		phi = PHI[ iphi ]
		inums = lnums[ 1 ] - 1
#		string[ inums ] = "\targv=\'-p\tquantile\t%s\'\n"%phi
		string[ inums ] = "\targv=\'-p\tn_thresh\t%s\'\n"%phi
#		string[ inums ] = "\targv=\'-p\tDf\t%s\'\n"%phi
#		string[ inums ] = "\targv=\'-p\tDf\t%s\t-p\tquantile\t%s\'\n"%(phi[0],phi[1])

		sfile=open('junk%s.py'%iphi,'w');sfile.writelines(string);sfile.close()
		os.system( 'python junk%s.py'%iphi )
		os.system( 'rm junk%s.py'%iphi )
