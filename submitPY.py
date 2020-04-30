if __name__ == '__main__':
	import sys
	import os
	import numpy as np
	#---
	lnums = [ 13, 19 ]
	string=open('Swarm.py').readlines() #--- python script
	#---
	nphi = 8
	PHI = np.logspace(-5.0,2.0,nphi,endpoint=True)
	#---
	jobname = 'multipleNth'



	for iphi in xrange( nphi ):
	#---	
		inums = lnums[ 0 ] - 1
		string[ inums ] = "\tjobname  = '%s'\n" % ('%s%s'%(jobname,iphi)) #--- change job name
	#---	densities
		phi = PHI[ iphi ]
		inums = lnums[ 1 ] - 1
		string[ inums ] = "\targv=\'-p\tbval\t%s\'\n"%phi

		sfile=open('junk%s.py'%iphi,'w');sfile.writelines(string);sfile.close()
		os.system( 'python junk%s.py'%iphi )
		os.system( 'rm junk%s.py'%iphi )
