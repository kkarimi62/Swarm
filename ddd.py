import os
import sys

ffile=sys.argv[1]
for i in xrange(5):
    os.system('cp runMogul%s/Run0/%s ../../%s.%s.png'%(i,ffile,ffile,i))

