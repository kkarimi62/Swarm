import os
import sys

ffile=sys.argv[1]
for i in xrange(8):
#    os.system('cp runMogul%s/Run0/%s.png ../../%s.%s.png'%(i,ffile,ffile,i))
#    os.system('cp runFillmore%s/Run0/%s.png ../../%s.%s.png'%(i,ffile,ffile,i))
    os.system('cp runFillmore/Run%s/%s.png ../../%s.%s.png'%(i,ffile,ffile,i))

