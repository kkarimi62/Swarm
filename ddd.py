import os
import sys

ffile=sys.argv[1]
nn=int(sys.argv[2])
for i in xrange(nn):
    os.system('cp runRidgeCrest%s/Run0/%s.png ../../%s.%s.png'%(i,ffile,ffile,i))
#    os.system('cp runMogul%s/Run0/%s.png ../../%s.%s.png'%(i,ffile,ffile,i))
#    os.system('cp runFillmore%s/Run0/%s.png ../../%s.%s.png'%(i,ffile,ffile,i))
#    os.system('cp runFillmore/Run%s/%s.png ../../%s.%s.png'%(i,ffile,ffile,i))
#    os.system('cp runOklahoma/Run%s/%s.png ../../%s.%s.png'%(i,ffile,ffile,i))

