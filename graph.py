
from scipy.ndimage.filters import gaussian_filter
from mytools import Lister
#from stream import readDump
import scipy.spatial as sp
import expand_periodic_box as pbc
import scipy.interpolate as scpy_int
from math import *
#import rmDuplicates as rmDUP
import numpy as np
import sys
import time
import sys
from math import *
import random as rnd
import scipy.interpolate as scpy_int
import numpy as np
#---
from math import *
import numpy as np
import random as rnd
from scipy.optimize import curve_fit
#---
import interpolation
#---
import h5py
import pdb
import scipy.interpolate as scpy_int

def adjacency_matrix( d_daughters, ids_sorted ):
#		ids_sorted = ( event_ids.sort_values() ).reset_index( drop = True ) #--- index:new id val=old id
		#--- include every pair
		n = len( ids_sorted )
		nsq = n * n
		cords = {}
		val = {}
		A = lil_matrix( ( n, n ) )
		for ikey in d_daughters: #--- adjacency matrix
			ms_id = ids_sorted[ ids_sorted == ikey ].index[ 0 ] #--- new id
			for afs_id in d_daughters[ikey]:
				afs_new_id = ids_sorted[ ids_sorted == afs_id ].index[ 0 ] #--- new id
				Index = ms_id * n + afs_new_id #--- scaler index
				assert Index < nsq
				A[ ms_id, afs_new_id ] = 1.0
				#---
				cords[ Index ] = [ afs_new_id, n - 1 - ms_id ]
				val[ Index ] = 1.0
		#--- plot
		comment = 'A_GLOBAL' # if index == 0 else 'A_LOCAL'
		st.makeDATA( open( 'adjacency.xyz', 'a' ), None ).Map( cords, comment, val ) #--- instantanuous
		#---
		return A
#---
def convert_to_distance( df ):
	xlo = df[ 'lon' ].min() - 1.0e-06 
	ylo = df[ 'lat' ].min() - 1.0e-06 
	xhi = df[ 'lon' ].max() + 1.0e-06  
	yhi = df[ 'lat' ].max() + 1.0e-06 
	returnDistX = lambda x, xlo: geopy.distance.vincenty( ( 0.0, xlo), ( 0.0, x ) ).km
	returnDistY = lambda y, ylo: geopy.distance.vincenty( ( ylo, 0.0 ), ( y, 0.0 ) ).km
	df[ 'x' ] = df[ 'lon' ].apply( returnDistX, xlo = xlo )
	df[ 'y' ] = df[ 'lat' ].apply( returnDistY, ylo = ylo )
def add_date( df ):
	df[ 'date' ] = pd.to_datetime( df[ [ 'year', 'month', 'day', 'hour', 'minute', 'second' ] ] )
#---
def DensityGaussianKernel( df_tmp, SIGMA ):
	df_tmp[['lon','lat','mag']].to_csv('xy.txt',sep='\t',header=['#lon','lat','M'],index=None)
#	convert_to_distance( df )
	add_date( df_tmp )
	#--- spatial bounds
#	df_dx = df[ 'x' ] - df[ df[ 'eventid' ] == id_epicenter ][ 'x' ].iloc[ 0 ]
#	df_dy = df[ 'y' ] - df[ df[ 'eventid' ] == id_epicenter ][ 'y' ].iloc[ 0 ]
	df_tmp[ 'dt' ] = df_tmp[ 'date' ] - df_tmp[ df_tmp[ 'eventid' ] == id_epicenter ][ 'date' ].iloc[ 0 ]
	#--- take a subset of data 
#	df_tmp = df[ ( df_dx * df_dx + df_dy * df_dy - radius * radius <= 0.0 ) & \
#				 ( df[ 'daughter' ] ) & \
#				 (  df[ 'date' ] >= df[ df[ 'eventid' ] == id_epicenter ][ 'date' ].iloc[ 0 ] ) ] #--- only triggered events inside the circle
	#--- time difference
	prefact = 1.0 / ( 24.0 * 60 * 60 )
	df_tmp[ 'dt' ] = df_tmp[ 'dt' ].dt.total_seconds() * prefact
	#--- print
	df_tmp[ [ 'dt','mag' ] ].to_csv('discrete_data.txt',sep='\t',header=['#DT', 'MAG'],index=None)	
	#--- histogram
	TMIN = df_tmp[ 'dt' ].min() - 1.0e-6 
	TMAX = TMIN + 365.0 #df_tmp[ 'dt' ].max() + 1.0e-6 
	dt = prefact * 60.0 
	nbins_x = int( ( TMAX - TMIN ) / dt )
	heatmap, xedges = np.histogram( df_tmp[ 'dt' ], bins = np.linspace(TMIN,TMAX,nbins_x+1), normed = True )
	heatmap *= len( df_tmp[ 'dt' ] ) #--- np.sum(heatmap)*dxy=n 
	#--- filter
	dt = ( TMAX - TMIN ) / nbins_x
	sigma = SIGMA / dt #--- kernel size == 2.0 days
	heatmap = gaussian_filter( heatmap, sigma = sigma )
	#--- print
	df_junk = pd.DataFrame( { 'density':heatmap,'xedges':xedges[0:len(xedges)-1]})
	df_junk[ [ 'xedges', 'density' ] ].to_csv( 'rate.txt', sep = '\t', index = None, header = [ '#T', 'N(T)' ] )
#---
def DensityRegularGrid( df, dx, dy ):
	convert_to_distance( df )
	#--- spatial bounds
	TMIN = df[ 'x' ].min() - 1.0e-6 
	TMAX = df[ 'x' ].max() + 1.0e-6 
	RMIN = df[ 'y' ].min() - 1.0e-6 
	RMAX = df[ 'y' ].max() + 1.0e-6 
	rmin = 0.999 * RMIN
	rmax = 1.001 * RMAX
	tmin = 0.999 * TMIN
	tmax = 1.001 * TMAX
	lx = TMAX - TMIN
	ly = RMAX - RMIN
	nbins_y = int( ly / dy )
	nbins_x = int( lx / dx )
	#--- 
	df_tmp = df[ df[ 'daughter' ] ] #--- only triggered events
	#--- plot
	giveList = lambda x: [ x ]
	coord_length = df_tmp[ 'x' ].apply( giveList ) + \
				   df_tmp[ 'y'].apply( giveList )
	DIR = '.'
	st.makeDATA( open( '%s/map.xyz'%DIR, 'a' ), None ).Map( coord_length.to_dict(), 'ITIME=%s'%0, df[ 'mag' ].to_dict() ) #--- instantanuous
	#---
	hist2d = hist.histogram2D( nbins_x, nbins_y, tmin, tmax, rmin, rmax )
	for rows in df_tmp.itertuples():
		t = rows.x
		r = rows.y
		try:
			hist2d.whichBin( [ t, r ], [ 1.0 ] )
		except:
			traceback.print_exc()
#	pdb.set_trace()
	sdict = hist2d.res( normalize = True, logScale = None, REGULAR = True )
	xy = [ [ k for k in sdict[ j ][ i ][ 0 : 2 ] ] for i in xrange( nbins_y ) for j in xrange( nbins_x ) ] #--- cords
	t_grid = np.linspace( tmin, tmax, 2 * nbins_x )
	r_grid = np.linspace( rmin, rmax, 2 * nbins_y )
	xyi = [ [ i, j ] for i in t_grid for j in r_grid ]
	npoin = sum( [ hist2d.kounter[ i ][ j ] for i in hist2d.kounter for j in hist2d.kounter[ i ] ] )
#	print 'npoin=%s'%npoin
	val = [ 0.0 for i in xrange( nbins_y ) for j in xrange( nbins_x ) ] #--- cords
	index = 0
	for irow in xrange( nbins_x ):
		for jcol in xrange( nbins_y ): 
			val[ index ] = sdict[ irow ][ jcol ][ 2 ] 
			index += 1
	valin = scpy_int.griddata( xy, val, xyi, method='cubic' ).tolist()
	#--- plot in ovito
	index = 0
	value = {}
	coord = {}
	for irow in xrange( nbins_x ):
		for jcol in xrange( nbins_y ):
			try:
				item = sdict[ irow ][ jcol ]
				tmean = item[ 0 ]
				rmean = item[ 1 ]
				value[ index ] = item[ 2 ]
				coord[ index ] = [ tmean, rmean ]
				index += 1
			except:
				traceback.print_exc()
				continue
	st.makeDATA( open( '%s/rho.xyz'%DIR, 'a' ), None ).Map( coord, 'INDEX\tX\tY\tVALUE', value ) #--- instantanuous
#---
def DensityDiscretePoints( df, dx, dy ):
	#---
	#--- voronoi tesselation analysis: only once
	#---
	#--- convert to distance
	convert_to_distance( df )
	#--- discretization
	xlo = df[ 'x' ].min() - 1.0e-6 
	xhi = df[ 'x' ].max() + 1.0e-6 
	ylo = df[ 'y' ].min() - 1.0e-6 
	yhi = df[ 'y' ].max() + 1.0e-6 
	lohi = [ [ xlo, xhi ], [ ylo, yhi ], 0.0 ] 
	print 'lohi=%s'%lohi
	lx = xhi - xlo
	ly = yhi - ylo
	m = int( ly / dy )
	n = int( lx / dx )
	dx = lx / n
	dy = ly / m
	print 'dx=%s,dy=%s'%(dx,dy)
	print 'm=%s,n=%s'%(m,n)
	xgrid, ygrid = np.mgrid[ xlo:xhi:n*1j, ylo:yhi:m*1j]
	t0 = time.time()
	xygrid = np.c_[ xgrid.ravel(), ygrid.ravel() ]
	print 'xygrid:%s'%(time.time() - t0)
	assert len( xygrid ) == m * n, 'len( xygrid ) = %s, mxn = %s'%( len( xygrid ), m*n )
	#--- convert to dictionary
	df_tmp = df[ df[ 'daughter' ] ] #--- only triggered events
	giveList = lambda x: [ x ]
	coord_length = df_tmp[ 'x' ].apply( giveList ) + \
				   df_tmp[ 'y'].apply( giveList )
	#--- plot
	st.makeDATA( open( '%s/map.xyz'%DIR, 'a' ), None ).Map( coord_length.to_dict(), 'ITIME=%s'%0, df[ 'mag' ].to_dict() ) #--- instantanuous
	#--- create object
	inobj = interpolation.interpolation( coord_length.to_dict(), lohi, xygrid, NDIME = 1, alpha=0.4)
	t0 = time.time()
	inobj.Voronoi()
	print 'voronoi:%s'%(time.time() - t0)
	t0 = time.time()
	inobj.ComputeArea()
	print 'area:%s'%(time.time() - t0)
	rho_field = {} 
	for keys in inobj.area:
		rho_field[ keys ] = 1.0 / inobj.area[ keys ] #--- assign density to scattered points
	t0 = time.time()
	valin = inobj.linear( rho_field, method = 'linear' ) #--- interpolate: valin has the same order as xygrid
	print 'interpolate:%s'%(time.time() - t0)
#---  interpolate
#	rhodn = [ [ valin[ i * n + j ][ 0 ] for j in xrange( n ) ] for i in xrange( m ) ] #--- matrix form
	#--- plot
	index = 0
	val = {}
	xy = {}
	t0 = time.time()
	for i in xrange( m ):
		for j in xrange( n ):
			val[ index ] = valin[ i * n + j ][ 0 ] #rhodn[i][j]
			xy[ index ] = xygrid[i*n+j][:]	
			index += 1
	print 'loop:%s'%(time.time() - t0)
	t0 = time.time()
	st.makeDATA( open( 'rho.xyz', 'a' ), None ).Map( xy, 'ITIME=%s'%0, val ) #--- instantanuous
	print 'plot:%s'%(time.time() - t0)
#---
class tree():
#	leafDepth = 0
#	n_leaves = 0
#	c = px.canvas.canvas()
	grace_outpt = None
	leaf_generation = {} #--- key: leaf id;  val: generation
	nsize = 1 #--- cluster size
	g_netx = None
#	INSERT = False 
	#---
	def __init__( self, node_id ): #, n_generation ):
		self.data = node_id
		self.n_daughters = 0
		self.daughter_obj = {}
		self.n_generation = 0
	def initialize( self ):
		tree.grace_outpt = None
		self.n_generation = 0
		for keys in self.daughter_obj: #--- search in daughters
			self.daughter_obj[ keys ].n_generation = 0
	#---
	def insert( self, parent_id, NODES ): #, n_generation ):
		if self.data == parent_id:
			for node_id in NODES:
				self.daughter_obj[ self.n_daughters ] = tree( node_id ) #, n_generation )
				self.n_daughters += 1
#			tree.INSERT = True #--- don't use return in this recursive function!
		else:
#			if self.n_daughters == 0: #--- it's a leaf!
#				return False
			for keys in self.daughter_obj:
				self.daughter_obj[ keys ].insert( parent_id, NODES ) #, n_generation )	
	#---
	def sprint( self, df ):
#		self.plotCircle()
		for keys in self.daughter_obj:
			tree.g_netx.add_node(self.daughter_obj[ keys ].data)
			tree.g_netx.add_edge( self.data, self.daughter_obj[ keys ].data)
			self.daughter_obj[ keys ].sprint( df )
	#---
	def getSize( self ):
		for keys in self.daughter_obj: #--- search in daughters
			tree.nsize += 1
#			print 'id=%s,size=%s'%(self.daughter_obj[ keys ].data, tree.nsize)
			self.daughter_obj[ keys ].getSize()
	#---
	def getLeaves( self ):
		if len( self.daughter_obj ) == 0: #--- it's a leaf
			tree.leaf_generation[ self.data ] = self.n_generation
#			tree.leafDepth += self.n_generation #--- depth
#			tree.n_leaves += 1 #--- number
#			print tree.leafDepth, self.data, self.n_generation
#		else:
		if 1:
			for keys in self.daughter_obj: #--- search in daughters
				self.daughter_obj[ keys ].n_generation = self.n_generation + 1
#				print 'id=%s'%self.daughter_obj[ keys ].data, 'n_generation=%s'%self.daughter_obj[ keys ].n_generation
				self.daughter_obj[ keys ].getLeaves()

	#---
	def Grace( self, df ):
		event_id = self.data
		print >> tree.grace_outpt, df[ df[ 'eventid' ] == event_id ][ 'TIME' ].iloc[0], df[ df[ 'eventid' ] == event_id ][ 'magnitude' ].iloc[0], df[ df[ 'eventid' ] == event_id ][ 'magnitude' ].iloc[0]
	#---
	def plotCircle( self, expand = 2 ):
		d=0.25 #--- diameter
		index = 1.0 #--- color index
#		[ x, y ] = [ i*expand for i in self.coord ] #--- coordinates
#		[ xj, yj ] = [ i*expand for i in self.coord_parent ]
#		if abs(x-xj)<1e-10 and abs(y-yj)<1e-10: #--- if root
#			index=0.0 #--- black
#		theta=atan2(yj-y,xj-x) #--- angle
#		line = px.path.path(px.path.moveto(x+0.5*d*cos(theta), y+0.5*d*sin(theta)), px.path.lineto(xj-0.5*d*cos(theta), yj-0.5*d*sin(theta))) #--- edge
#		tree.c.stroke( line, [ px.style.linewidth(0.01)])
#		tree.c.stroke(px.path.circle(x,y,d*0.5),[px.style.linewidth( 0.001 ), px.deco.filled([px.color.gray(index)])])  #--- draw circle
#		c.text(x, y, self.n_generation,[px.trafo.scale(d)]) #--- id
	def get( self, title ):
		pass
#		tree.c.writePDFfile( title )
def isParentDependent( irow, A ):
	cols, junk = A[ :, irow ].nonzero()
#	print len( cols )
	return len( cols ) == 1
#---
if __name__ == '__main__':
	import histogram as hist
	import stream as st
	import os
	from scipy import ndimage
	import traceback
	import geopy.distance
	import csv
	import datetime
	import random
	import histogram as hist
	import crltdField
	import plot as plt
	import scipy
	import scipy.interpolate
	from scipy.interpolate import Rbf
	import interpolation
	import scipy.interpolate as scpy_int
	import pandas as pd
	from scipy.sparse import csr_matrix, find
	from scipy.sparse import lil_matrix
	import scipy.sparse as sparse
	import networkx as nx
	import matplotlib.pyplot as plt

#---
	NTHRESH = float( sys.argv[ 1 ] ) 
	mc = float( sys.argv[ 2 ] ) 
	DIR = sys.argv[ 3 ]
        if not( DIR == '.' or DIR == './' ):
		if os.path.exists(DIR):
			os.system('rm -r %s'%DIR)
		os.system('mkdir %s'%DIR)
	csv_file = open( sys.argv[ 4 ], mode='r') 
	sfile = open( sys.argv[ 5 ])
	ROOT_ID = [16, 2070, 760, 3471, 1854, 4501, 4503, 2664, 2726, 2741, 731, 2795, 805, 860, 3921, 3151, 1211, 1179, 3282, 1255, 1265, 3492, 5568, 3611, 1642, 7956, 3927]
#int( sys.argv[ 6 ] )
#	radius = float( sys.argv[ 7 ] ) #--- events within radius [km] from the epicenter
#	SIGMA = float( sys.argv[ 7 ] ) #--- characteristic time [day] for the kernel
	#--- store in a data frame
	df = pd.read_csv( csv_file, sep = ',' )
	df_complete = df[ df[ 'magnitude' ] >= mc ]
#	pdb.set_trace()
#	df_complete[ 'second' ] =  df_complete[ 'second' ].apply( lambda x: int( x ) )
	df_complete[ 'TIME' ] = pd.to_datetime( df_complete[['year','month','day','hour','minute','second']] ) #, format = "%Y-%m-%d-%H-%M-%s" ) #--- add time object
#	df_complete[ 'TIME' ] = pd.to_datetime( df_complete[['year','month','day' ]], format = "%Y-%m-%d-" ) #--- add time object
	df_complete[ 'TIME' ] = df_complete[ 'TIME' ].dt.strftime("%Y-%m-%d-%H-%M-%S")
	df_complete[ 'eventid' ] = df_complete.index
	df_complete.reset_index( drop = True, inplace = True )
	#--- parse triggered events
	sfile.readline() #---header
	slist=[[float(i) for i in line.split()] for line in sfile]
	sfile.close()
	d_daughters_triggered = {}
	d = {}
	for items in slist:
		event_id =int(items[0])
		parent_id=int(items[1])
		R=items[2]
		T=items[3]
		d.setdefault( parent_id, [] ).append( 1 ) #--- store then reindex
		d.setdefault( event_id, [] ).append( 1 )
		if not R*T < NTHRESH:
			continue
		d_daughters_triggered.setdefault( parent_id, [] ).append( event_id ) #--- key: parent val: children				
	sfile.close()
	assert len( d ) == len( df_complete ) #--- assert both objects contain all the events
	#--- setup adjacency matrix
	A = adjacency_matrix( d_daughters_triggered, df_complete[ 'eventid' ] )
	#--- set up trees	
	rows, junk = A.nonzero() #--- parents' rows 
	cls_tree = {}
	ROWS = list( set( rows ) )
	ROWS.sort() #--- data must be sorted in time!
	for irow, index in zip( ROWS, xrange( sys.maxint ) ): #--- loop over parents
		parent_id = df_complete[ 'eventid' ].iloc[ irow ]
		nodes = []
		junk, cols = A[ irow ].nonzero()
		for jcol in set( cols ): #--- children
			node_id = df_complete[ 'eventid' ].iloc[ jcol ]
			nodes.append( node_id )
#		if parent_id == 6534: # and root_id == 6533:
#			pdb.set_trace()
#		tree.INSERT = False
		for root_id in cls_tree:
#			if parent_id == 6534 and root_id == 6533:
#				pdb.set_trace()
			cls_tree[ root_id ].insert( parent_id, nodes )
#			if tree.INSERT: #--- parent is a dependent event
#				break
		if not isParentDependent( irow, A ):
#			print 'parent_id=%s'%parent_id
#		if not tree.INSERT: #--- create a new cluster
			cls_tree[ parent_id ] = tree( parent_id )
			cls_tree[ parent_id ].insert( parent_id, nodes )
	#--- sort based on size
	slist = []
	for sid, index in zip( cls_tree, xrange( sys.maxint ) ): #[3499,3575,3794]:
		cls_tree[ sid ].initialize()
		tree.nsize = 1 #--- initialize
		cls_tree[ sid ].getSize() #--- call function
		slist.append( [ tree.nsize, sid ] )
	slist.sort( reverse = True )
	#--- output
	options = {
    'node_color': 'red',
    'node_size': 200,
    'width': 1,
    'arrowstyle': '-|>', #'fancy',
    'arrowsize': 12 ,
	'with_labels': False
	}
#	for items, index in zip( slist, xrange( 10 ) ): #sys.maxint ) ): #[3499,3575,3794]:
#		sid = items[ 1 ]
	for sid, index in zip( ROOT_ID, xrange( 10 ) ): #sys.maxint ) ): #[3499,3575,3794]:
		print 'sid=%s'%sid
		tree.g_netx = None
		tree.g_netx = nx.DiGraph(directed=True)
		tree.g_netx.add_node(cls_tree[ sid ].data)
		cls_tree[ sid ].sprint( df_complete ) #getSize() #--- call function
		p= nx.drawing.nx_pydot.to_pydot(tree.g_netx)

#		pos = hierarchy_pos(tree.g_netx,1) 
#		nx.draw_networkx( tree.g_netx, pos=pos,arrows=True, **options)
#		plt.draw()
#		plt.savefig('test%s.png'%sid)
		p.write_png('test%s.png'%sid)
#	pdb.set_trace()
	#--- ovito		
#	for i in xrange( 2 ):
#		st.makeDATA( open( '%s/map.xyz'%DIR, 'a' ), None ).Map( cords, 'ITIME=%s'%0, mag ) #--- instantanuous
