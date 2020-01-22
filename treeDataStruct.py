
#import pyx as px
from math import *

class tree():
	leafDepth = 0
	n_leaves = 0
#	c = px.canvas.canvas()
	grace_outpt = None
    fig=None
	#---
	def __init__( self, data, coord, coord_parent, n_generation ):
		self.data = data #--- id
		self.coord = coord
		self.coord_parent = coord_parent
		self.n_daughters = 0
		self.daughter_obj = {}
		self.n_generation = n_generation
	#---
	def insert( self, data, parent_id, coord, coord_parent, n_generation ):
		if parent_id == self.data: #--- parent exists
			self.daughter_obj[ self.n_daughters ] = tree( data, coord, coord_parent, n_generation )
			self.n_daughters += 1
			return
		else:
			for keys in self.daughter_obj:
				self.daughter_obj[ keys ].insert( data, parent_id, coord, coord_parent, n_generation )	
	#---
	def sprint( self ):
#		self.Grace()
#		self.plotCircle()
		for keys in self.daughter_obj:
			self.daughter_obj[ keys ].sprint()
			self.daughter_obj[ keys ].Grace()
            self.daughter_obj[ keys ].plotCircle()
	#---
	def getLeaves( self ):
		if len( self.daughter_obj ) == 0: #--- it's a leaf
			tree.leafDepth += self.n_generation #--- depth
			tree.n_leaves += 1 #--- number
#			print tree.leafDepth, self.data, self.n_generation
		else:
			for keys in self.daughter_obj: #--- search in daughters
				self.daughter_obj[ keys ].getLeaves()
	#---
	def Grace( self, expand = 1 ):
		r = [ i for i in self.coord ] #--- coordinates
#		rj = [ i for i in self.coord_parent ]
		
#		print >> tree.grace_outpt, 'PARENT\tX\tY\tZ\tM\tDT'
#		print >> tree.grace_outpt,self.data,
#		for R in rj: #--- parent event
#			print >> tree.grace_outpt, R,
#		print >> tree.grace_outpt, '\nCHILD'
#		print >> tree.grace_outpt,self.data,
#		print hello!
		print >> tree.grace_outpt, self.n_generation,
		for R in r: #--- child
			print >> tree.grace_outpt, R,
		print >> tree.grace_outpt
	#---
	def plotCircle( self, expand = 2 ):
		d=0.25 #--- diameter
		index = 1.0 #--- color index
		[ x, y ] = [ i*expand for i in self.coord ] #--- coordinates
		[ xj, yj ] = [ i*expand for i in self.coord_parent ]
        print self.coord_parent
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

if __name__ == '__main__':
	import random
	import sys
	#---
	d_trig = {}
	d_trig['j']=['h','i']
	d_trig['h']=['d','e']
	d_trig['d']=['a','b','c']
	d_trig['e']=['f','g' ]
	NoTriggerList = ['j']
	Magnitude = {}
	TIME = {}
	for i, index in zip( ['j','h','i','d','e','a','b','c','f','g'], xrange( sys.maxint ) ):
		Magnitude[ i ] = random.random()
		TIME[ i ] = index
	#---
	CLUSTER={}
	clusterID=0
	CLUSTER={}
	clusterID = 0
	stree = {}
	for rootID in NoTriggerList: #--- roots: events with no parents
		tree_level = 0
		mainShockList = [ rootID ]
		CLUSTER[clusterID]=[mainShockList[0]]
		stree[ clusterID ] = tree( rootID, [ 0, Magnitude[ rootID ] ], [ 0, Magnitude[ rootID ] ], tree_level ) #--- create object
		while len(mainShockList): #--- from root down to leaves
			newMainShockList = []
			for mainShockID, counter in zip( mainShockList, xrange( sys.maxint ) ):
				try:
					newMainShockList += d_trig[ mainShockID ] #--- aftershocks become new
				except:
#					traceback.print_exc()
					continue
				for aftershockID in d_trig[ mainShockID ]:
					CLUSTER[ clusterID ].append(aftershockID)
					dt = TIME[ aftershockID ]-TIME[ rootID ]
					dt_parent = TIME[ mainShockID ]-TIME[ rootID ]
					stree[ clusterID ].insert( aftershockID, mainShockID, [ dt, Magnitude[ aftershockID ] ], [ dt_parent, Magnitude[ mainShockID ] ], tree_level + 1 ) #--- insert node
			mainShockList = [ i for i in newMainShockList ]
			tree_level += 1
		assert len(mainShockList) == 0 #--- leaf
		#--- plot tree
		tree.grace_outpt = open( 'grace.txt', 'w' )
		print >> tree.grace_outpt, '#M\tTIME'
		stree[ clusterID ].sprint()
		stree[ clusterID ].get( '%s/graph%s.pdf'% ('.',clusterID ) )
		stree[ clusterID ].grace_outpt.close()
		#--- mean depth
		stree[ clusterID ].getLeaves()
		mean_depth = 1.0 * stree[ clusterID ].leafDepth / stree[ clusterID ].n_leaves
#		print stree[ clusterID ].n_leaves, mean_depth
		#---
		clusterID+=1
