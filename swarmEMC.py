#!/usr/bin/env python
# coding: utf-8

# In[38]:


#--- extract the catalog

import pandas as pd
import matplotlib; matplotlib.use('agg')
import sys
#--- add a new time attribute
def ConvertTime( df_in ):
    df=df_in.copy()
    df.insert(0,'date',pd.to_datetime(swarm[['year', 'month', 'day', 'hour', 'minute', 'second']]))
    df.drop(['year', 'month', 'day', 'hour', 'minute', 'second'],axis=1,inplace=True)
    return df


SWARM_PATH = './dataset/El_Mayor_Cucpah/EMC.csv' #sys.argv[1]
DIR_OUTPT = './dataset/El_Mayor_Cucpah' #sys.argv[2]

swarm = pd.read_csv( SWARM_PATH, sep = ' ' ) #--- parse data

swarm = ConvertTime( swarm ) #--- add new column 'date'
swarm.sort_values(by=['date'],inplace=True)
swarm.reset_index(inplace=True,drop=True)
swarm.head(16)


# In[39]:


#--- plot spatial map 
#--- fault map california: temporal evolution of events

import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D

def DataFrameSubSet( df, column, (xmin,xmax) ):
    return df[ ( df[column] >= xmin ) & 
               ( df[column] < xmax ) ]
#--- subset of data
swarm_lohi = DataFrameSubSet( swarm, 
                             'date', 
                             ( pd.to_datetime('2010-04-04'), pd.to_datetime('2010-06-26') ) )

swarm_lohi.plot.scatter('longitude','latitude',
                        s=3**(swarm['magnitude']),
                        c='depth',cmap='jet') #--- plot

plt.figure(figsize=(6,6)).gca(projection='3d')
plt.xlabel('Long')
plt.ylabel('Lat')
plt.scatter(swarm_lohi['longitude'],
            swarm_lohi['latitude'],
            swarm_lohi['depth']) #s=10*swarm['magnitude'],cmap='jet') #--- plot


# In[40]:


#--- plot timeseries

import matplotlib.pylab as plt
import numpy as np

plt.figure(figsize=(8,4))
plt.xlabel('Time')
plt.ylabel('M')
plt.ylim(-2,swarm['magnitude'].max())
plt.xlim(pd.to_datetime('2010-04-04'),pd.to_datetime('2010-06-26'))
plt.scatter(swarm['date'],swarm['magnitude'],
            s=2*np.exp(swarm['magnitude']),
            alpha=0.04)

plt.plot([pd.to_datetime('2010-04-08'),pd.to_datetime('2010-04-08')],
        [-2,7],'r-')

plt.plot([pd.to_datetime('2010-06-15'),pd.to_datetime('2010-06-15')],
        [-2,7],'r-')


# In[41]:


#--- freq-magnitude relation

import sys 

def histogramACCUMLTD( slist ):
    assert type( slist ) == type( [] ), 'arg must be a list. a %s is given!' %( type( slist ) )
    d = {}
    for item in slist:
        try:
            d[ item ] += 1
        except:
            d[ item ] = 1
    keys = d.keys()
    keys.sort()

    cdf = 0.0
    xi = min( slist ) - 1.0e-6
    xf = max( slist ) + 1.0e-6
    npoin = len( slist )
    adict = {}
    for ikey, index in zip( keys, xrange( sys.maxint ) ):
        adict[ index ] = [ xi, ikey, cdf ]
        cdf += 1.0 * d[ ikey ] # / npoin
        xi = ikey
    adict[ index + 1 ] = [ xi, xf, cdf ]
    return adict

#--- set min/max time to avoid temporal incompletenesss issue
swarm_copy = DataFrameSubSet( swarm, 
                             'date', 
                             ( pd.to_datetime('2010-04-08'), 
                               pd.to_datetime('2010-06-15') ) )

#--- accumulated histogram
N = len(swarm_copy['magnitude'])
slist=np.array(swarm_copy['magnitude'])
slist.sort()
d = histogramACCUMLTD( slist.tolist() )
keys=d.keys()
keys.sort()

plt.figure( figsize = (4,8))
plt.subplot(2,1,1)
plt.xlabel('m')
plt.ylabel('N(mag)>m')
plt.xlim(-3,6)
plt.ylim(1,1e9)
plt.yscale('log')
junk = plt.hist( swarm_copy['magnitude'],
                bins=128,
               label='histogram') #--- histogram
for ikey in keys:
    plt.plot([d[ikey][0],d[ikey][1]],
             [N-d[ikey][2],N-d[ikey][2]],
             '-',color='black') #--- accumulated

#--- estimate b-value
plt.subplot(2,1,2)
plt.xlim(-3,6)
plt.ylim(1,1e9)
plt.xlabel('m')
plt.ylabel('N*10^(bm)')
plt.yscale('log')
b=1.0
for ikey in keys: #--- rescaled
    c = 10**(b*d[ikey][0])
    plt.plot([d[ikey][0],d[ikey][1]],
             [c*(N-d[ikey][2]),c*(N-d[ikey][2])],'-',color='red') #--- accumulated


# In[42]:


#--- plot complete catalog

import sys

def ConvertDailyRate(hist, bin_edges ):
#---convert data to daily rate     
    t0 = pd.to_datetime( bin_edges[ 0 ] )
    t1 = pd.to_datetime( bin_edges[ 1 ] )
    delta_t = ( t1 - t0 ).total_seconds() / ( 60 * 60 * 24 )
    hist *= ( bin_edges[ 1 ] - bin_edges[ 0 ] ) / delta_t

def ActivityRate( swarm ):
    nbins = int( (swarm['date'].max()-swarm['date'].min()).days ) #--- number of bins
    
    tmax = swarm['date'].max().value #--- min/max
    tmin = swarm['date'].min().value
    
    hist, bin_edges = np.histogram(swarm['date'].apply(lambda x:x.value),                                   bins=np.linspace(tmin,tmax,nbins+1,endpoint=True),density=True) #--- probability dist.
    hist *= len( swarm['date'] ) #--- int(hist).dt=n
    cumm_number = np.cumsum(hist)*(bin_edges[1]-bin_edges[0]) #--- accumulated number
    ConvertDailyRate( hist, bin_edges ) #--- daily activity
    return bin_edges, hist, cumm_number

#--- completeness
mc = 0.5

#--- t0<t<t1
swarm_tmp = DataFrameSubSet( swarm, 
                             'date', 
                             ( pd.to_datetime('2010-04-04'), 
                               pd.to_datetime('2010-06-26') ) )
#--- m > mc
swarm_lohi = DataFrameSubSet( swarm_tmp, 
                             'magnitude', 
                             ( mc, sys.maxint ) ) 

#--- spatial map
swarm_lohi.plot.scatter('longitude','latitude',
                        s=3**(swarm_lohi['magnitude']),
                        c='date',cmap='jet',
                        alpha = 0.4) #--- plot
    
#--- temporal map
plt.figure(figsize=(8,4))
plt.xlabel('Time')
plt.ylabel('M')
plt.ylim(mc,swarm_lohi['magnitude'].max())
plt.xlim(pd.to_datetime('2010-04-04'),pd.to_datetime('2010-06-26'))
plt.scatter(swarm_lohi['date'],swarm_lohi['magnitude'],
            s=2*np.exp(swarm_lohi['magnitude']),
            alpha=0.04)

plt.plot([pd.to_datetime('2010-04-08'),pd.to_datetime('2010-04-08')],
        [-2,7],'r-')

plt.plot([pd.to_datetime('2010-06-15'),pd.to_datetime('2010-06-15')],
        [-2,7],'r-')

#--- activity rate
bin_edges, hist, cumm_number = ActivityRate( swarm_lohi )
#--- plot
plt.figure(figsize=(8,4))
plt.yscale('log')
plt.ylim(1e1,2e5)
plt.plot(pd.to_datetime(bin_edges[:-1]),hist,'.-',label='Daily rate')
plt.plot(pd.to_datetime(bin_edges[:-1]),cumm_number,'r-',label='acc. #')
plt.legend()


# In[70]:


import datetime 
swarm[(swarm['magnitude']>=mc) &
(swarm['date'] <= swarm['date'].min()+datetime.timedelta(days=5) )].info()


# In[43]:


#--- evaluate fractal dimension

import geopy.distance
from math import *
import random as rnd
import warnings

warnings.filterwarnings('ignore') #--- get rid of warnings

class bins:
    def __init__(self, nBin, xlo, xhi, ydim = 1, err = None):
        self.lo = xlo - 1.0e-10
        self.hi = xhi + 1.0e-10
        self.dx = (self.hi-self.lo)/nBin
        self.xlist = [0.0 for i in xrange(nBin)]
        self.kounter = [0 for i in xrange(nBin)]
        self.ylist = [[0.0 for j in xrange(ydim)] for i in xrange(nBin)]
        self.nBin = nBin
        self.err = err
        self.max_y = [[-sys.maxint for j in xrange(ydim)] for i in xrange(nBin) ]
        self.min_y = [[sys.maxint for j in xrange(ydim)] for i in xrange(nBin)]
        if err:
            self.ySQlist = [[0.0 for j in xrange(ydim)] for i in xrange(nBin)]
    def GetBin(self,x):
        return int(floor((x-self.lo)/self.dx))
    def whichBin(self,x,y, ibin=[] ):
        assert x >= self.lo, 'x=%s,self.lo=%s'%(10**x,10**self.lo)
        assert x < self.hi, 'x=%s,self.hi=%s'%(10**x,10**self.hi)
        nr = int(floor((x-self.lo)/self.dx))
        if ibin:
            ibin[ 0 ] = nr
        self.kounter[nr] += 1
        self.xlist[nr] += x
        for idim in xrange(len(y)):
            self.ylist[nr][idim] += y[idim]
            if y[idim] >= self.max_y[nr][idim]: #--- set max value
                self.max_y[nr][idim]=y[idim]
            if y[idim] <= self.min_y[nr][idim]:
                self.min_y[nr][idim]=y[idim]
            if self.err:
                self.ySQlist[nr][idim] += y[ idim ] * y[ idim ]

    def res(self, logScaleX = None, logScaleY = None, base = 10, SUM = None, MINMAX=None ):
         indices = xrange(10**6)
         someList = []
         for x,index in zip(self.xlist,indices):
             nb = self.kounter[index]
             if nb == 0: continue
             xbar = self.xlist[index]/nb
             ybar = [y/nb for y in self.ylist[index]]
             if self.err:
                 sigmaY = [ ( ysq / nb - YBAR * YBAR ) ** 0.5 / nb ** 0.5 for ysq, YBAR in zip( self.ySQlist[ index ], ybar )]
                 if SUM:
                     sigmaY = [ i * nb for i in sigmaY ]
             if SUM:
                 ybar = [y for y in self.ylist[index]]
             if MINMAX:
                 MAX_y = [y for y in self.max_y[index]]
             if logScaleX:
                 xbar = base ** xbar
             if logScaleY:
                 ybar = [ base ** item for item in ybar ]
             if self.err:
                 someList.append([ xbar, ybar, sigmaY ])
             elif MINMAX:
                 someList.append([ xbar, ybar, MAX_y ])
             else:
                 someList.append([ xbar, ybar ])
         return someList


class histogram( bins ):
    def res( self, Radial = None, logScale = None, normalize = True, base = 10.0, ACCUMLTD = None ):
        PDF = []
        self.nPoint = nPoint = sum( self.kounter )
        indices = xrange( sys.maxint )
        y_accm = nPoint
        for y, index in zip( self.kounter, indices ):
            if not ACCUMLTD and y == 0:
                continue
            if not y == 0:
                x = self.xlist[ index ] / y #self.lo + index * self.dx
            else:
                x = self.lo + index * self.dx
            Y = 1.0 * y
            dx = self.dx
            if logScale:
                x = base ** x
                dx = x * ( base ** self.dx - 1.0 )
#               print Y, dx
            if normalize:
                Y /= ( nPoint * dx )
                if Radial:
                    Y /= ( 2.0 * pi * x )
#           PDF.append( [ x, Y ] )
#           PDF.append( [ x + dx, Y ] )
#           PDF.append( [ x + 0.5 * dx, Y, 0.0, ( 1.0 * y_accm / nPoint if normalize else 1.0 * y_accm )  ] )
            error_std = 0.0
            if self.err:
                error_std = sqrt( nPoint * Y * dx ) / ( nPoint * dx ) #--- poisson
                error_std = sqrt( nPoint * Y * dx * ( 1.0 - Y * dx ) ) / ( nPoint * dx ) #--- bi-nomial
            PDF.append( [ x, Y, 0.0, ( 1.0 * y_accm / nPoint if normalize else 1.0 * y_accm ), error_std ] )
            y_accm -= y
        return PDF

def GetCartesian( dff ):
    df = dff.copy()
    xlo = df['longitude'].min()
    xhi = df['longitude'].max()
    ylo = df['latitude'].min()
    yhi = df['latitude'].max()
    getDistX = lambda x: geopy.distance.vincenty( ( 0.0, xlo ), ( 0.0, x ) ).km
    getDistY = lambda y: geopy.distance.vincenty( ( ylo, 0.0 ), ( y, 0.0 ) ).km
    df[ 'r(km)' ] = df[ 'longitude' ].apply( getDistX ) + df[ 'latitude' ].apply( getDistY ) * 1j
    return df

def fractalDimension2nd( coord ):
    #--- sort
    if type( coord ) == type( [] ):
        coord = ListToDict( coord )
    points = coord.keys()
    points.sort()
    hsObject = histogram( 18 * 8, log( 1e-10,10), log( 1e8, 10 ) )
    for point_i in points:
        for point_j in points:
            if not point_i < point_j: #--- pairs ij with i<j
                continue
            rij = sum( [ ( i - j ) ** 2 for i, j in zip( coord[ point_i ], coord[ point_j ] ) ] ) # ** 0.5
            assert rij > 0, 'rij=%s,coord[ %s ]=%s, coord[ %s ]=%s' %(rij,point_i,coord[ point_i ], point_j, coord[ point_j ] )
            hsObject.whichBin( 0.5 * log( rij,10 ), [ 1.0 ] )
    for items in hsObject.res( logScale = True, normalize = True, ACCUMLTD = True ):
        if items[ 3 ] > 0.0: 
            yield items[ 0 ], items[ 3 ]

#--------------------
#----- subset
#--------------------
swarm_tmp = DataFrameSubSet( swarm, 
                             'date', 
                             ( pd.to_datetime('2010-04-08'), 
                               pd.to_datetime('2010-06-15') ) )
swarm_lohi = DataFrameSubSet( swarm_tmp, #--- complete catalog
                             'magnitude', 
                             ( mc, sys.maxint ) ) 
swarm_lohi = swarm_lohi.sample( n = 1000 ) #--- sample

#--------------------
#--- cartesian coords
#--------------------
swarm_cartesian = GetCartesian( swarm_lohi )

#--------------------
#--- evaluate df
#--------------------
tmp_coord = swarm_cartesian['r(km)'].apply(lambda x: [x.real+rnd.random()*1e-6]) +             swarm_cartesian['r(km)'].apply(lambda x: [x.imag+rnd.random()*1e-6])
tmp_coord = tmp_coord.to_dict()
dict_NR = fractalDimension2nd( tmp_coord ) #, dmin = 1.0e-02 )


#--------------------
#--- scattered plot
#--------------------
swarm_cartesian.plot.scatter('longitude','latitude',
                        s=3**(swarm_lohi['magnitude']),
                        c='date',cmap='jet',
                        alpha = 0.4) 

#--------------------
#--- N(r) vs r
#--------------------
plt.figure( figsize = (4,4))
plt.xlabel('r(km)')
plt.ylabel('N(r)')
plt.xlim(1e-3,1e2)
plt.ylim(1e-5,1)
plt.yscale('log')
plt.xscale('log')
d_f = 1.6
for i in dict_NR:
    plt.plot([i[ 0 ]],
             [1-i[ 1 ]],
             'o',color='black') #--- accumulated
    plt.plot(i[ 0 ],
             (1-i[ 1 ])/i[0]**d_f,
             '.',color='red') #--- accumulated
    


# In[107]:


# --- trig analysis

from scipy.sparse import lil_matrix
import time
from IPython.display import display
import datetime

def getTmat( df_complete ):
    nmax = len( df_complete )
    prefact = 1.0 / ( 24.0 * 60 * 60 ) #--- daily
    for i in xrange( nmax ):
        df_dt = ( df_complete[ 'date' ] - df_complete[ 'date' ][ i ] ).dt.total_seconds() * prefact #--- time diff between i-th event and all subsequent events	
        df_dt[ : i ] = df_dt[ i ] #--- must have an upper triangular matrix 
        #---
        if i == 0:
            tmat = np.array( [ df_dt ] )
        else:
            tmat = np.append( tmat, [ df_dt ], axis = 0 )
    return tmat

def getTmat2nd( df_complete ):
    nmax = len( df_complete )
    prefact = 1.0 / ( 24.0 * 60 * 60 ) #--- daily
    tmat = np.matrix(np.zeros(nmax*nmax).reshape(nmax,nmax))
    for i in xrange( nmax ):
        df_dt = ( df_complete[ 'date' ] - df_complete[ 'date' ][ i ] ).dt.total_seconds() * prefact #--- time diff between i-th event and all subsequent events	
        df_dt[ : i ] = df_dt[ i ] #--- must have an upper triangular matrix 
        tmat[ i ] = np.array(df_dt)
         #---
    return np.array( tmat )


def getRmat( df_complete ):
    nmax = len( df_complete )
    for i in xrange( nmax ):
        #--- distance matrix
        df_dx = ( df_complete[ 'r(km)' ] - df_complete[ 'r(km)' ][ i ] )
        df_sq = abs( df_dx )
        df_sq[ : i ] = 0
        #---
        if i == 0:
            rsq_mat = np.array( [ df_sq ] )
        else:
            rsq_mat = np.append( rsq_mat, [ df_sq ], axis = 0 )
    return rsq_mat 

def getRmat2nd( df_complete ):
    nmax = len( df_complete )
    rmat = np.matrix(np.zeros(nmax*nmax).reshape(nmax,nmax))
    for i in xrange( nmax ):
        #--- distance matrix
        df_dx = ( df_complete[ 'r(km)' ] - df_complete[ 'r(km)' ][ i ] )
        df_sq = abs( df_dx )
        df_sq[ : i ] = 0
        rmat[ i ] = np.array(df_sq)
    return np.array( rmat ) 

def getMmat( df_complete ):
    nmax = len( df_complete )
    m_mat = np.matrix(np.zeros(nmax*nmax).reshape(nmax,nmax))
    for i in xrange( nmax ):
        #--- magnitude
        df_m = pd.Series( [ df_complete[ 'magnitude' ][ i ] ] * nmax )
        df_m[ : i ] = 0
        m_mat[ i ] = np.array(df_m)
    return np.array( m_mat )

def vectorizedAnalysis( df_complete ):
    #--- setup t, r, m
    t0 = time.time()
    tmat = getTmat2nd( df_complete )
    print 'setting up tmat:%s s'%(time.time() - t0)
    t0 = time.time()
    r_mat = getRmat2nd( df_complete )
    print 'setting up rmat:%s s'%(time.time() - t0)
    t0 = time.time()
    m_mat = getMmat( df_complete )
    print 'setting up m_mat:%s s'%(time.time() - t0)
        
    #--- nij
    NIJ = tmat * r_mat ** ( Df ) * 10 ** ( - bval * ( m_mat - mc ) )
    TIJ = tmat * 10 ** ( - 0.5 * bval * ( m_mat - mc ) ) #--- scaled time
    RIJ = r_mat ** ( Df ) * 10 ** ( - 0.5 * bval * ( m_mat - mc ) ) #--- scaled time
    nmax = len( df_complete )
    N_sparse = lil_matrix( ( nmax, nmax ) ) #--- sparse matrix
    T_sparse = lil_matrix( ( nmax, nmax ) )
    R_sparse = lil_matrix( ( nmax, nmax ) )
    for junk, j in zip( NIJ, xrange( sys.maxint ) ): 
        if j == 0:
            continue
        x = min( NIJ[ :, j ][ NIJ[ :, j ] != 0 ] ) #--- min nij
        assert x != 0.0
        rowi = np.where( NIJ[ :, j ] == x )[ 0 ][ 0 ] #--- find row
        N_sparse[ rowi, j ] = x #--- insert
        T_sparse[ rowi, j ] = TIJ[ rowi, j ]
        R_sparse[ rowi, j ] = RIJ[ rowi, j ]    
    return N_sparse, (T_sparse, tmat ), ( R_sparse, r_mat)

#--------------------
#----- subset
#--------------------
swarm_lohi = DataFrameSubSet( swarm, #--- complete catalog
                             'magnitude', 
                             ( mc, sys.maxint ) ) 

swarm_lohi = DataFrameSubSet( swarm_lohi, #--- temporal window
                             'date', 
                             ( pd.to_datetime('2010-04-08'), pd.to_datetime('2010-06-15') ) 
                            ) 
swarm_lohi.reset_index(inplace=True,drop=True)

#--------------------
#--- cartesian coords
#--------------------
swarm_lohi = GetCartesian( swarm_lohi )

#-------------------
#--- set parameters
#-------------------
Df = d_f
bval = b

#--- vectorized
t0 = time.time()
N_sparse, (T_sparse,tmat), (R_sparse,rmat) = vectorizedAnalysis( swarm_lohi )
print 'duration=%s s'%(time.time() - t0)


# In[109]:


#--- random analysis

def shuffleDIC( dataFrame, column = None ):
    junk =dataFrame[column].sample(frac=1).reset_index(drop=True)
    dataFrame[column] = junk

def shuffle(df, n=1, axis=0):     
    df = df.copy()
    for _ in range(n):
        df.apply(np.random.shuffle, axis=axis)
    return df
    
#--------------------
#--- randomize
#--------------------
swarm_shuffled = swarm.copy()
swarm_shuffled = GetCartesian( swarm_shuffled )
shuffleDIC( swarm_shuffled, column = 'magnitude' )
shuffleDIC( swarm_shuffled, column = 'r(km)' )
swarm_shuffled.to_csv('%s/swarm_shuffled_incomplete.csv'%DIR_OUTPT) #--- store

#--------------------
#----- subset
#--------------------
swarm_shuffled = DataFrameSubSet( swarm_shuffled, #--- complete catalog
                                  'magnitude', 
                             ( mc, sys.maxint ) ) 

swarm_shuffled = DataFrameSubSet( swarm_shuffled, #--- temporal window
                             'date', 
                             ( pd.to_datetime('2010-04-08'), pd.to_datetime('2010-06-15') ) 
swarm_shuffled.reset_index(inplace=True,drop=True)

#--------------------
#--- vectorized
#--------------------
t0 = time.time()
N_sparse_rnd, (T_sparse_rnd, junk), (R_sparse_rnd,junk) = vectorizedAnalysis(swarm_shuffled)
print 'analysis duration=%s s'%(time.time() - t0)


# In[118]:


#--- save sparse matrices

import scipy.sparse

for mat, title in zip([N_sparse,T_sparse,R_sparse],
                      ['N_sparse_matrix.npz','T_sparse_matrix.npz','R_sparse_matrix']):
    scipy.sparse.save_npz('%s/%s'%(DIR_OUTPT,title), mat.tocsr()) #--- output

for mat, title in zip([N_sparse_rnd,T_sparse_rnd,R_sparse_rnd],
                      ['N_sparse_rnd_matrix.npz','T_sparse_rnd_matrix.npz','R_sparse_rnd_matrix']):
    scipy.sparse.save_npz('%s/%s'%(DIR_OUTPT,title), mat.tocsr()) #--- output

swarm_shuffled.to_csv('%s/swarm_shuffled.csv'%DIR_OUTPT)


# In[44]:


#--- load sparse matrices
#--- 1- run on the cluster: 
#--- 2- sbatch --mem=8gb --partition=single  --time=02:59:59 -n 1 ./oarScript.sh
#--- 3- oarScript.sh: source activate conda-env; python ./swarmEMC.py
#--- 4- copy: scp arc:/home/kamran.karimi1/Project/seismic/dataset/El_Mayor_Cucpah/* ./swarm/dataset/El_Mayor_Cucpah/

import scipy.sparse

N_sparse = scipy.sparse.load_npz('%s/N_sparse_matrix.npz'%DIR_OUTPT )
T_sparse = scipy.sparse.load_npz('%s/T_sparse_matrix.npz'%DIR_OUTPT )
R_sparse = scipy.sparse.load_npz('%s/R_sparse_matrix.npz'%DIR_OUTPT )

N_sparse_rnd = scipy.sparse.load_npz('%s/N_sparse_rnd_matrix.npz'%DIR_OUTPT )
T_sparse_rnd = scipy.sparse.load_npz('%s/T_sparse_rnd_matrix.npz'%DIR_OUTPT )
R_sparse_rnd = scipy.sparse.load_npz('%s/R_sparse_rnd_matrix.npz'%DIR_OUTPT )

swarm_shuffled = pd.read_csv('%s/swarm_shuffled.csv'%DIR_OUTPT)
swarm_shuffled_incomplete = pd.read_csv('%s/swarm_shuffled_incomplete.csv'%DIR_OUTPT)


# In[45]:


#--------------------
#--- temporal map (actual)
#--------------------

plt.figure(figsize=(8,4))
plt.xlabel('Time')
plt.ylabel('M')
plt.title('Actual')
plt.ylim(-2,swarm['magnitude'].max())
plt.xlim(pd.to_datetime('2010-04-04'),pd.to_datetime('2010-06-26'))
plt.scatter(swarm['date'],swarm['magnitude'],
            s=2*np.exp(swarm['magnitude']),
            alpha=0.04)

plt.plot([pd.to_datetime('2010-04-08'),pd.to_datetime('2010-04-08')],
        [-2,7],'r-')

plt.plot([pd.to_datetime('2010-06-15'),pd.to_datetime('2010-06-15')],
        [-2,7],'r-')

#--------------------
#--- temporal map (shuffled)
#--------------------
def shuffleDIC( dataFrame, column = None ):
    junk =dataFrame[column].sample(frac=1).reset_index(drop=True)
    dataFrame[column] = junk

swarm_junk = swarm.copy()
swarm_junk = GetCartesian( swarm_junk )
shuffleDIC( swarm_junk, column = 'magnitude' )
shuffleDIC( swarm_junk, column = 'r(km)' )

plt.figure(figsize=(8,4))
plt.xlabel('Time')
plt.ylabel('M')
plt.title('shuffled')
plt.ylim(-2,swarm_junk['magnitude'].max())
plt.xlim(pd.to_datetime('2010-04-04'),pd.to_datetime('2010-06-26'))
plt.scatter(swarm_junk['date'],swarm_junk['magnitude'],
            s=2*np.exp(swarm_junk['magnitude']),
            alpha=0.04)
plt.plot([pd.to_datetime('2010-04-08'),pd.to_datetime('2010-04-08')],
        [-2,7],'r-')
plt.plot([pd.to_datetime('2010-06-15'),pd.to_datetime('2010-06-15')],
        [-2,7],'r-')


# In[ ]:


#--- matrix map of nij & nij_rand

plt.subplot(2,1,1)
plt.pcolormesh(np.log(N_sparse.toarray()),cmap='jet')
plt.subplot(2,1,2)
plt.pcolormesh(np.log(N_sparse_rnd.toarray()),cmap='jet')


# In[89]:


#--- scattered plot (actual) and interpolated density field

import matplotlib.pylab as plt
from math import *
from scipy.ndimage import gaussian_filter
import numpy as np
import datetime

def GetTrigListLoHi( nij_trig, col, (tlo,thi), swarm_lohi ):
    list_of_mothers = nij_trig.groupby(by='Parent_id').groups.keys() 
    list_of_mothers.sort()

    tmp_df = swarm_lohi.loc[ list_of_mothers ]
    tmp_df = tmp_df[ (tmp_df[ col ] >= tlo) & 
                 (tmp_df[ col ] < thi) ]

    return nij_trig[ pd.DataFrame([nij_trig['Parent_id'] == i for i in tmp_df.index]).any() ]

def GetInterpolatedData( df0 ):
    TMIN=-6
    TMAX=2
    RMIN=-5
    RMAX=3
    nbins_per_decade = 16
    nbins_x=nbins_y=int(TMAX-TMIN)*nbins_per_decade
    df=df0.copy()

    df['T*']=df[ 'T*' ].apply(lambda x: log(x,10))
    df['R*']=df[ 'R*' ].apply(lambda x: log(x,10))
    heatmap, xedges, yedges = np.histogram2d( df[ 'T*' ], df[ 'R*'], bins=[np.linspace(TMIN,TMAX,nbins_x+1), np.linspace(RMIN,RMAX,nbins_y+1)], normed=True)
    heatmap *= len( df )
    heatmap = gaussian_filter( heatmap, sigma = nbins_per_decade/4 )
    return heatmap

#--------------------
#--- store data in a dataframe
#--------------------
rows, cols = N_sparse.nonzero()
tlist = T_sparse[rows, cols].tolist()[0]
rlist = R_sparse[rows, cols].tolist()[0]
nij = pd.DataFrame({'Event_id':cols, 'Parent_id':rows,'T*':tlist,'R*':rlist})

#--------------------
#----- subset (the same as trig. analysis)
#--------------------
swarm_lohi = DataFrameSubSet( swarm, #--- complete catalog
                             'magnitude', 
                             ( mc, sys.maxint ) ) 
swarm_lohi = DataFrameSubSet( swarm_lohi, #--- temporal window
                             'date', 
                             ( swarm_lohi['date'].min(), swarm_lohi['date'].min()+datetime.timedelta(days=90))) 
swarm_lohi.reset_index(inplace=True,drop=True)

#--------------------------------------------
#----- only include mother events with 
#----- t_mother between (t_lo, t_hi) 
#--------------------------------------------
nij = GetTrigListLoHi( nij, 'date', 
#                       ( pd.to_datetime('2010-04-08'), pd.to_datetime('2010-06-15')), 
#                       ( pd.to_datetime('2010-04-08'), pd.to_datetime('2010-04-14')), 
                       ( pd.to_datetime('2010-04-14'), pd.to_datetime('2010-06-15')), 
                           swarm_lohi )
nij.reset_index(inplace=True, drop=True)

#--------------------
#--- plot scattered
#--------------------
n_thresh = 2e-5
plt.figure(figsize=(4,4))
plt.xscale('log')
plt.yscale('log')
plt.xlim(1e-6,1e2)
plt.ylim(1e-5,1e3)
plt.xlabel('T*')
plt.ylabel('R*')
plt.scatter(nij['T*'],nij['R*'],alpha=0.01)
plt.plot( nij['T*'], n_thresh/nij['T*'],'r')

#--------------------
#--- interpolated
#--------------------
plt.figure(figsize=(4,4))
heatmap=GetInterpolatedData(nij)
plt.pcolormesh(heatmap.T,cmap='jet')


# In[105]:


#--- scattered plot (fake) and interpolated density field
#--- evaluate the threshold

import matplotlib.pylab as plt
from math import *
from scipy.ndimage import gaussian_filter
import numpy as np

rows, cols = N_sparse_rnd.nonzero()
tlist = T_sparse_rnd[rows, cols].tolist()[0]
rlist = R_sparse_rnd[rows, cols].tolist()[0]
nij_rnd = pd.DataFrame({'Event_id':cols, 'Parent_id':rows,'T*':tlist,'R*':rlist})

#--------------------------------------------
#----- only include mother events with 
#----- t_mother between (t_lo, t_hi) 
#--------------------------------------------
nij_rnd = GetTrigListLoHi( nij_rnd, 'date', 
                         ( pd.to_datetime('2010-04-08'), pd.to_datetime('2010-06-15')), 
                           swarm_shuffled )
nij_rnd.reset_index(inplace=True, drop=True)

#--- plot scattered
plt.figure(figsize=(4,4))
plt.xscale('log')
plt.yscale('log')
plt.xlim(1e-6,1e2)
plt.ylim(1e-5,1e3)
plt.xlabel('T*')
plt.ylabel('R*')
#plt.title('1st phase')
plt.scatter(nij_rnd['T*'],nij_rnd['R*'],alpha=0.01)
plt.plot( nij_rnd['T*'], n_thresh/nij_rnd['T*'])

#--- interpolated
plt.figure(figsize=(4,4))
heatmap=GetInterpolatedData(nij_rnd)
plt.pcolormesh(heatmap.T,cmap='jet')


# In[90]:


#--- triggering part
nij_trig=nij[nij['T*']*nij['R*']<=n_thresh]


# In[91]:


#--- plot clusters

import matplotlib.dates as md

def Inside(t,(tmin,tmax)):
    if tmin<= t<tmax:
        return True
    return False    

def GetTrigList( nij_trig ):
    d_trig = {}
    for items in nij_trig.itertuples():
        triggerID = items.Parent_id
        d_trig.setdefault( triggerID, [] ).append( items.Event_id )
    return d_trig

def PlotArrows(df_complete,d_trig, (tmin,tmax), (xmin,xmax), (ymin,ymax) ):
    tlist=[]
    mlist=[]
    for triggerID in d_trig:
        t0 = df_complete['date'].iloc[ triggerID ]
        if not ( Inside(t0,(tmin,tmax) ) and 
                 Inside(df_complete['longitude'].iloc[triggerID],(xmin,xmax)) and 
                 Inside(df_complete['latitude'].iloc[triggerID],(ymin,ymax)) ):
            continue
        tlist.append( t0 )
        x0 = md.date2num(t0)
        y0 = df_complete['magnitude'].iloc[triggerID]
        mlist.append( y0 )
        for daughter_id in d_trig[triggerID]:
            t1 = df_complete['date'].iloc[ daughter_id ]
            tlist.append( t1 )
            mlist.append( df_complete['magnitude'].iloc[daughter_id] )
            if not ( Inside(t1,(tmin,tmax) ) and 
                     Inside(df_complete['longitude'].iloc[daughter_id],(xmin,xmax)) and 
                     Inside(df_complete['latitude'].iloc[daughter_id],(ymin,ymax)) ):
                continue
            xw = md.date2num(t1) - x0
            yw = df_complete['magnitude'].iloc[daughter_id] - y0
            plt.annotate("", (t1,df_complete['magnitude'].iloc[daughter_id]), xytext=(t0, y0),
                         textcoords='data',
                        arrowprops=dict(arrowstyle="-|>,head_width=.4,head_length=.8",color="b",linewidth="0.3")) 
    
    #--- plot circles
    df=pd.DataFrame({'date':tlist,'mag':mlist})
    plt.scatter(df['date'], df['mag'],
                s=4**(df['mag']),
                alpha=1,
                facecolors='red',color='black')
    #plt.savefig('timeSeries.png')
    

    
#--------------------------------------
#----- key: event value: aftershock id
#--------------------------------------
d_trig = GetTrigList( nij_trig )

#--------------------
#----- subset
#--------------------
swarm_lohi = DataFrameSubSet( swarm, #--- complete catalog
                             'magnitude', 
                             ( mc, sys.maxint ) ) 
swarm_lohi.reset_index(inplace=True,drop=True)

#--------------------
#--- cartesian coords
#--------------------
#swarm_lohi = GetCartesian( swarm_lohi )

#--- setup
fig = plt.figure(figsize=(16,4),frameon=False, dpi=300)

#--- xlimit
xmin = swarm_lohi['longitude'].min()
xmax = swarm_lohi['longitude'].max()
ymin = swarm_lohi['latitude'].min()
ymax = swarm_lohi['latitude'].max()
plt.tick_params(axis='x', labelsize=14)
plt.tick_params(axis='y', labelsize=14)

#--- plot all events
plt.ylim(mc,swarm['magnitude'].max())
plt.scatter(swarm['date'],swarm['magnitude'],
            s=2*np.exp(swarm['magnitude']),
            alpha=0.04)

#--- plot arrows
PlotArrows( swarm_lohi, d_trig,
                       ( pd.to_datetime('2010-04-14'), pd.to_datetime('2010-06-15')), 
#          (pd.to_datetime('2010-04-08'),pd.to_datetime('2010-04-14')),
#          (pd.to_datetime('2010-04-04'),pd.to_datetime('2010-06-26')),
           (xmin,xmax),
           (ymin,ymax) )
plt.xlim(pd.to_datetime('2010-04-04'),pd.to_datetime('2010-06-26'))


# In[92]:


#--- spatial clusters

import matplotlib.cm as cm
import matplotlib as mpl
import itertools


def PlotMap(df_complete,TRIG_ID, (tmin,tmax), (xmin,xmax), (ymin,ymax) ):
    #--- plot 
    counter = 0
    for triggerID,colin in zip(TRIG_ID,colors):
        t0 = df_complete['date'].iloc[ triggerID ]
        if not ( Inside(t0,(tmin,tmax) ) and 
                 Inside(df_complete['longitude'].iloc[triggerID],(xmin,xmax)) and 
                 Inside(df_complete['latitude'].iloc[triggerID],(ymin,ymax)) ):  
              continue
        counter += 1
        if counter > 3:
            break
        #--- plot    
        fig = plt.figure(figsize=(5,5))#,dpi=150)
        ax = fig.add_subplot(111)
        plt.xlim(xmin,xmax)
        plt.ylim(ymin,ymax)
        plt.tick_params(axis='x', labelsize=14)
        plt.tick_params(axis='y', labelsize=14)
        plt.scatter(df_complete.loc[triggerID]['longitude'],
                    df_complete.loc[triggerID]['latitude'],
                    s=4**df_complete.loc[triggerID]['magnitude'],
                    facecolors='white',color='white')
        d={'x':[],'y':[],'m':[],'t':[]}
        #--- plot daughters
        for daughter_id in d_trig[triggerID]:
            t1 = df_complete['date'].iloc[ daughter_id ]
            if not ( Inside(t1,(tmin,tmax) ) and 
                     Inside(df_complete['longitude'].iloc[daughter_id],(xmin,xmax)) and 
                     Inside(df_complete['latitude'].iloc[daughter_id],(ymin,ymax)) ):
                continue
            d['t'].append( t1 )
            d['x'].append(df_complete['longitude'].iloc[daughter_id])
            d['y'].append(df_complete['latitude'].iloc[daughter_id])
            d['m'].append(df_complete['magnitude'].iloc[daughter_id])
        df=pd.DataFrame(d)
        plt.scatter(df['x'],df['y'],
                    s=4**df['m'],
                    c=df['t'], cmap='jet') #,alpha=0.1)
        plt.colorbar()
        #--- plot mother
        plt.scatter(df_complete['longitude'].iloc[triggerID],df_complete['latitude'].iloc[triggerID],
                    s=4**df_complete['magnitude'].iloc[triggerID],
                    marker='*',
                    facecolors='yellow',color='black',
                    alpha=1.0)
        #--- 


    
colors = itertools.cycle(["r", "b", "g","b"])

#--- sort
new_list = [[len(d_trig[triggerID]),triggerID] for triggerID in d_trig]
new_list.sort(reverse=True)
TRIG_ID = [i[1] for i in new_list]

#--- xlimit
xmin = swarm_lohi['longitude'].min()
xmax = swarm_lohi['longitude'].max()
ymin = swarm_lohi['latitude'].min()
ymax = swarm_lohi['latitude'].max()
#plt.tick_params(axis='x', labelsize=14)
#plt.tick_params(axis='y', labelsize=14)

#--- plot arrows
PlotMap( swarm_lohi, TRIG_ID,
#          (pd.to_datetime('2010-04-04'),pd.to_datetime('2010-06-26')),
#          (pd.to_datetime('2010-04-08'),pd.to_datetime('2010-04-14')),
                       ( pd.to_datetime('2010-04-14'), pd.to_datetime('2010-06-15')), 
           (xmin,xmax),
           (ymin,ymax) )


# In[93]:


#--- density plots: rho(r)
#--- split based on the mainshock's magnitude

import numpy as np
import math

def GetPairsWithSpecifiedParentMag((m0,m1),catalog,df):
    df_parents = df.groupby(by='Parent_id').groups #--- parent events
    ds = catalog['magnitude'].loc[df_parents.keys()] #--- mag. of parent events
    ds_m=ds[(m0 <= ds) & (ds < m1)] #--- parent events with m0<m<m1
    df_m=df[pd.DataFrame([df['Parent_id'] ==k for k in ds_m.index]).any()] #--- data frame
    return df_m

def AddDist( df_trig, df_complete ):
    x=df_complete.loc[ df_trig['Event_id'] ]['r(km)'] 
    y=df_complete.loc[ df_trig['Parent_id'] ]['r(km)']
    df_trig['R'] = pd.Series(np.abs(np.array(x)-np.array(y)))
    assert len ( df_trig[ df_trig['R'] == 0.0 ] ) == 0, '%s'%display( df_trig[ df_trig['R'] == 0.0 ] )

def AddTime( df_trig, df_complete ):
    prefact = 1.0 / ( 24.0 * 60 * 60 ) #--- daily
    x=df_complete.loc[ df_trig['Event_id'] ]
    y=df_complete.loc[ df_trig['Parent_id'] ]
    x.reset_index(inplace=True)
    y.reset_index(inplace=True)
    df_trig['T'] = x['date']-y['date']
    df_trig['T'] = df_trig['T'].apply(lambda x: x.total_seconds() * prefact)

def RemovePair( df, cs ):
    return df [ df[ 'R' ] <= df[ 'T' ] * cs ]

#--------------------
#----- subset
#--------------------
swarm_lohi = DataFrameSubSet( swarm, #--- complete catalog
                             'magnitude', 
                             ( mc, sys.maxint ) ) 

swarm_lohi = DataFrameSubSet( swarm_lohi, #--- temporal window
                             'date', 
                             ( swarm_lohi['date'].min(), swarm_lohi['date'].min()+datetime.timedelta(days=90))) 
swarm_lohi.reset_index(inplace=True,drop=True)

#--------------------
#--- cartesian coords
#--------------------
df_complete = GetCartesian( swarm_lohi )

#--- choose m0<m<m1
n = 10
m = list(np.linspace(0.5,5.0,n))

X={}
rho_dict={}
m_range={}
Err={}
for i in xrange( len(m) - 1 ):
    m0 = m[ i ]
    m1 = m[ i + 1 ]
    df_trig = GetPairsWithSpecifiedParentMag((m0,m1),df_complete,nij_trig) #--- get parent with m0<m<m1
    df_trig.reset_index( inplace = True, drop = True )
    
    #--- add distance & time column
    AddDist( df_trig, df_complete )
    AddTime( df_trig, df_complete )
    
    #--- remove pairs with r > cs.t
    cs = 3 * (24*3600) #--- km per day
    df_trig = RemovePair( df_trig, cs )

    #--- rho plots
    n_decades = int( math.ceil( log(df_trig['R'].max()/df_trig['R'].min(),10) ) )
    nbin_per_decade = 4
    nbins =  nbin_per_decade * n_decades
    rho, xedges=np.histogram(df_trig['R'], 
                             density=True,
                             bins=np.logspace(log(df_trig['R'].min(),10),
                                              log(df_trig['R'].max(),10),nbins))
    hist, xedges=np.histogram(df_trig['R'],
                              bins=np.logspace(log(df_trig['R'].min(),10),
                                               log(df_trig['R'].max(),10),nbins))
    err = rho/np.sqrt(hist)
        
    #--- scattered plot
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.title('%s<m<%s'%(m0,m1))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('T')
    plt.ylabel('R')
    #plt.xlim(1e-3,100)
    #plt.ylim(1e-3,1e2)
    plt.scatter(df_trig['T'],df_trig['R'],alpha=0.1)
    
    #--- discretization effects
    dt_min=1e-6 #tmat[tmat != 0.0].min()
    dt_max=10 #tmat[tmat != 0.0].max()
    dr_min=1e-4 #rmat[rmat != 0.0].min()
    dr_max=1e3 #rmat[rmat != 0.0].max()
    
    plt.plot([dt_min,dt_min],[dr_min,dr_max],
             '-',color='blue')
    plt.plot([dt_max,dt_max],[dr_min,dr_max],
             '-',color='blue')
    plt.plot([dt_min,dt_max],[dr_min,dr_min],
             '-',color='blue')
    plt.plot([dt_min,dt_max],[dr_max,dr_max],
             '-',color='blue')
    
    plt.ylim(dr_min/2,dr_max*2)
    plt.plot([dt_min,dt_max],[dt_min*cs,dt_max*cs],
             '-',color='red')
    
    #--- rho
    plt.subplot(1,2,2)
    plt.xlim(1e-3,1e2)
    plt.ylim(1e-3,1e2)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('R')
    plt.ylabel('rho(R)')
    plt.errorbar(xedges[:-1],rho, yerr=err)
    
    exp = -1.35
#    plt.plot([min(xedges),max(xedges)],
#            [max(rho),max(rho)*(max(xedges)/min(xedges)) ** exp]
#            ,'-')

    x=xedges[:-1]
    plt.plot(np.array(x),np.array(rho)/x**exp)
    
    X[i] = x[:]
    rho_dict[i]=rho[:]
    m_range[i]=np.array([m0,m1])
    Err[i]=err[:]


# In[81]:


#--- scaling of the rupture size

plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('R')
plt.ylabel('rho(R)')
plt.xlim(1e-3,1e2)
plt.ylim(1e-3,1e2)
for key in X:
    plt.errorbar(X[key],rho_dict[key],yerr=Err[key], label='%s'%m_range[key])
plt.legend()  

plt.subplot(1,2,2)
alpha=0.4
plt.xscale('log')
plt.yscale('log')
plt.xlabel('R/10^a.m')
plt.ylabel('rho(R)*10^a.m')
plt.xlim(1e-5,1e0)
plt.ylim(1e-2,1e3)
for key in X:
    plt.errorbar(X[key]/10**(alpha*np.mean(m_range[key])),
                 rho_dict[key]*10**(alpha*np.mean(m_range[key])),
                 yerr=Err[key]*10**(alpha*np.mean(m_range[key])))


# In[88]:


#--- scaling of the rupture size

plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('R')
plt.ylabel('rho(R)')
plt.xlim(1e-3,1e2)
plt.ylim(1e-3,1e2)
for key in X:
    plt.errorbar(X[key],rho_dict[key],yerr=Err[key], label='%s'%m_range[key])
plt.legend()  

plt.subplot(1,2,2)
alpha=0.4
plt.xscale('log')
plt.yscale('log')
plt.xlabel('R/10^a.m')
plt.ylabel('rho(R)*10^a.m')
plt.xlim(1e-5,1e0)
plt.ylim(1e-2,1e3)
for key in X:
    plt.errorbar(X[key]/10**(alpha*np.mean(m_range[key])),
                 rho_dict[key]*10**(alpha*np.mean(m_range[key])),
                 yerr=Err[key]*10**(alpha*np.mean(m_range[key])))


# In[94]:


#--- scaling of the rupture size

plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('R')
plt.ylabel('rho(R)')
plt.xlim(1e-3,1e2)
plt.ylim(1e-3,1e2)
for key in X:
    plt.errorbar(X[key],rho_dict[key],yerr=Err[key], label='%s'%m_range[key])
plt.legend()  

plt.subplot(1,2,2)
alpha=0.4
plt.xscale('log')
plt.yscale('log')
plt.xlabel('R/10^a.m')
plt.ylabel('rho(R)*10^a.m')
plt.xlim(1e-5,1e0)
plt.ylim(1e-2,1e3)
for key in X:
    plt.errorbar(X[key]/10**(alpha*np.mean(m_range[key])),
                 rho_dict[key]*10**(alpha*np.mean(m_range[key])),
                 yerr=Err[key]*10**(alpha*np.mean(m_range[key])))


# In[41]:


#--- temporal plots split based on the mainshock's magnitude

import numpy as np

#--- make a copy
df_complete = swarm_cartesian.copy()
df_complete.reset_index(inplace=True,drop=True)

#--- choose m0<m<m1
n = 7
m = list(np.linspace(2.0,5.0,n))
m += [8.0]
for i in xrange( len(m) - 1 ):
    m0 = m[ i ]
    m1 = m[ i + 1 ]
    df_trig = GetPairsWithSpecifiedParentMag((m0,m1),df_complete,nij_trig) #--- get parent with m0<m<m1
    df_trig.reset_index( inplace = True, drop = True )
    
    #--- add distance & time column
    AddDist( df_trig, df_complete )
    AddTime( df_trig, df_complete )

    #--- remove pairs with r > cs.t
    cs = 3 * (24*3600) #--- km per day
    df_trig = RemovePair( df_trig, cs )

    #--- rho plots
    n_decades = int( math.ceil( log(df_trig['T'].max()/df_trig['T'].min(),10) ) )
    nbin_per_decade = 4
    nbins =  nbin_per_decade * n_decades
    rho, xedges=np.histogram(df_trig['T'], 
                             density=True,
                             bins=np.logspace(log(df_trig['T'].min(),10),
                                              log(df_trig['T'].max(),10),nbins))
    hist, xedges=np.histogram(df_trig['T'],
                              bins=np.logspace(log(df_trig['T'].min(),10),
                                               log(df_trig['T'].max(),10),nbins))
    err = rho/np.sqrt(hist)
        
    #--- scattered plot
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.title('%s<m<%s'%(m0,m1))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('T')
    plt.ylabel('R')
    #plt.xlim(1e-3,100)
    #plt.ylim(1e-3,1e2)
    plt.scatter(df_trig['T'],df_trig['R'],alpha=0.1)
    
    #--- discretization effects
    dt_min=tmat[tmat != 0.0].min()
    dt_max=tmat[tmat != 0.0].max()
    dr_min=rmat[rmat != 0.0].min()
    dr_max=rmat[rmat != 0.0].max()
    
    plt.plot([dt_min,dt_min],[dr_min,dr_max],
             '-',color='blue')
    plt.plot([dt_max,dt_max],[dr_min,dr_max],
             '-',color='blue')
    plt.plot([dt_min,dt_max],[dr_min,dr_min],
             '-',color='blue')
    plt.plot([dt_min,dt_max],[dr_max,dr_max],
             '-',color='blue')
    
    cs = 3 * (24*3600) #--- km per day
    plt.ylim(dr_min/2,dr_max*2)
    plt.plot([dt_min,dt_max],[dt_min*cs,dt_max*cs],
             '-',color='red')
    
    #--- rho
    plt.subplot(1,2,2)
    plt.xlim(1e-6,1e2)
    plt.ylim(1e-3,1e5)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('T')
    plt.ylabel('rho(T)')
#    plt.errorbar(xedges[:-1],rho, yerr=err)
    
    x=xedges[:-1]
    plt.plot(np.array(x),np.array(rho)*x**1.35)
    
#    exp = -2
#    plt.plot([min(xedges),max(xedges)],
#            [max(rho),max(rho)*(max(xedges)/min(xedges)) ** exp]
#            ,'-')


# In[86]:


#--- histograms: magnitude of mother events

#--------------------
#----- subset
#--------------------
swarm_lohi = DataFrameSubSet( swarm, #--- complete catalog
                             'magnitude', 
                             ( mc, sys.maxint ) ) 

swarm_lohi = DataFrameSubSet( swarm_lohi, #--- temporal window
                             'date', 
                             ( swarm_lohi['date'].min(), swarm_lohi['date'].min()+datetime.timedelta(days=90))) 
swarm_lohi.reset_index(inplace=True,drop=True)


mlist = swarm_lohi.loc[ nij_trig.groupby(by='Parent_id').groups.keys() ]['magnitude']
plt.hist(mlist)


# In[ ]:


#--- productivity

prod = nij_trig.groupby(by='Parent_id').count()['Event_id']
prod.index
#type(prod)
df_complete.loc[prod.index]['magnitude']

plt.figure(figsize=(4,4))
plt.xlim(2,6)
plt.ylim(1e0,1e4)
plt.yscale('log')
plt.xlabel('m')
plt.ylabel('# of aftershocks')
plt.scatter(df_complete.loc[prod.index]['magnitude'],prod,
           marker='x')


# In[ ]:


df_trig.head()


# In[ ]:


x


# In[ ]:


x^

