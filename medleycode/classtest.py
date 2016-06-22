
# coding: utf-8

# In[4]:

import os
import medley_util
import medley_helper
import pickle
import numpy as np
from sklearn.neighbors import BallTree


dataDir = 'test/'
songDir = '../dataset/testSongs/' + dataDir
pName = '../dataset/preStore/' + dataDir[:-1] + 'noTree.p'


# In[5]:

segs = medley_util.Segments(songDir,'test',doMash = False)
pickle.dump(segs, open( pName, "wb" ) )


# In[6]:

segs = pickle.load( open( pName, "rb" ) )
segs.construct_tree()
segs.doMashup(useDiffSong = True)
def doMashup0621(the_segs,queries, outPath,N = 1, useDiffSong = True):
    for st_idx in queries:
        all_idx, timeLabel, finalMedley = medley_helper.createMedleyEasy(the_segs,st_idx, N)
        #medley_helper.printSegInfo(segs,all_idx,timeLabel)
        medley_helper.writeMedley(the_segs,outPath,finalMedley, all_idx, timeLabel)
od = '../output/0621/04myDist111test/'
#query_ids = [10,20,50, 60,80,365]
query_ids = np.arange(10,segs.treeSize,10)
np.insert(query_ids,len(query_ids),365)
doMashup0621(segs, query_ids, outPath = od)


# In[2]:

segs = pickle.load( open( pName, "rb" ) )
segs.construct_tree()
segs.doMashup(useDiffSong = True)


# In[ ]:




# In[21]:

import numpy as np

def evalMashup0621(the_segs,alpha, beta, theta):
    count = 0
    the_segs.construct_tree(alpha, beta, theta)
    the_segs.doMashup(useDiffSong = True)
    for i in range(the_segs.treeSize):
        if the_segs.mashPairs[i] == i + 1:
            count += 1
            print i
    return count
evalMashup0621(segs,0,1,0)


# In[ ]:




# In[3]:

s1 = segs.segments[0]
s2 = segs.segments[10]


# In[32]:

from scipy import spatial
alpha = 1
beta = 1
theta = 1
def mydist(x, y):
    xMFCC = x[0,0:13]
    yMFCC = y[0,0:13]
    xChroma = x[0,13:13+12]
    yChroma = y[0,13:13+12]
    xRms = x[0,-1:]
    yRms = y[0,-1:]

    dist1 = np.sum((xMFCC-yMFCC)**2) / 5000
#    dist2 = 1.0 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    dist2 = spatial.distance.cosine(xChroma,yChroma)     #unknown why this doesn't work
    dist3 = np.sum((xRms-yRms)**2)
    dist = alpha * dist1 + beta * dist2 + theta * dist3
    return [dist,dist1,dist2,dist3]


# In[33]:

d = mydist(s1.featureV,s2.featureV)


# In[ ]:



