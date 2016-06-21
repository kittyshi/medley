
# coding: utf-8

# In[2]:

import os
import medley_util
import medley_helper
import pickle
import numpy as np

dataDir = 'test/'
songDir = '../dataset/testSongs/' + dataDir
pName = '../dataset/preStore/' + dataDir[:-1] + 'noTree.p'


# In[ ]:

#segs = medley_util.Segments(songDir,'test',doMash = False)
#pickle.dump(segs, open( pName, "wb" ) )


# In[4]:

segs = pickle.load( open( pName, "rb" ) )
segs.construct_tree()
segs.doMashup(useDiffSong = True)
def doMashup0621(the_segs,queries, outPath,N = 2, useDiffSong = True):
    for st_idx in queries:
        all_idx, timeLabel, finalMedley = medley_helper.createMedleyEasy(the_segs,st_idx, N)
        #medley_helper.printSegInfo(segs,all_idx,timeLabel)
        medley_helper.writeMedley(the_segs,outPath,finalMedley, all_idx, timeLabel)
od = '../output/0621/02defaultPair/'
#query_ids = [10,20,50, 60,80,365]
query_ids = np.arange(10,segs.treeSize,10)
query_ids.append(365)
doMashup0621(segs, query_ids, outPath = od)


# In[ ]:

segs2 = pickle.load( open( pName, "rb" ) )


# In[24]:

import numpy as np
import librosa
from IPython.display import Audio

def getNewTime(t):
    """
    Given a time in second, convert into time in min'sec" in string format
    """
    tmin = int(t/60)
    tsec = (t-int(t/60)*60)  # * 100
    return str(tmin) + "\'" + str(int(tsec))

def printSegInfo(segs,all_idx,timeLabel):
    """
    Print the medley transition info
    """
    for i in range(len(all_idx)):
        currSeg = segs.segments[all_idx[i]]
        newStart = getNewTime(timeLabel[i])
        oldSt = getNewTime(currSeg.st * 1. / currSeg.sr)
        oldEd = getNewTime(currSeg.ed * 1. / currSeg.sr)
        print newStart, currSeg.songname, oldSt, oldEd

            
def writeMedley(segs,outPath,sig,all_idx,timeLabel):
    if not os.path.exists(outPath):
        os.makedirs(outPath)
    outName = outPath + str(all_idx[0]) + '_' + str(all_idx[1])
    
    # write txt file
    text_file = open(outName+'.txt',"w")
    for i in range(len(all_idx)):
        curr_idx = all_idx[i]
        currSeg = segs.segments[curr_idx]
        newStart = getNewTime(timeLabel[i])
        oldSt = getNewTime(currSeg.st * 1. / currSeg.sr)
        oldEd = getNewTime(currSeg.ed * 1. / currSeg.sr)
        text_file.write(newStart + '\t' + currSeg.songname + '\t' + oldSt + '\t' + oldEd + '\n')
    text_file.close
    
    # write wave file
    librosa.output.write_wav(outName+'.wav', sig ,currSeg.sr)


def createMedley(segs,st_idx,N, outPath):
    """
    Given a segs object, a starting index
    Specify the length of the medley, and the output directory
    Create a medley
    """
    
    all_idx = [st_idx]
    timeLabel = [0.0]
    # get the value of the start segment
    currSeg = segs.segments[st_idx].y
    finalMedley = currSeg
    
    for i in range(N):
        next_idx = segs.find_best_match(st_idx, diffSong = True)
        all_idx.append(next_idx)
        currSeg = segs.segments[next_idx]                             # get next best match
        timeLabel.append(round(len(finalMedley) * 1. / currSeg.sr,2)) # when to start next segment
        finalMedley = np.append(finalMedley,currSeg.y)
        st_idx = next_idx
   
    return all_idx, timeLabel,finalMedley
    
        
    

