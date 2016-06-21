import os
import numpy as np
import librosa
import collections


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
        text_file.write(newStart + '\t' + str(all_idx[i]) + '\t' + currSeg.songname + '\t' + oldSt + '\t' + oldEd + '\n')
    text_file.close
    
    # write wave file
    librosa.output.write_wav(outName+'.wav', sig ,currSeg.sr)

def createMedleyEasy(segs,st_idx, N):
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
        next_idx = segs.mashPairs[st_idx]
        all_idx.append(next_idx)
        currSeg = segs.segments[next_idx]                             # get next best match
        timeLabel.append(round(len(finalMedley) * 1. / currSeg.sr,2)) # when to start next segment
        finalMedley = np.append(finalMedley,currSeg.y)
        st_idx = next_idx
   
    return all_idx, timeLabel,finalMedley


def createMedley(segs,st_idx,N):
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
    
        
    
