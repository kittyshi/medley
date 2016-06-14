
# coding: utf-8

# In[ ]:

import os
import numpy as np
import collections
import librosa
import pickle


# ## Parse songs in the directory

# In[ ]:

## parse Chroma and Signal for a small dataset (two albums)

#def predictChords(chroma):
    # predict chord in 24 types
    

def parseSongs(myDir,outFile = "tmp", store = False):
    allSong = collections.defaultdict()
    songLookup = collections.defaultdict()
    segLookup = collections.defaultdict()
    allY = []
    allC = []
    allMFCC = []
    allRMS = []
    sr = 22050
    currId = 0
    for (root, dirnames,files) in os.walk(myDir):
        for name in files:
            if name.endswith(".wav") and "silence" not in name:
                print name
                # read in and store the wave file
                songpath = os.path.join(root,name)
                y,sr = librosa.load(songpath)
                allY.append(y)
                # get and store the beat-synchronous chromagram/MFCC
                chroma = librosa.feature.chromagram(y, sr)
                tempo, beats    = librosa.beat.beat_track(y, sr)
                S               = librosa.feature.melspectrogram(y, sr, hop_length=64)
                mfcc            = librosa.feature.mfcc(S=S)
                mfcc_sync       = librosa.feature.sync(mfcc, beats)
                chroma_sync     = librosa.feature.sync(chroma, beats)
                
                #chords = predictChords(chroma_sync)
                #print chords
                
                allC.append( np.average(chroma_sync, axis = 1))
                allMFCC.append( np.average(mfcc_sync, axis = 1))
                # get and store the RMS
                rms = librosa.feature.rmse(y)
                allRMS.append(np.average(rms))
                # store the song/seg look up table
                songLookup[currId] = name   # songbook maps id to song name
                segLookup[name] = currId    # segbook maps name to segment number
                currId += 1
    
    # store in pickle
    if store:
        myData = collections.defaultdict()
        myData['dir'] = myDir
        myData['sr'] = sr
        myData['Chroma'] = allC
        myData['mfcc'] = allMFCC
        myData['Ysig'] = allY
        myData['rms'] = allRMS
        myData['seg'] = segLookup
        myData['songBook'] = songLookup
        pickle.dump(myData, open( outFile, "wb" ) )
        print "finish writing %s" % outFile
    
    return sr, allY, allC,allMFCC,allRMS,songLookup,segLookup


# ## Map chord sequence string into number (chordId 1-24)

# In[1]:

def mapCapital(capital):
    capital = capital[0]
    if capital == 'A':
        return 10;
    if capital == 'B':
        return 12;
    if capital == 'C':
        return 1;
    if capital == 'D':
        return 3;
    if capital == 'E':
        return 5;
    if capital == 'F':
        return 6;
    if capital == 'G':
        return 8;
    else:
        assert("incorrect capital")

def char2No(letter):
    baseNo = mapCapital(letter)
    chordNo = baseNo
    if len(letter) == 1:
        return baseNo
    else:
        # sharp or flat
        if letter[1] == '#':
            chordNo = baseNo + 1
        elif letter[1] == 'b':
            chordNo = baseNo - 1
        # for Cb equals B
        if chordNo == 0:
            chordNo = 12 
        assert(chordNo <= 12 and chordNo > 0)
        # major or minor
        if len(letter) > 4:
            if letter[3] == 'i' or letter[4] == 'i':
                chordNo += 12
        return chordNo


# ## Map chordId 1-24 to chordSequence

# In[22]:

def id2Chord(no):
    assert no > 0 and no < 25
    base = no % 12
    if base == 1:
        chord = 'C'
    elif base == 2:
        chord = 'C#'
    elif base == 3:
        chord = 'D'
    elif base == 4:
        chord = 'Eb'
    elif base == 5:
        chord = 'E'
    elif base == 6:
        chord = 'F'
    elif base == 7:
        chord = 'F#'
    elif base == 8:
        chord = 'G'
    elif base == 9:
        chord = 'Ab'
    elif base == 10:
        chord = 'A'
    elif base == 11:
        chord = 'Bb'
    elif base == 0:
        chord = 'B'
    
    if no > 12:
        chord = chord + " min"
    return chord


# ## Extract smooothed chord sequence from one song

# In[ ]:

def getChordSeq(filename,smooth = False):
    # read in lines without \n
    lines = open(filename).read().splitlines()
    chordSeq = []
    for line in lines:
        chord = line.split(' ')[2]
        if chord != 'N':
            chordSeq.append(char2No(chord))
    if smooth:
    # smooth the chord sequence to avoid repetition
        chordSmooth = [chordSeq[0]]
        i = 1
        while i < len(chordSeq):
            if chordSeq[i] != chordSeq[i-1]:
                chordSmooth.append(chordSeq[i])
            i = i+1
        chordSeq = chordSmooth 
    
    return chordSeq


# In[ ]:

def getChordSeqT(filename,smooth = False, tab = True):
    # read in lines without \n
    lines = open(filename).read().splitlines()
    chordSeq = []
    startTime = []
    endTime = []
    for line in lines:
        if not tab:
            chord = line.split(' ')[2]
        else:
            chord = line.split('\t')[2]
        if chord != 'N':
            chordSeq.append(char2No(chord))
            if not tab:
                startTime.append(line.split(' ')[0])
                endTime.append(line.split(' ')[1])
            else:
                startTime.append(line.split('\t')[0])
                endTime.append(line.split('\t')[1])
    if smooth:
    # smooth the chord sequence to avoid repetition
        chordSmooth = [chordSeq[0]]
        i = 1
        while i < len(chordSeq):
            if chordSeq[i] != chordSeq[i-1]:
                chordSmooth.append(chordSeq[i])
            i = i+1
        chordSeq = chordSmooth 
    
    return chordSeq,startTime,endTime


# ## Packing four 5-bit numbers into one 20-bit number
# 

# In[12]:

def pack4(x):
    
    def mask(z):
        return int(z) & 0b11111
    
    return (mask(x[0]) << 15) | (mask(x[1]) << 10) | (mask(x[2]) << 5) | mask(x[3])

def pack2(x):
    
    def mask(z):
        return int(z) & 0b11111
    return (mask(x[0]) << 5) | mask(x[1])

# pack a pair into a key
def packKey(x1,x2):
    return(pack2(x1),pack2(x2))


# In[ ]:

# convert a key to chord Id
def unpackKey(num):
    f1 = (num >> 5) & 0b11111
    f2 = num & 0b1111
    return f1,f2

# convert a key to chord name
def analyzeKey(num):
    f1 = (num >> 5) & 0b11111
    f2 = num & 0b11111
    return id2Chord(f1),id2Chord(f2)

