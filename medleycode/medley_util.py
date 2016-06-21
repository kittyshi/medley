import os
import librosa
import numpy as np
from sklearn.neighbors import BallTree
import collections


class Seg(object):
    """
    each seg is a phrase segment
    songname: name that the segment came from
    st: starting time in the song in samples
    ed: ending time in the song in samples
    y : the signal of the segment
    y_harm: harmonic part of the signal
    y_perc: percussive part of the signal
    mfcc_size: size of mfcc (default 13)
    beat_mfcc : beat mfcc feature
    beat_chroma: beat chroma feature
    """
    
    def __init__(self, songname, st, ed, y, y_harm, y_perc, sr, mfcc_size = 13):
        self.y = y
        self.sr = sr
        self.y_harm = y_harm
        self.y_perc = y_perc
        self.st = st
        self.ed = ed
        self.songname = songname
        self.mfcc_size = mfcc_size
        self.chroma_size = 12
        self.idx = -1           # order in the segments

        self.extract()

    def extract(self):
        mfcc = librosa.feature.mfcc(y=self.y, sr=self.sr, n_mfcc=self.mfcc_size)
        self.tempo, beat_frames = librosa.beat.beat_track(y=self.y_perc, sr = self.sr)

        # Stack and synchronize between beat events, use median value
        self.beat_mfcc = librosa.feature.sync(mfcc, beat_frames, aggregate = np.median)
        # Compute chroma features from the harmonic signal
        chromagram = librosa.feature.chroma_cqt(y=self.y_harm, sr=self.sr)
        # Energy
        self.rms = librosa.feature.rmse(self.y)
        # Aggregate chroma features between beat events
        # We'll use the median value of each feature between beat frames
        self.beat_chroma = librosa.feature.sync(chromagram, beat_frames, aggregate=np.median)

    def get_tail_mfcc(self, n=1):
        return self.beat_mfcc[:,-n]

    def get_head_mfcc(self, n=1):
        return self.beat_mfcc[:,n-1]

    def get_tail_chroma(self, n=1):
        return self.beat_chroma[:,-n]

    def get_head_chroma(self, n=1):
        return self.beat_chroma[:,n-1]

    def get_tail_rms(self, n = 5):
        return np.median(self.rms[:,-n])
        #return self.rms(:,-n)

    def get_head_rms(self, n = 5):
        return np.median(self.rms[:,n-1])
        #return self.rms(:,n-1)



class Segments(object):
    """
    The object segments contains all the songs in the database
    it is an array of different segments
    segments: a list of Seg objects
    """
    def __init__(self, songDir, setName, doMash = False, useDiffSong = True):
        """
        For every song in the directory, analyze the song
        """
        self.segments = []
        self.treeSize = 0
        self.mashPairs = collections.defaultdict()
        for (root, dirnames,files) in os.walk(songDir):
            for name in files:
                if name.endswith(".wav") :
                    print name
                    songpath = os.path.join(root,name)
                    self.analyze(name[:-4],songpath, setName)

        if doMash:
            # Construct the tree
            self.construct_tree()
            # Do medley
            self.doMashup(useDiffSong)

    def get_segment(self, i):
        return self.segments[i]
    
    def analyze(self, songname,songpath, setName):
        # Step one. Read in an audio file, extract all features for the song, 
        # extract phrase boundaries, and store features into small seg object
        y, sr = librosa.load(songpath)
        # Separate harmonics and percussives into two waveforms
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        # Beat track on the percussive signal
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr = sr)
        # Step two. Get phrase boundary
        # Extract Phrase info TO-DO: automatically
        phraseLabName = '../dataset/Annotations/myphraselab/%s/%s.txt' % (setName, songname)
        phraseBound = [float(l) for l in open(phraseLabName).read().splitlines()]

        for v in range(0,len(phraseBound)-1):
            # start time, and end time in second
            st = int(phraseBound[v] * 1. * sr)
            ed = int(phraseBound[v+1] * 1. * sr)
            currSeg = Seg(songname, st, ed, y[st:ed], y_harmonic[st:ed], y_percussive[st:ed], sr)
            self.segments.append(currSeg)
        self.mfcc_size = currSeg.mfcc_size            
        self.chroma_size = currSeg.chroma_size
        self.rms_size = 1
        
        
            
    def construct_tree(self):
        # construct a tree with the mfcc of the 1st frame of each seg, and beat chroma 
        self.feature_size = self.mfcc_size + self.chroma_size + self.rms_size
        X = np.zeros((len(self.segments), self.feature_size))
        for idx, seg in enumerate(self.segments):
            X[idx,0:self.mfcc_size] = seg.get_head_mfcc(n=1)
            X[idx,self.mfcc_size:-self.rms_size] = seg.get_head_chroma(n=1)
            X[idx,-self.rms_size:] = seg.get_head_rms(n=5)
            seg.idx = self.treeSize
            self.treeSize += 1
            #X = np.array(X).reshape((1,-1))


        def mydist(x, y):
            
            #xChroma = x[0:12]
            #yChroma = y[0:12]
            #xMfcc = x[12:38]
            #yMfcc = y[12:38]
            #xRms = x[38]
            #yRms = y[38]
            
            #dist = spatial.distance.cosine(xChroma,yChroma) + beta * np.sum((xMfcc-yMfcc)**2)/5000.  +                   theta * np.sum((xRms-yRms)**2)
            dist = np.sum((x-y)**2)
            #dist = alpha * np.sum((xChroma-yChroma)**2) + beta * np.sum((xMfcc-yMfcc)**2)/5000.  +  \
             #    theta * np.sum((xRms-yRms)**2)  #+ np.sum((xEd-ySt)**2)
            return dist

        self.tree = BallTree(X, leaf_size=2, metric = 'pyfunc',func = mydist)


    def find_best_match(self, query_seg_idx, diffSong = False):    
        f1 = self.get_segment(query_seg_idx).get_tail_mfcc()
        f2 = self.get_segment(query_seg_idx).get_tail_chroma()
        f3 = self.get_segment(query_seg_idx).get_tail_rms()
        query_f = np.hstack((f1,f2))
        query_f = np.hstack((query_f,f3))
        query_f = np.array(query_f).reshape((1,-1))
        dist, idx = self.tree.query(query_f, k=20)
        idx = idx[0]
        # remove query_seg_idx from the list
        idx = [p for p in idx if p != query_seg_idx]
        if diffSong:
            tmp = [p for p in idx if self.segments[p].songname != self.segments[query_seg_idx].songname]
            return tmp[0]
        return idx[0]

    def doMashup(self, useDiffSong):
        """
        Do pairs of music mashups depending on the starting point
        """
        for i in range(self.treeSize):
            self.mashPairs[i] = self.find_best_match(i,useDiffSong)



