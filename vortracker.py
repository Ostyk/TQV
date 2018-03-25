import numpy as np

def dist(l1,l2):
    '''self explanatory. distance between two blobs'''
    return (l1[0]-l2[0])**2+(l1[1]-l2[1])**2

def rename(blobs,past,i,bonus):
    '''sets up name for vortices according to rule, that closest one is 
    the one contiuous in time'''
    D=np.ndarray((len(blobs),len(past)))
    for w in range(len(blobs)):
        for ww in range(len(past)):
            D[w][ww]=dist(blobs[w],past[ww])    
    if len(past)>=len(blobs):
        #print(D)
        for w in range(len(blobs)):
            mini=np.argmin(D[w])
            blobs[w][2]=past[mini][2]
    else:
        #print(D.T)
        N=np.arange(len(blobs))+np.ones(len(blobs))*i*100
        N.tolist()
        for w in range(len(past)):
                mini=np.argmin(D.T[w])
                N[mini]=past[w][2]
        for q in range(len(blobs)):
            blobs[q][2]=N[q]