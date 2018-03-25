import numpy as np
from skimage.feature import blob_dog
from skimage.filters import gaussian
import phasegradint as pgi
from numba import jit

Rcurve = 3
blur = 3.0
minSigma = Rcurve-1
maxSigma = Rcurve+1
threshold = 2.5
Rcorrect = 1

@jit
def normalize(x):
	xmean = x.mean()
	xstd = x.std()
	return (x-xmean)/xstd

@jit
def findBetterBlob(vorticity,div,x0,y0,r,maxDelta):
	x0 = int(x0+0.5)
	y0 = int(y0+0.5)
	xm, ym = x0, y0
	sm = -1
	for ix in range(-maxDelta,maxDelta+1):
		for iy in range(-maxDelta,maxDelta+1):
			if ix*ix+iy*iy<=maxDelta*maxDelta:
				s = pgi.circleMean(vorticity,x0+ix,y0+iy,r)
				t = pgi.circleMean(div,x0+ix,y0+iy,r)
				s = s*s+t*np.abs(t)
				if s>sm:
					sm = s
					xm = x0+ix
					ym = y0+iy
	return xm, ym

@jit
def calcDiv(absval):
	div = np.gradient(absval)
	x = np.gradient(div[0])[0]
	y = np.gradient(div[1])[1]
	div = gaussian(x+y,0.5)
	return div

@jit
def calcWeightedVorticity(phase,weights):
	vorticity = pgi.vorticity(phase,weights,Rcurve)
	vorticity = gaussian(vorticity,blur)
	vorticity = normalize(vorticity)
	return vorticity

@jit
def findAllBlobs(phase,absval):
	div = calcDiv(absval)
	vorticity = calcWeightedVorticity(phase,div)
	blobs = blob_dog(np.abs(vorticity), min_sigma=minSigma, max_sigma=maxSigma, threshold=threshold)
	blobs[:, 2] = blobs[:, 2] * np.sqrt(2)
	for blob in blobs:
		y, x, r = blob
		blob[1], blob[0] = findBetterBlob(vorticity,div,x,y,Rcorrect,int(r+0.5))
	return blobs, vorticity, div
