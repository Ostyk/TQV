import numpy as np
from scipy import ndimage
from numba import jit

@jit
def interp(phase,x,y):
	''' get unit vector from phase value at real x and y indices'''
	x = int(x+0.5)
	y = int(y+0.5)
	ph = phase[y,x]
	av = 1.0
	return av*np.cos(ph), av*np.sin(ph)

@jit
def integrate(phase,mask,x0,y0,r,dl=0.25):

	''' integrate phase over circle's edge with radius r and center at x0,y0; multiply result by mask integral over filled circle '''

	n = int(2.0*np.pi*r/dl+0.5)
	da = 2.0*np.pi/n
	v = 0.0
	for i in range(n):
		a = da*i
		b = da*(i+1)
		xa = x0+r*np.cos(a)
		ya = y0+r*np.sin(a)
		xb = x0+r*np.cos(b)
		yb = y0+r*np.sin(b)
		xa, ya = interp(phase,xa,ya)
		xb, yb = interp(phase,xb,yb)
		v += np.arcsin(xa*yb-ya*xb)

	I = 0.0
	n = 0
	for ix in range(-r,r+1):
		for iy in range(-r,r+1):
			if ix*ix+iy*iy<=r*r:
				I += mask[y0+iy,x0+ix]
				n += 1
	I /= n

	return I*v/da/r

@jit
def circleMean(data,x0,y0,r):

	''' mean value of data over circle inside '''

	I = 0.0
	n = 0
	for ix in range(-r,r+1):
		for iy in range(-r,r+1):
			if ix*ix+iy*iy<=r*r:
				I += data[y0+iy,x0+ix]
				n += 1
	return I/n

@jit
def cutoff(mask_data,mask_threshold=0.02,mask_dilation=3):
	transf = mask_data / mask_data.max()
	mask = np.zeros_like(transf)
	mask[transf > mask_threshold] = 1
	mask = ndimage.binary_dilation(mask, iterations=mask_dilation)
	mask = ndimage.binary_fill_holes(mask)

	mask_data[mask == 0] = 0.0
	return np.array(mask_data)

@jit
def select(mask_data,low,high,mask_threshold=0.02,mask_dilation=3):
	transf = mask_data / mask_data.max()
	mask = np.zeros_like(mask_data)
	mask[transf > mask_threshold] = 1
	mask = ndimage.binary_dilation(mask, iterations=mask_dilation)
	mask = ndimage.binary_fill_holes(mask)

	mask_data[mask <= 0] = low
	mask_data[mask > 0] = high
	return np.array(mask_data)

@jit
def vorticity(phase,mask,r,mask_threshold=0.02,mask_dilation=3,dl=0.25):
	''' calculate vorticity weighted by mask for almost every point '''
	assert(phase.shape==mask.shape)

	phase = np.array(phase)
	mask = np.array(mask)
	#mask = cutoff(np.array(mask),mask_threshold,mask_dilation)

	shape = phase.shape
	v = np.zeros_like(mask)
	for ix in range(r,shape[1]-r):
		for iy in range(r,shape[0]-r):
			v[iy,ix] = integrate(phase,mask,ix,iy,r,dl)

	return v

@jit
def dist(l1,l2):
    #self explanatory. distance between two blobs
    return (l1[1]-l2[1])**2+(l1[2]-l2[2])**2

def rename(blobs,past,i):
    #sets up name for vortices according to rule, that 
    D=np.ndarray((len(blobs),len(past)))
    for w in range(len(blobs)):
        for ww in range(len(past)):
            D[w][ww]=dist(blobs[w],past[ww])    
    if len(past)>=len(blobs):
        #print(D)
        for w in range(len(blobs)):
            mini=np.argmin(D[w])
            blobs[w][4]=past[mini][4]
    else:
        #print(D.T)
        N=np.arange(len(blobs))+np.ones(len(blobs))*i*100
        for w in range(len(blobs)):
            try:
                mini=np.argmin(D.T[w])
                N[mini]=past[w][4]
            except IndexError:
                #print('new label 2'+str(i))
                continue
        blobs[:,4]=N
