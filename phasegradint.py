import numpy as np
from scipy import ndimage
from numba import jit

@jit
def interp(phase,mask,x,y):
	x = int(x+0.5)
	y = int(y+0.5)
	#print("x, y =",x,y,flush=True)
	ph = phase[y,x]
	av = mask[y,x]
	av = 1.0
	return av*np.cos(ph), av*np.sin(ph)

@jit
def integrate(phase,mask,x0,y0,r):
	n = 100
	da = 2.0*np.pi/n
	v = 0.0
	for i in range(n):
		a = da*i
		b = da*(i+1)
		xa = x0+r*np.cos(a)
		ya = y0+r*np.sin(a)
		xb = x0+r*np.cos(b)
		yb = y0+r*np.sin(b)
		xa, ya = interp(phase,mask,xa,ya)
		xb, yb = interp(phase,mask,xb,yb)
		v += xa*yb-ya*xb
	return mask[int(y0+0.5),int(x0+0.5)]*v/da/r

@jit
def vorticityMap(phase,mask,r,mask_threshold=0.02,mask_dilation=3):
	phase = np.array(phase)
	mask_data = np.array(mask)

	transf = mask_data / mask_data.max()
	mask = np.zeros_like(transf)
	mask[transf > mask_threshold] = 1
	mask = ndimage.binary_dilation(mask, iterations=mask_dilation)
	mask = ndimage.binary_fill_holes(mask)

	mask_data[mask == 0] = 0.0
	mask = mask_data

	#print("mask:",np.min(mask),np.max(mask),flush=True)
	assert(phase.shape==mask.shape)
	shape = phase.shape
	v = np.zeros_like(mask)
	for ix in range(r,shape[1]-r):
		for iy in range(r,shape[0]-r):
			v[iy,ix] = integrate(phase,mask,ix,iy,r)
	return v
