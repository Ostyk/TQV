import numpy as np
import xarray as xr
import holoviews as hv
import matplotlib.pyplot as plt
from skimage.feature import blob_dog
from skimage.filters import gaussian
import phasegradint as pgi
from collections import OrderedDict

ds = xr.open_dataset("fuw hackathon 2018/data/2017-11-03 mag sequence phase.nc")
hvds = hv.Dataset(ds)
phases = hvds.to(hv.Image, kdims=["x", "y"], vdims=["Phase"])
masks = hvds.to(hv.Image, kdims=["x", "y"], vdims=["Absolute value"])
ph_fields = phases.dimension_values("field", expanded=False)
av_fields = masks.dimension_values("field", expanded=False)
fn = len(ph_fields)
ph = OrderedDict()
av = OrderedDict()
vorticity = OrderedDict()

def findBlob(data,x0,y0,r,maxDelta):
	x0 = int(x0+0.5)
	y0 = int(y0+0.5)
	xm, ym = x0, y0
	sm = pgi.circleSum(data,x0,y0,r)
	for ix in range(-maxDelta,maxDelta+1):
		for iy in range(-maxDelta,maxDelta+1):
			if ix*ix+iy*iy<=maxDelta*maxDelta:
				s = pgi.circleSum(data,x0+ix,y0+iy,r)
				if s<sm:
					sm = s
					xm = x0+ix
					ym = y0+iy
	return xm, ym

Rcurve = 3
blur = 3.0
minSigma = 2
maxSigma = 7
threshold = 3.0
Rcorrect = 1

def normalize(x):
	xmean = x.mean()
	xstd = x.std()
	return (x-xmean)/xstd

fig, pla = plt.subplots(1,3)

for i in range(fn):
	print(i,flush=True)
	ph[i] = phases[ph_fields[i]].dimension_values("Phase", flat=False)
	av[i] = masks[av_fields[i]].dimension_values("Absolute value", flat=False)
	vorticity[i] = pgi.vorticityMap(ph[i], av[i], Rcurve)
	vorticity[i] = gaussian(vorticity[i],blur)
	vorticity[i] = normalize(vorticity[i])

	blobs = blob_dog(vorticity[i], min_sigma=minSigma, max_sigma=maxSigma, threshold=threshold)
	blobs[:, 2] = blobs[:, 2] * np.sqrt(2)

	blobs2 = blob_dog(-vorticity[i], min_sigma=minSigma, max_sigma=maxSigma, threshold=threshold)
	blobs2[:, 2] = blobs2[:, 2] * np.sqrt(2)
	blobs = list(blobs)+list(blobs2)

	for k in range(3):
		pla[k].clear()

	pla[0].imshow(ph[i])
	pla[1].imshow(av[i])
	pla[2].imshow(vorticity[i], cmap="inferno")
	for blob in blobs:
		y, x, r = blob
		x2, y2 = findBlob(av[i],x,y,Rcorrect,int(r/2+0.5))
		for j in range(3):
			pla[j].add_patch(plt.Circle((x2, y2), r, color='w', linewidth=1, fill=False))
	plt.savefig("blobs-"+str(i).rjust(4,'0')+".png")

#plt.show()
