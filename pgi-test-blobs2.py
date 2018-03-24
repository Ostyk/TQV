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
fn = 3#len(ph_fields)
fig, pla = plt.subplots(fn,3)
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

R = 2

for i in range(fn):
	ph[i] = phases[ph_fields[i+9]].dimension_values("Phase", flat=False)
	av[i] = masks[av_fields[i+9]].dimension_values("Absolute value", flat=False)
	vorticity[i] = pgi.vorticityMap(ph[i], av[i], R)
	vmin = vorticity[i].min()
	vmean = vorticity[i].mean()
	vmax = vorticity[i].max()
	vstd = vorticity[i].std()
	#vorticity[i] = np.maximum((vorticity[i]-vmean)/vstd,0.0)
	vorticity[i] = (vorticity[i]-vmean)/vstd
	#vorticity[i] = (vorticity[i]-vorticity[i].min())/(vorticity[i].max()-vorticity[i].min())
	#print("norm",vorticity[i].min(),vorticity[i].max())

	vorticity[i] = gaussian(vorticity[i],2.0)

	blobs = blob_dog(vorticity[i], min_sigma=2, max_sigma=7)
	blobs[:, 2] = blobs[:, 2] * np.sqrt(2)

	blobs2 = blob_dog(-vorticity[i], min_sigma=2, max_sigma=7)
	blobs2[:, 2] = blobs2[:, 2] * np.sqrt(2)
	blobs = list(blobs)+list(blobs2)

	print(i,vmin,vmean,vmax,vstd,flush=True)
	pla[i,0].imshow(ph[i])
	pla[i,1].imshow(av[i])
	pla[i,2].imshow(vorticity[i], cmap="inferno")
	for blob in blobs:
		y, x, r = blob
		x2, y2 = findBlob(av[i],x,y,1,int(r/2+0.5))
		for j in range(3):
			#pla[i,j].add_patch(plt.Circle((x, y), r, color='r', linewidth=1, fill=False))
			pla[i,j].add_patch(plt.Circle((x2, y2), r, color='g', linewidth=1, fill=False))

plt.show()
