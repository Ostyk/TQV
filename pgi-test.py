import numpy as np
import xarray as xr
import holoviews as hv
import matplotlib.pyplot as plt
#import skimage.feature as skif
import phasegradint as pgi

ds = xr.open_dataset("fuw hackathon 2018/data/2017-11-03 mag sequence phase.nc")
hvds = hv.Dataset(ds)
phases = hvds.to(hv.Image, kdims=["x", "y"], vdims=["Phase"])
masks = hvds.to(hv.Image, kdims=["x", "y"], vdims=["Absolute value"])
ph_fields = phases.dimension_values("field", expanded=False)
av_fields = masks.dimension_values("field", expanded=False)
fn = 3#len(ph_fields)
fig, pla = plt.subplots(fn,3)
for i in range(fn):
	ph = phases[ph_fields[0]].dimension_values("Phase", flat=False)
	av = masks[av_fields[0]].dimension_values("Absolute value", flat=False)
	vorticity = np.abs(pgi.vorticityMap(ph,av,3))
	vmin = vorticity.min()
	vmean = vorticity.mean()
	vmax = vorticity.max()
	vstd = vorticity.std()
	vorticity = np.maximum((vorticity-vmean)/vstd-3.0,0.0)
	#blobs_log = skif.blob_log(vorticity)
	#blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)
	print(i,vmin,vmean,vmax,vstd,flush=True)
	pla[i,0].imshow(ph)
	pla[i,1].imshow(av)
	pla[i,2].imshow(vorticity,cmap="inferno")
	'''for blob in blobs:
		y, x, r = blob
		c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
		pla[i,0].add_patch(c)
		pla[i,1].add_patch(c)
		pla[i,2].add_patch(c)'''
plt.show()
