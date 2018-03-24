import numpy as np
import xarray as xr
import holoviews as hv
import matplotlib.pyplot as plt
from skimage.feature import blob_dog
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
for i in range(fn):
	ph[i] = phases[ph_fields[0]].dimension_values("Phase", flat=False)
	av[i] = masks[av_fields[0]].dimension_values("Absolute value", flat=False)
	vorticity[i] = np.abs(pgi.vorticityMap(ph[i], av[i], 3))
	vmin = vorticity[i].min()
	vmean = vorticity[i].mean()
	vmax = vorticity[i].max()
	vstd = vorticity[i].std()
	vorticity[i] = np.maximum((vorticity[i]-vmean)/vstd-3.0,0.0)
	#blobs_log = skif.blob_log(vorticity)
	#blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)
	print(i,vmin,vmean,vmax,vstd,flush=True)
	pla[i,0].imshow(ph[i])
	pla[i,1].imshow(av[i])
	pla[i,2].imshow(vorticity[i], cmap="inferno")

blobs = blob_dog(vorticity[0], min_sigma=2, max_sigma=7)
blobs[:, 2] = blobs[:, 2] * np.sqrt(2)

fig, ax = plt.subplots()

ax.imshow(vorticity[0])
for blob in blobs:
    y, x, r = blob
    c = plt.Circle((x, y), r, color='r', linewidth=2, fill=False)
    ax.add_patch(c)
plt.show()
