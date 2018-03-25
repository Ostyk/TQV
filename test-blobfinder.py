import numpy as np
import xarray as xr
import holoviews as hv
import matplotlib.pyplot as plt
import blobfinder as bf

ds = xr.open_dataset("fuw hackathon 2018/data/2017-11-03 mag sequence phase.nc")
hvds = hv.Dataset(ds)
phases = hvds.to(hv.Image, kdims=["x", "y"], vdims=["Phase"])
masks = hvds.to(hv.Image, kdims=["x", "y"], vdims=["Absolute value"])
ph_fields = phases.dimension_values("field", expanded=False)
av_fields = masks.dimension_values("field", expanded=False)
fn = len(ph_fields)

fig, pla = plt.subplots(1,4)

fig.set_size_inches(20,10)

for i in range(fn):
	print(i,ph_fields[i],flush=True)

	ph = phases[ph_fields[i]].dimension_values("Phase", flat=False)
	av = masks[av_fields[i]].dimension_values("Absolute value", flat=False)
	blobs, vorticity, div = bf.findAllBlobs(ph,av)

	for k in range(4):
		pla[k].clear()
	pla[0].imshow(ph)
	pla[1].imshow(av,cmap='hot')
	pla[2].imshow(vorticity, cmap="inferno")
	pla[3].imshow(div,cmap='gist_heat')
	for blob in blobs:
		y, x, r = blob
		for j in range(4):
			pla[j].add_patch(plt.Circle((x, y), r, color='w', linewidth=2, fill=False))
	plt.savefig("blobs/"+str(i).rjust(4,'0')+".png")
