import os
import numpy as np
import xarray as xr
import holoviews as hv
import matplotlib.pyplot as plt
import blobfinder as bf
import phasegradint as pgi
import vortracker as vt


max_tuples=10

ds = xr.open_dataset("fuw hackathon 2018/data/2017-11-03 mag sequence phase.nc")
hvds = hv.Dataset(ds)
phases = hvds.to(hv.Image, kdims=["x", "y"], vdims=["Phase"])
masks = hvds.to(hv.Image, kdims=["x", "y"], vdims=["Absolute value"])
ph_fields = phases.dimension_values("field", expanded=False)
av_fields = masks.dimension_values("field", expanded=False)
fn = len(ph_fields)

fig, pla = plt.subplots(1,4)
X=[]
fig.set_size_inches(20,10)

if not os.path.exists("blobs"):
	os.makedirs("blobs")
bonus=[]
for i in range(fn):
    print(i,ph_fields[i],flush=True)
    ph = phases[ph_fields[i]].dimension_values("Phase", flat=False)
    av = masks[av_fields[i]].dimension_values("Absolute value", flat=False)
    blobs, vorticity, div = bf.findAllBlobs(ph,av)
        
    #print(X[i][1].path())
    for k in range(4):
        pla[k].clear()
    pla[0].imshow(ph)
    pla[1].imshow(av,cmap='hot')
    pla[2].imshow(vorticity, cmap="inferno")
    pla[3].imshow(div,cmap='gist_heat')
    
    for blob in blobs:
        y, x, r = blob
        c = 'b'
        if vorticity[int(y+0.5),int(x+0.5)]>0:
        	c = 'r'
        for j in range(4):
            pla[j].add_patch(plt.Circle((x, y), r, color=c, linewidth=2, fill=False))
            for u in bonus:
                pointsx,pointsy=[],[]
                for q in X:
                    for b in q:
                        if b[2]==u:
                            pointsx.append(b[1])
                            pointsy.append(b[0])
                pla[j].plot(pointsx,pointsy)
    plt.savefig("blobs/"+str(i).rjust(4,'0')+".png")
    blobs=blobs.tolist()
    X.append(blobs)
    if i==0:
        for q in range(len(X[0][0])):
            X[0][q][2]=q
            bonus.append(q)
    else:
        vt.rename(X[i],X[i-1],i,bonus,bf.Rcurve)
        
    
