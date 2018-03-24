import numpy as np
import xarray as xr
import holoviews as hv
import goal_3_funcs

ds = xr.open_dataset("fuw hackathon 2018/data/2017-11-03 mag sequence phase.nc")
hvds = hv.Dataset(ds)
phases = hvds.to(hv.Image, kdims=["x", "y"], vdims=["Phase"])
fields = phases.dimension_values("field", expanded=False)

data = phases[fields[1]].dimension_values("Phase", flat=False)

r = np.array([0.70,0.85,1.00,1.15,1.30,1.5])
#r = np.array([0.5,1,1.5])
#r = np.array([1.5])
goal_3_funcs.single_frame(data = data, vortex = [2.4,1.4] ,radii=r,savefig=True)
