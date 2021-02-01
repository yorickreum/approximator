Goal: Check if Campbell / Clapp & Hornberger Simulation works, with "real" boundary data from REMO

use (time-dependent) WSI of layer 1 as upper boundary
use (time-dependent) WSI of layer 3 as lower boundary
initial condition whole layer has homogenous WSI wsi_h corresponding to initial WSI in REMO data,
  smooth out edges using (wsi_h - (wsi_h + wsi_e) * torch.exp(-(t / 0.000001) ** 2)) for both edges  
--> simulate WSI of layer 2
--> integrate WSI in layer 2 to compare with plot from remo

WSI of layer 1/3 --> get data from REMO data, use numpy.interp to interpolate --> put into residual

time steps [200, 1600) of REMO data --> 5600 min = 3,8888 days = 0,55555 weeks

1: 0 - 0.065 m
2: 0.065 - 0.254 m
3: 0.254 - 0.913 m

filename_bclapp_1 = "/Volumes/Transcend/remo-data/data/bclapp/bclapp1.nc_remap.nc"
ds_bclapp_1 = Dataset(filename_bclapp_1, "r")  # , format="NETCDF4"
ds_bclapp_1.variables["var9921"][:,i,j]

filename_fmpot_1 = "/Volumes/Transcend/remo-data/data/fmpot/fmpot1.nc_remap.nc"
ds_fmpot_1 = Dataset(filename_fmpot_1, "r")  # , format="NETCDF4"
ds_fmpot_1.variables["var9911"][:,i,j]

filename_fksat_1 = "/Volumes/Transcend/remo-data/data/fksat/fksat1.nc_remap.nc"
ds_fksat_1 = Dataset(filename_fksat_1, "r")  # , format="NETCDF4"
ds_fksat_1.variables["var9901"][:,i,j]

filename_vpor_1 = "/Volumes/Transcend/remo-data/data/vpor/vpor1.nc_remap.nc"
ds_vpor_1 = Dataset(filename_vpor_1, "r")  # , format="NETCDF4"
ds_vpor_1.variables.keys()
ds_vpor_1.variables["var9931"][:,i,j]