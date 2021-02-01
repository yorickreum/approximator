import os

import numpy as np
from netCDF4 import Dataset

dir_path = os.path.dirname(os.path.realpath(__file__))

i, j = 60, 70  # 60, 70 --> lat = 53.275978, lon = 10.553134


def remo_constants():
    filename_bclapp_2 = "/Volumes/Transcend/remo-data/data/bclapp/bclapp2.nc_remap.nc"
    ds_bclapp_2 = Dataset(filename_bclapp_2, "r")  # , format="NETCDF4"
    keys = ds_bclapp_2.variables.keys()
    bclapp_2 = ds_bclapp_2.variables["var9922"][:, i, j].data[0]
    filename_fmpot_2 = "/Volumes/Transcend/remo-data/data/fmpot/fmpot2.nc_remap.nc"
    ds_fmpot_2 = Dataset(filename_fmpot_2, "r")  # , format="NETCDF4"
    keys = ds_fmpot_2.variables.keys()
    fmpot_2 = ds_fmpot_2.variables["var9912"][:, i, j].data[0]
    filename_fksat_2 = "/Volumes/Transcend/remo-data/data/fksat/fksat2.nc_remap.nc"
    ds_fksat_2 = Dataset(filename_fksat_2, "r")  # , format="NETCDF4"
    keys = ds_fksat_2.variables.keys()
    fksat_2 = ds_fksat_2.variables["var9902"][:, i, j].data[0]
    filename_vpor_2 = "/Volumes/Transcend/remo-data/data/vpor/vpor2.nc_remap.nc"
    ds_vpor_2 = Dataset(filename_vpor_2, "r")  # , format="NETCDF4"
    keys = ds_vpor_2.variables.keys()
    vpor_2 = ds_vpor_2.variables["var9932"][:, i, j].data[0]
    pass


# remo_constants()


def remo_wsi_layers_to_csv():
    def get_vars_from_ds(ds):
        var1 = ds.variables["var1"]
        var2 = ds.variables["var2"]
        var3 = ds.variables["var3"]
        var4 = ds.variables["var4"]
        var5 = ds.variables["var5"]
        return var1, var2, var3, var4, var5

    filename_wsi2 = "/Volumes/Transcend/remo-data/data/wsi_1-5_2_remap.nc"
    ds_wsi2 = Dataset(filename_wsi2, "r")
    var1, var2, var3, _, _ = get_vars_from_ds(ds_wsi2)
    t0, t1 = 200, 1600
    wsi_2_h = var2[t0, :, i, j].data
    np.savetxt("./goal1/wsi_2_h.csv", wsi_2_h, delimiter=",")
    wsi_1 = var1[t0:t1, :, i, j].data.ravel()
    wsi_3 = var3[t0:t1, :, i, j].data.ravel()
    t_step_to_h = lambda step: ((step * 4) / (7 * 24 * 60))
    wsi_1_with_t = np.asarray([[t_step_to_h(ti), wsi] for ti, wsi in enumerate(wsi_1)])
    np.savetxt("./goal1/wsi_1_with_t.csv", wsi_1_with_t, delimiter=",")
    wsi_3_with_t = np.asarray([[t_step_to_h(ti), wsi] for ti, wsi in enumerate(wsi_3)])
    np.savetxt("./goal1/wsi_3_with_t.csv", wsi_3_with_t, delimiter=",")


# remo_wsi_layers_to_csv()

wsi_1_with_t = np.genfromtxt(
    os.path.join(dir_path, f'./goal1/wsi_1_with_t.csv'),
    delimiter=',')
wsi_3_with_t = np.genfromtxt(
    os.path.join(dir_path, f'./goal1/wsi_3_with_t.csv'),
    delimiter=',')

remo_wsi_2_h = float(np.genfromtxt(
    os.path.join(dir_path, f'./goal1/wsi_2_h.csv'),
    delimiter=','))

# Layer Depths:
# 1: 0-0.065 m --> 0.065 m thick
# 2: 0.065-0.254 m
# 3: 0.254-0.913 m --> 0.659 m thick
# 4: 0.913-2.902 m
# 5: 2.902-5.7 m

# Layer Thickness: Layer Depths:
# 1: 0.065 m		0-0.065 m
# 2: 0.254 m		0.065-0.319 m
# 3: 0.913 m		0.319-1,232 m
# 4: 2.902 m		1,232-4.134 m
# 5: 5.7 m		4.134-9.834 m

remo_theta_1 = lambda t: np.interp(t, wsi_1_with_t.T[0], wsi_1_with_t.T[1])/0.065
remo_theta_3 = lambda t: np.interp(t, wsi_3_with_t.T[0], wsi_3_with_t.T[1])/0.659

pass
