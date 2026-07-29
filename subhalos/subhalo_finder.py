import requests
import h5py
import numpy as np

with open("api_key.txt") as f:
    api_key = f.read()

baseUrl = "http://www.tng-project.org/api/"
headers = {"api-key": api_key}

def get(path, params=None):
    r = requests.get(path, params=params, headers=headers)
    r.raise_for_status()
    if r.headers["content-type"] == "application/json":
        return r.json()
    if "content-disposition" in r.headers:
        filename = r.headers["content-disposition"].split("filename=")[1]
        with open(filename, "wb") as fobj:
            fobj.write(r.content)
        return filename
    return r

subhalo_positions_np = {"x_pos": np.zeros(int(1e7)), "y_pos": np.zeros(int(1e7)), "z_pos": np.zeros(int(1e7))}

# one request --> every subhalo's position at snapshot 135 (z=0)
fn = get(baseUrl + "Illustris-3/files/groupcat-135/", {"Subhalo": "SubhaloPos"})
with h5py.File(fn, "r") as f:
    print("groups in file:", list(f.keys()))
    pos = f["Subhalo"]["SubhaloPos"][:]     # (N, 3)

count = pos.shape[0]
subhalo_positions_np["x_pos"][:count] = pos[:, 0]
subhalo_positions_np["y_pos"][:count] = pos[:, 1]
subhalo_positions_np["z_pos"][:count] = pos[:, 2]
print("filled", count, "subhalos")

# save (only the real entries, not the 10M trailing zeros)
np.savez("illustris3_z0_positions.npz",
         x_pos=subhalo_positions_np["x_pos"][:count],
         y_pos=subhalo_positions_np["y_pos"][:count],
         z_pos=subhalo_positions_np["z_pos"][:count])
print("saved -> illustris3_z0_positions.npz")