"""
downloads the TNG300-1 z=0 group catalog chunk-by-chunk and saves subhalo positions
"""

import os
import re
import time
from urllib.parse import urljoin

import requests
import h5py
import numpy as np

# ---- settings -------------------------------------------------------------
SIM  = "TNG300-1"
SNAP = 99                                   # z = 0 for TNG
OUTDIR = "groupcat_099"                     # where chunk files are stored
API = "https://www.tng-project.org/api/"
LISTING = f"{API}{SIM}/files/groupcat-{SNAP}/?format=api"
# ---------------------------------------------------------------------------

with open("api_key.txt") as f:
    api_key = f.read().strip()
headers = {"api-key": api_key}

os.makedirs(OUTDIR, exist_ok=True)


def chunk_index(name):
    m = re.search(r"\.(\d+)\.hdf5$", name)
    return int(m.group(1)) if m else -1


def list_chunk_urls():
    """Ask the API for the file listing and pull out every .hdf5 link."""
    r = requests.get(LISTING, headers=headers, timeout=120)
    r.raise_for_status()
    hrefs = re.findall(r'href=[\'"]([^\'"]+?\.hdf5[^\'"]*)[\'"]', r.text)
    urls = sorted(set(urljoin(LISTING, h) for h in hrefs),
                key=lambda u: chunk_index(u.split("?")[0]))
    if not urls:
        raise SystemExit(
            "0 .hdf5 links in the listing"
        )
    return urls


def download(url):
    """Stream one chunk to disk; return the local path. Skips if already valid."""
    fname = url.split("?")[0].split("/")[-1]
    path = os.path.join(OUTDIR, fname)
    if os.path.exists(path):
        try:
            with h5py.File(path, "r"):
                return path            # already downloaded and openable
        except Exception:
            os.remove(path)            # corrupt/partial -> redownload

    for attempt in range(1, 6):
        try:
            with requests.get(url, headers=headers, stream=True, timeout=300) as r:
                r.raise_for_status()
                tmp = path + ".part"
                with open(tmp, "wb") as out:
                    for block in r.iter_content(chunk_size=1 << 20):
                        out.write(block)
                os.replace(tmp, path)
            return path
        except (requests.exceptions.RequestException,) as e:
            wait = 15 * attempt
            print(f"    retry {attempt}/5 after error ({type(e).__name__}); waiting {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"failed to download {url}")


print("asking API for the list of chunk files...")
urls = list_chunk_urls()
print(f"catalog has {len(urls)} chunks\n")

for i, url in enumerate(urls):
    print(f"[{i+1}/{len(urls)}] {url.split('?')[0].split('/')[-1]}")
    download(url)

print("\nall chunks present -- reading SubhaloPos...")
files = sorted(
    (os.path.join(OUTDIR, f) for f in os.listdir(OUTDIR) if f.endswith(".hdf5")),
    key=chunk_index,
)
parts = []
for path in files:
    with h5py.File(path, "r") as f:
        if "Subhalo" in f and "SubhaloPos" in f["Subhalo"]:
            parts.append(f["Subhalo"]["SubhaloPos"][:])   # (n_i, 3), ckpc/h

pos = np.concatenate(parts, axis=0)
print("total subhalos:", pos.shape[0])

np.savez("tng300-1_z0_positions.npz",
        x_pos=pos[:, 0], y_pos=pos[:, 1], z_pos=pos[:, 2])
print("saved -> tng300-1_z0_positions.npz")
