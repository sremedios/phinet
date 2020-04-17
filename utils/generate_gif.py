import os
import sys
import shutil
from tqdm import tqdm
from pathlib import Path
import numpy as np
import nibabel as nib
from PIL import Image
import matplotlib.pyplot as plt
import turbo

cmap = plt.get_cmap('turbo')

TMP_DIR = Path("tmp")
fname = Path(sys.argv[1])
TARGET_DIR = Path(sys.argv[2])
for d in [TMP_DIR, TARGET_DIR]:
    if not d.exists():
        d.mkdir(parents=True)

img = nib.load(fname).get_fdata(dtype=np.float32)
# move axial index to front
img = np.moveaxis(img, 2, 0)

for i, axial_slice in tqdm(enumerate(img), total=len(img)):
    out_fname = TMP_DIR / "{}_{:03d}.png".format(fname.name.split('.')[0], i)
    colored_slice = cmap(axial_slice.T)
    colored_slice = (colored_slice[:, :, :3] * 255).astype(np.uint8)
    Image.fromarray(colored_slice).save(out_fname)


# use ImageMagick to create gif
delay = str(int(1400/len(img)))
imagemagick_cmd = "convert -delay {} -loop 0 {} {}"
os.system(
    imagemagick_cmd.format(
        delay,
        TMP_DIR / '*',
        TARGET_DIR / (fname.name.split('.')[0] + ".gif"),
    )
)

# delete all ims from tmp
shutil.rmtree(TMP_DIR)
