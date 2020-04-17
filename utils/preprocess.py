import os
import sys
import shutil
from pathlib import Path

def preprocess(src, dst, fixed_fpath, ants_reg_path, denoise_path):
    tmp_dir = Path("tmp_preprocessing")
    if not tmp_dir.exists():
        tmp_dir.mkdir(parents=True)

    # orient to RAI
    print("Orienting {} to RAI...".format(src.name))
    reoriented_fpath = tmp_dir / ("oriented_" + src.name)
    reorient_call = "3dresample -orient RAI -inset {} -prefix {} > /dev/null 2>&1"
    os.system(reorient_call.format(src, reoriented_fpath))

    # orient fixed image to RAI
    print("Orienting {} to RAI...".format(fixed_fpath.name))
    reoriented_fixed_fpath = tmp_dir / ("oriented_" + fixed_fpath.name)
    os.system(reorient_call.format(fixed_fpath, reoriented_fixed_fpath))

    # register to RAI fixed
    print("Registering {} to {}...".format(reoriented_fpath.name, reoriented_fixed_fpath.name))
    registered_fpath = tmp_dir / ("registered_" + src.name)
    register_call = "{} {} {} {} {} > /dev/null 2>&1"
    os.system(register_call.format(
        ants_reg_path, 
        reoriented_fixed_fpath, 
        reoriented_fpath, 
        "fastfortesting",
        registered_fpath,
    ))

    # denoise
    print("Denoising {} to {}...".format(registered_fpath.name, dst.name))
    denoise_call = "python {} --input={} --output={} --pc={} --scale={} --n_repetitions={} > /dev/null 2>&1"
    os.system(denoise_call.format(
        denoise_path,
        registered_fpath,
        dst,
        40,
        1.2,
        4,
    ))

    shutil.rmtree(tmp_dir)

if __name__ == '__main__':
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    fixed_fpath = Path(sys.argv[3])
    ants_reg_path = Path(sys.argv[4])
    denoise_path = Path(sys.argv[5])
    preprocess(src, dst, fixed_fpath, ants_reg_path, denoise_path)
