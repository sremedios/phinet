import numpy as np
import nibabel as nib
import argparse
from scipy.ndimage import morphology

def iterative_fill_and_close(img, structure):
    for _ in range(3):
        img = morphology.binary_fill_holes(img, structure=structure)
    return morphology.binary_closing(img, structure=structure, iterations=4)

def FillHoles2D(vol, structure):
    tmp = np.moveaxis(vol, 2, 0)
    tmp = np.array(list(map(lambda cur_slice: iterative_fill_and_close(cur_slice, structure), tmp)))

    return np.moveaxis(tmp, 0, 2)

def apply_denoise(invol, quant, scale, n_repetitions):
    q = np.percentile(invol[invol>0], quant)
    se = morphology.generate_binary_structure(2, 1)

    for _ in range(n_repetitions):
        mask = invol >= q

        # apply in 2D over each axis
        for _ in range(3):
            mask = FillHoles2D(mask, structure=se)
            mask = np.transpose(mask, axes=(1,2,0))
        
        # apply mask
        invol = invol * mask
        # increase q for next iteration
        q = q * scale 
        
    return invol

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remove background noise from MR images')

    parser.add_argument('--input', required=True, type=str, dest='SRC',
                        help='Source input nifti image from where background noise is to be removed.')
    parser.add_argument('--output', required=True, type=str, dest='TARG',
                        help='Output background noise-free nifti image. ')
    parser.add_argument('--pc', required=False, type=float, default=50.0, dest='QUANT',
                        help='The histogram percentile is used to initialize the noise threshold. '
                             'Default is 50.0, meaning the initial noise threshold is 50%% of the histogram.')
    parser.add_argument('--scale', required=False, type=float, default=1.4, dest='SCALE',
                        help='Scale to increase noise threshold.'
                             'Default is 1.4, meaning the noise threshold will increase by 1.4 each iteration.')
    parser.add_argument('--n_repetitions', required=False, type=int, default=4, dest='N_REPETITIONS',
                        help='Number of repetitions to apply denoising.'
                             'Default is 4 meaning this algorithm will be applied 4 times sequentially.')
    
    results = parser.parse_args()
    SRC = nib.load(results.SRC)
    invol = SRC.get_fdata(dtype=np.float32)
    
    quant = results.QUANT
    scale = results.SCALE
    n_repetitions = results.N_REPETITIONS
    
    invol = apply_denoise(invol=invol, quant=quant, scale=scale, n_repetitions=n_repetitions)

    SRC.header['bitpix'] = 32
    SRC.header['datatype'] = 16
    nii_obj = nib.Nifti1Image(invol, SRC.affine, SRC.header)
    nib.save(nii_obj, results.TARG)
