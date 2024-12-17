import os
import torch
from glob import glob
from os.path import join
from tqdm import tqdm
import numpy as np
from joblib import Parallel, delayed
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--patch_size", default=128, type=int, help="input patch size")
options = parser.parse_args()
# python data/prepare_negtive_samples.py --patch_size 128


def generate_negative_samples(file_path, result_dir, K=10, dim=3, h=32, w=32):
    samples = np.random.randn(K, dim, h, w).astype(np.float32)
    img_name = os.path.basename(file_path).split('.')[0]
    # print(file_path)
    np.save(os.path.join(result_dir, img_name + '.npy'), samples)


if __name__ == "__main__":

    input_dir = '/home/ps/Data/ImageRestoration/DFWB_{}patches'.format(options.patch_size)
    result_dir = '/home/ps/Data/ImageRestoration/DFWB_{}patches_negative_samples'.format(options.patch_size)
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    input_lists = glob(join(input_dir, '*.*'))
    # num_cores = 10
    # patch_size = options.patch_size
    # Parallel(n_jobs=num_cores)(delayed(generate_negative_samples)(file_, result_dir, K=10, dim=3, h=patch_size//4,
    #                                                               w=patch_size//4) for file_ in tqdm(input_lists))
    for file_ in tqdm(input_lists):
        generate_negative_samples(file_, result_dir, K=10, dim=3, h=options.patch_size//4, w=options.patch_size//4)

