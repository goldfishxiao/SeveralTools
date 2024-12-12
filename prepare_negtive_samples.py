import os
import torch
from glob import glob
from os.path import join
from tqdm import tqdm


def generate_negative_samples(file_path, result_dir, K=100, dim=3, h=32, w=32):
    samples = torch.randn(dim*h*w, K)
    img_name = os.path.basename(file_path).split('.')[0]
    # print(file_path)
    torch.save(samples, os.path.join(result_dir, img_name + '.pt'))


if __name__ == "__main__":
    from joblib import Parallel, delayed
    import multiprocessing

    input_dir = '/home/ps/Data/ImageRestoration/DFWB_128patches'
    result_dir = '/home/ps/Data/ImageRestoration/DFWB_128patches_negative_samples'
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    input_lists = glob(join(input_dir, '*.*'))
    num_cores = 10
    Parallel(n_jobs=num_cores)(delayed(generate_negative_samples)(file_, result_dir) for file_ in tqdm(input_lists))

