import os

from tqdm import tqdm
import time
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

import sys
sys.path.append(".")
sys.path.append("..")
import glob
from configs import data_configs
from datasets.inference_dataset import InferenceDataset
from utils.common import tensor2im
from utils.inference_utils import run_inversion
from utils.model_utils import load_model
from options.test_options import TestOptions
from torchvision import transforms
import tqdm
def get_coupled_results(result_batch, transformed_image):
    result_tensors = result_batch[0]  # there's one image in our batch
    final_rec = tensor2im(result_tensors[-1]).resize(resize_amount)
    input_im = tensor2im(transformed_image).resize(resize_amount)
    res = np.concatenate([np.array(input_im), np.array(final_rec)], axis=1)
    res = Image.fromarray(res)
    return res


def run():
    test_opts = TestOptions().parse()

    out_path_results = os.path.join(test_opts.exp_dir, 'inference_results')
    out_path_coupled = os.path.join(test_opts.exp_dir, 'inference_coupled')

    os.makedirs(out_path_results, exist_ok=True)
    os.makedirs(out_path_coupled, exist_ok=True)

    # update test options with options used during training
    net, opts = load_model(test_opts.checkpoint_path, update_opts=test_opts)
    img_transforms = transforms.Compose([
                            transforms.Resize((256, 256)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                        ])

    all_latents = {}
    all_id_path = glob.glob('/root/sample/*')
    for id_path in all_id_path:
        id_name = os.path.basename(id_path)
        cur_out_path_results = out_path_results + f'/{id_name}/'
        cur_out_path_coupled = out_path_coupled + f'/{id_name}/'
        os.makedirs(cur_out_path_coupled, exist_ok=True)
        os.makedirs(cur_out_path_results, exist_ok=True)
        all_image_path = glob.glob(id_path+'/*.jpg')
        for image_path in tqdm.tqdm(all_image_path[:20], desc = id_name):
            original_image = Image.open(image_path).convert("RGB")
            original_image = original_image.resize((256, 256))
            transformed_image = img_transforms(original_image)
            with torch.no_grad():
                tic = time.time()
                result_batch, result_latents, _ = run_inversion(transformed_image.unsqueeze(0).cuda(), 
                                                                net, 
                                                                opts,
                                                                return_intermediate_results=True)
                toc = time.time()
                print('Inference took {:.4f} seconds.'.format(toc - tic))

            resize_amount = (256, 256) if opts.resize_outputs else (opts.output_size, opts.output_size)

            results = [tensor2im(result_batch[0][iter_idx]) for iter_idx in range(opts.n_iters_per_batch)]

            input_im = tensor2im(transformed_image)
            res = np.array(input_im.resize(resize_amount))
            for idx, result in enumerate(results):
                res = np.concatenate([res, np.array(result.resize(resize_amount))], axis=1)
                # save individual outputs
                save_dir = os.path.join(cur_out_path_results, str(idx))
                os.makedirs(save_dir, exist_ok=True)
                result.resize(resize_amount).save(os.path.join(save_dir, os.path.basename(image_path)))
            # save coupled image with side-by-side results
            Image.fromarray(res).save(os.path.join(cur_out_path_coupled, os.path.basename(image_path)))


if __name__ == '__main__':
    run()
