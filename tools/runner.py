import torch
import torch.nn as nn
import os
import json
from tools import builder
from utils.logger import *
from torch import Tensor
import time
import sys
import numpy as np
import random
import numpy as np

from models.MCNet.autoencoder import *
from datasets.io import IO

import numpy as np

def normalize_point_clouds(pcs, mode):
    for i in range(pcs.size(0)):
        pc = pcs[i]
        if mode == 'shape_unit':
            shift = pc.mean(dim=0).reshape(1, 3)
            scale = pc.flatten().std().reshape(1, 1)
        elif mode == 'shape_bbox':
            pc_max, _ = pc.max(dim=0, keepdim=True) # (1, 3)
            pc_min, _ = pc.min(dim=0, keepdim=True) # (1, 3)
            shift = ((pc_min + pc_max) / 2).view(1, 3)
            scale = (pc_max - pc_min).max().reshape(1, 1) / 2
        pc = (pc - shift) / scale
        pcs[i] = pc
    return pcs, shift, scale

def denormalize_multi_recons(pcs, pcs_shift, pcs_scale):
    for i in range(pcs.size(0)):
        pc = pcs[i]
        pc = (pc * pcs_scale) + pcs_shift
        pcs[i] = pc
    return pcs

def normalize_partial(pcs, pcs_shift, pcs_scale):
    for i in range(pcs.size(0)):
        pc = pcs[i]
        pc_shift = pcs_shift[i]
        pc_scale = pcs_scale[i]
        pc = (pc - pcs_shift) / pcs_scale
        pcs[i] = pc
    return pcs

def denormalize_pc(pcs, pcs_shift, pcs_scale):
    for i in range(pcs.size(0)):
        pc = pcs[i]
        pc_shift = pcs_shift[i]
        pc_scale = pcs_scale[i]
        pc = (pc * pc_scale) + pc_shift
        pcs[i] = pc
    return pcs

### shapenet 34
def test_shapenet34(args, config):
    logger = get_logger(args.log_name)
    # Model
    logger.info('Loading model...')
    MCNet = AutoEncoder(args).to(args.gpu)
    # load checkpoints
    builder.load_model(MCNet, args.ckpts, logger = logger)
    if args.use_gpu:
        MCNet.to(args.local_rank)
    #  DDP
    if args.distributed:
        raise NotImplementedError()

    MCNet.eval()  # set model to eval mode
    with torch.no_grad():

        gt_root = './experiments/demo/gt/'
        partial_root = './experiments/demo/partial/'

        output_points = args.output_points # can be any number. e.g. 16, 32, 64, 128, 5000, 8000, 10000, 300000, etc.
        num_multi_completion = args.num_multi_completion # can be any number. e.g. 5, 10, 20, 50, 100, 200, etc.
        lamda = args.lamda # from 0.0 to 1.0   e.g. 0.2, 0.3, 0.4, 0.5, etc. 
        
        pc_file_list = os.listdir(gt_root)
        for pc_file in pc_file_list:
            model_id = pc_file.split('.')[0]
            gt_file = os.path.join(gt_root, pc_file)
            partial_file = os.path.join(partial_root, pc_file)

            gt_ndarray = IO.get(gt_file).astype(np.float32)
            gt = torch.from_numpy(gt_ndarray).unsqueeze(0).cuda() # [1, 16384, 3]
            partial_ndarray = IO.get(partial_file).astype(np.float32)
            partial = torch.from_numpy(partial_ndarray).unsqueeze(0).cuda() # [1, 2048, 3]

            gt, gt_shift, gt_scale = normalize_point_clouds(gt, 'shape_unit')
            partial = normalize_partial(partial, gt_shift, gt_scale)
            
            code = MCNet.encode(partial) 

            recons = MCNet.diffusion_sample(code, output_points, flexibility=args.flexibility, ret_traj=False)
            final_recons = torch.cat((partial, recons), 1)
            allpc = torch.cat((final_recons, gt), 1)
            
            z = torch.randn([num_multi_completion, args.latent_dim]).to(args.device)
            recons_rand_shape, z_recons = MCNet.sample(z, output_points, flexibility=args.flexibility, ret_traj=False)
            code = code.repeat(num_multi_completion,1)

            interpolated_codes = (1.0 - lamda) * code + lamda * z_recons
            
            multi_recons = MCNet.diffusion_sample(interpolated_codes, output_points, flexibility=args.flexibility, ret_traj=False)
            
            partial = denormalize_pc(partial, gt_shift, gt_scale)
            recons = denormalize_pc(recons, gt_shift, gt_scale)
            final_recons = torch.cat((partial, recons), 1)
            gt = denormalize_pc(gt, gt_shift, gt_scale)

            multi_recons = denormalize_multi_recons(multi_recons, gt_shift, gt_scale)
            multi_recons_final = torch.cat((multi_recons, partial.repeat(num_multi_completion,1,1)), 1)


            
    return
