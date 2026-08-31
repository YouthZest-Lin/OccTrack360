# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import division
import argparse
import shutil
import copy
import os
import time
import warnings
from os import path as osp
import importlib
import torch
import torch.distributed as dist
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist

from mmdet3d.apis import init_random_seed
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import collect_env, get_root_logger
from mmdet.apis import set_random_seed

from datetime import datetime
from loaders.builder import build_iter_dataloader
from mmdet.core import EvalHook, DistEvalHook
from models.hooks import CustomDistEvalHook
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (build_optimizer, load_checkpoint, build_runner,
                         Fp16OptimizerHook)

try:
    # If mmdet version > 2.20.0, setup_multi_processes would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import setup_multi_processes
except ImportError:
    from mmdet3d.utils import setup_multi_processes


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--auto-resume',
        action='store_true',
        help='resume from the latest checkpoint automatically')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='(Deprecated, please use --gpu-id) number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--diff-seed',
        action='store_true',
        help='Whether or not set different seeds for different ranks')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both specified, '
            '--options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # register custom module
    importlib.import_module('models')
    importlib.import_module('loaders')
    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if args.resume_from is not None:
        cfg.resume_from = args.resume_from

    if args.auto_resume:
        cfg.auto_resume = args.auto_resume
        warnings.warn('`--auto-resume` is only supported when mmdet'
                      'version >= 2.20.0 for 3D detection model or'
                      'mmsegmentation verision >= 0.21.0 for 3D'
                      'segmentation model')

    if args.gpus is not None:
        cfg.gpu_ids = range(1)
        warnings.warn('`--gpus` is deprecated because we only support '
                      'single GPU mode in non-distributed training. '
                      'Use `gpus=1` now.')
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed training. Use the first GPU '
                      'in `gpu_ids` now.')
    if args.gpus is None and args.gpu_ids is None:
        cfg.gpu_ids = [args.gpu_id]

    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
        rank = 0
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        rank, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)
        gpu = rank % torch.cuda.device_count()
        os.environ['LOCAL_RANK'] = str(gpu)

    work_dir = cfg.get('work_dirs', './work_dirs')
    debug = cfg.get('debug', False)
    # resume or start a new run
    if cfg.resume_from is not None:
        assert os.path.isfile(cfg.resume_from)
        work_dir = os.path.dirname(cfg.resume_from)
    elif debug:
        run_name = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
        work_dir = os.path.join(work_dir, cfg.model.type, run_name)
        os.makedirs(work_dir, exist_ok=True)
    else:
        work_dir = osp.join(work_dir,
                                osp.splitext(osp.basename(args.config))[0])
        os.makedirs(work_dir, exist_ok=True)

    cfg.dump(osp.join(work_dir, osp.basename(args.config)))
    # init the logger before other steps
    log_file = os.path.join(work_dir, 'train.log')

    logger = get_root_logger(
        log_file=log_file, log_level=cfg.log_level, name='mmdet')

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    seed = init_random_seed(args.seed)
    seed = seed + dist.get_rank() if args.diff_seed else seed
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = osp.basename(args.config)

    logger.info('Creating model: %s' % cfg.model.type)
    model = build_model(cfg.model)
    model.init_weights()

    sync_bn = cfg.get('sync_bn', False)
    if distributed and sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        print('Convert to SyncBatchNorm')
    
    logger.info(f'Model:\n{model}')

    n_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    logger.info('Trainable parameters: %d (%.1fM)' % (n_params, n_params / 1e6))
    logger.info('Batch size per GPU: %d' % (cfg.data.samples_per_gpu))
    
    logger.info('Loading training set from %s' % cfg.dataset_root)
    
    train_dataset = build_dataset(cfg.data.train)
    train_runner_type = 'IterBasedRunner' if 'runner' not in cfg else cfg.runner[
        'type']
    world_size = 1
    train_loader = build_iter_dataloader(
        train_dataset,
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        num_gpus=world_size,
        dist=distributed,
        shuffle=True,
        seed=cfg.seed,
        runner_type=train_runner_type,
        val=False
    )

    logger.info('Loading validation set from %s' % cfg.dataset_root)
    val_dataset = build_dataset(cfg.data.val)

    val_loader = build_iter_dataloader(
        val_dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        num_gpus=world_size,
        dist=distributed,
        shuffle=False,
        runner_type=cfg.data.test_dataloader.get('runner_type', 'IterBasedRunner'), #val_runner_type,
        val=True,
    )

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)
    
    logger.info('Creating optimizer: %s' % cfg.optimizer.type)
    optimizer = build_optimizer(model, cfg.optimizer)
    
    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            batch_processor=None,
            optimizer=optimizer,
            work_dir=work_dir,
            logger=logger,
            meta=meta))

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    else:
        optimizer_config = cfg.optimizer_config

    runner.register_lr_hook(cfg.lr_config)
    runner.register_optimizer_hook(optimizer_config)
    runner.register_checkpoint_hook(cfg.checkpoint_config)
    runner.register_logger_hooks(cfg.log_config)
    runner.register_timer_hook(dict(type='IterTimerHook'))

    custom_hooks_config=cfg.get('custom_hooks', None)
    runner.register_custom_hooks(custom_hooks_config)

    eval_cfg = cfg.get('evaluation', {})
    eval_hook = DistEvalHook if distributed else EvalHook
    # In this PR (https://github.com/open-mmlab/mmcv/pull/1193), the
    # priority of IterTimerHook has been modified from 'NORMAL' to 'LOW'.
    # eval_cfg['jsonfile_prefix'] = osp.join('val', cfg.work_dir, time.ctime().replace(' ','_').replace(':','_'))
    if cfg.get('use_custom_eval_hook', False):
        eval_hook = CustomDistEvalHook if distributed else eval_hook
    runner.register_hook(
        eval_hook(val_loader, **eval_cfg), priority='LOW')


    if cfg.resume_from is not None:
        logger.info('Resuming from %s' % cfg.resume_from)
        runner.resume(cfg.resume_from)

    elif cfg.load_from is not None:
        logger.info('Loading checkpoint from %s' % cfg.load_from)
        if cfg.revise_keys is not None:
            load_checkpoint(
                model, cfg.load_from, map_location='cpu',
                revise_keys=cfg.revise_keys
            )
        else:
            load_checkpoint(
                model, cfg.load_from, map_location='cpu',
            )

    runner.run([train_loader], [('train', 1)])

if __name__ == '__main__':
    main()
