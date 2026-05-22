import traceback
import training_old_version
training_old_version.use_old_version()
assert training_old_version.get_old_version() == True
import argparse
import os
import shutil
import warnings
warnings.filterwarnings('ignore')
import random
import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,wrap_fp16_model)
import mmdet
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet3d.apis import multi_gpu_test
from mmdet.apis import set_random_seed
from mmdet.datasets import replace_ImageToTensor
import numpy as np
from PIL import Image

from collections import OrderedDict
if mmdet.__version__ > '2.23.0':
    # If mmdet version > 2.23.0, setup_multi_processes would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import setup_multi_processes
else:
    from mmdet3d.utils import setup_multi_processes

try:
    # If mmdet version > 2.23.0, compat_cfg would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import compat_cfg
except ImportError:
    from mmdet3d.utils import compat_cfg

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('--config',default='./configs/sgd3d/vod-R4Det_det3d_2x4_12e.py', help='test config file path')
    parser.add_argument('--checkpoint', default='./checkpoints/FINAL-VoD.pth', help='checkpoint file')
    parser.add_argument('--watermark', type=str, default='none', help='use watermark')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument('--saveoutput', type=str, default='none', help='save an output file for ui')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        default=['bbox'],
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--no-aavt',
        action='store_true',
        help='Do not align after view transformer.')
    parser.add_argument(
        '--aavt',
        action='store_true',
        help='Do not align after view transformer.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
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
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local-rank', '--local_rank', type=int, default=-8848)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    try:
        args = parse_args()
        SEED = 0
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        assert args.out or args.eval or args.format_only or args.show \
            or args.show_dir, \
            ('Please specify at least one operation (save/eval/format/show the '
             'results / save the results) with the argument "--out", "--eval"'
             ', "--format-only", "--show" or "--show-dir"')
        if args.format_only:
            print('Only for submission ...')
            args.eval = None

        if 'waymo' in args.config.lower():
            args.eval = ['waymo']

        if args.eval and args.format_only:
            raise ValueError('--eval and --format_only cannot be both specified')

        if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
            raise ValueError('The output file must be a pkl file.')

        cfg = Config.fromfile(args.config)

        if args.cfg_options is not None:
            cfg.merge_from_dict(args.cfg_options)
        project = args.config.split('/')[-1].split('.')[0]
        if 'vod' in project.lower():
            src_path = 'tools/eval_tools/vod-eval.py'
            dst_path = 'mmdet3d/core/evaluation/kitti_utils/eval.py'
            shutil.copy(src_path, dst_path)
            src_path = 'tools/eval_tools/vod-kitti_dataset.py'
            dst_path = 'mmdet3d/datasets/kitti_dataset.py'
            shutil.copy(src_path, dst_path)
            print('USING EVAL TOOLS OF VOD DATASET')
        if 'tj4d' in project.lower():
            src_path = 'tools/eval_tools/TJ4D-eval.py'
            dst_path = 'mmdet3d/core/evaluation/kitti_utils/eval.py'
            shutil.copy(src_path, dst_path)
            src_path = 'tools/eval_tools/TJ4D-kitti_dataset.py'
            dst_path = 'mmdet3d/datasets/kitti_dataset.py'
            shutil.copy(src_path, dst_path)
            print('USING EVAL TOOLS OF TJ4D DATASET')

        if not 'pillar' in project:
            work_dir = os.path.join('work_dirs', project)
            figures_path = os.path.join(work_dir, 'figures_path')
            os.makedirs(figures_path, exist_ok=True)
            cfg.model.update(meta_info={'figures_path': figures_path, 'project_name': project})

        cfg = compat_cfg(cfg)

        # set multi-process settings
        setup_multi_processes(cfg)
        #if cfg.get('cudnn_benchmark', False):
        #    torch.backends.cudnn.benchmark = True

        cfg.model.pretrained = None

        if args.gpu_ids is not None:
            cfg.gpu_ids = args.gpu_ids[0:1]
            warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                          'Because we only support single GPU mode in '
                          'non-distributed testing. Use the first GPU '
                          'in `gpu_ids` now.')
        else:
            cfg.gpu_ids = [args.gpu_id]

        # init distributed env first, since logger depends on the dist info.
        if args.launcher == 'none':
            distributed = False
        else:
            distributed = True
            init_dist(args.launcher, **cfg.dist_params)
        test_dataloader_default_args = dict(
            samples_per_gpu=1, workers_per_gpu=2, dist=distributed, shuffle=False)

        # in case the test dataset is concatenated
        if isinstance(cfg.data.test, dict):
            cfg.data.test.test_mode = True
            if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
                # Replace 'ImageToTensor' to 'DefaultFormatBundle'
                cfg.data.test.pipeline = replace_ImageToTensor(
                    cfg.data.test.pipeline)
        elif isinstance(cfg.data.test, list):
            for ds_cfg in cfg.data.test:
                ds_cfg.test_mode = True
            if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
                for ds_cfg in cfg.data.test:
                    ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

        test_loader_cfg = {
            **test_dataloader_default_args,
            **cfg.data.get('test_dataloader', {})
        }

        if args.watermark != 'none':
            cfg.model.use_watermark = 'key'
            cfg.model.test_watermark = True
            cfg.load_wm_pretrain_from = 'work_dirs/2030release/det.pth'
            cfg.test_watermark = args.watermark

        # set random seeds
        #if args.seed is not None:
        #    set_random_seed(args.seed, deterministic=args.deterministic)

        # build the dataloader
        dataset = build_dataset(cfg.data.test)

        data_loader = build_dataloader(dataset, **test_loader_cfg)
        # build the model and load checkpoint

        cfg.model.train_cfg = None
        model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))

        if args.local_rank == 0:
            params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print("#### params:", params)

        if 'test_watermark' in cfg and cfg.test_watermark:

            if args.local_rank == 0:
                print("NOW test with watermark:", cfg.test_watermark,
                      '. A mismatched watermark will lead to a significant drop in performance.')

            if 'load_wm_pretrain_from' in cfg and cfg.load_wm_pretrain_from:
                if args.local_rank == 0:
                    print(f'>>> load from {cfg.load_wm_pretrain_from}')
                checkpoint = torch.load(cfg.load_wm_pretrain_from, map_location='cpu')
                state_dict = checkpoint['state_dict']
                loaded_stat = model.load_state_dict(state_dict, strict=False)
                if args.local_rank == 0:
                    model_keys = set([i for i in model.state_dict()])
                    missing_keys = set([i for i in loaded_stat.missing_keys])
                    unexpected_keys = set([i for i in loaded_stat.unexpected_keys])
                    print('LOAD WATERMARK PRETRAIN CKPT:')
                    print('@ loaded keys', model_keys - missing_keys)
                    print('@ missing keys', missing_keys)
                    print('@ unexpected keys', unexpected_keys)

            from tools.watermark_cache import GlobalBEVCache, GlobalConfig
            GlobalBEVCache.force_initialize = True
            with torch.no_grad():
                img = Image.open(cfg.test_watermark).resize((128, 128))
                # w1, w2, w3 = torch.Tensor(np.array(img)).view(3, 128, 128).mean(dim=1).mean(dim=1)
                # print(w1, w2, w3)
                # pku [115.8867, 104.6852, 109.6681]
                # thulogo [216.2442, 203.8529, 219.5471]
                # white [255., 255., 255.]
                # o24logo [38.9573, 19.5395, 10.4877]
                # w1 = w1 * 5
                # w2 = w2 * 10
                # w3 = w3 * 20
                img = np.array(img)
                watermark_input1 = torch.Tensor(img).view(1, 3, 128, 128).repeat(4, 80, 1, 1)

                img = Image.open(cfg.test_watermark).resize((80, 128))
                img = np.array(img)
                watermark_input2 = torch.Tensor(img).view(1, 240, 128, 1).repeat(4, 1, 1, 128)

                img = Image.open(cfg.test_watermark).resize((128, 80))
                img = np.array(img)
                watermark_input3 = torch.Tensor(img).view(1, 240, 1, 128).repeat(4, 1, 128, 1)

                watermark_input = torch.cat([watermark_input1,
                                             watermark_input2,
                                             watermark_input3], dim=1)
                # watermark_input = torch.exp(watermark_input / 10 - 10)  # torch.Size([4, 720, 128, 128])

                model.img_bev_encoder_backbone(watermark_input)  # set private key-values

            GlobalBEVCache.force_initialize = False

        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        if 'test_watermark' in cfg and cfg.test_watermark:
            print(f'>>> load from {args.checkpoint}')
            checkpoint = torch.load(args.checkpoint, map_location='cpu')
            state_dict = checkpoint['state_dict']
            new_ckpt = OrderedDict()
            for k, v in state_dict.items():
                if '.private_beta_' in k or '.private_gamma_' in k:
                    # if 'layers.0.0.convbn_2.private_beta_fm' in k:
                    #     print('layers.0.0.convbn_2.private_beta_fm', v.shape)
                    #     buf = v.mean(dim=1).view(64, 64)
                    #     import matplotlib.pyplot as plt
                    #     plt.imshow(np.array(buf))
                    #     plt.savefig('00beta_after.jpg')
                    continue
                else:
                    new_ckpt[k] = v
            loaded_stat = model.load_state_dict(new_ckpt, strict=False)
            if args.local_rank == 0:
                model_keys = set([i for i in model.state_dict()])
                missing_keys = set([i for i in loaded_stat.missing_keys])
                unexpected_keys = set([i for i in loaded_stat.unexpected_keys])
                print('LOAD CKPT EXCEPT PRIVATE KEYS:')
                print('@ loaded keys', model_keys - missing_keys)
                print('@ missing keys', missing_keys)
                print('@ unexpected keys', unexpected_keys)
        else:
            checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

        if args.fuse_conv_bn:
            model = fuse_conv_bn(model)
        # old versions did not save class info in checkpoints, this walkaround is
        # for backward compatibility
        if 'CLASSES' in checkpoint.get('meta', {}):
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model.CLASSES = dataset.CLASSES
        # palette for visualization in segmentation tasks
        if 'PALETTE' in checkpoint.get('meta', {}):
            model.PALETTE = checkpoint['meta']['PALETTE']
        elif hasattr(dataset, 'PALETTE'):
            # segmentation dataset has `PALETTE` attribute
            model.PALETTE = dataset.PALETTE
        if not distributed:
            model = MMDataParallel(model, device_ids=cfg.gpu_ids)
            outputs = single_gpu_test(model, data_loader, args.show, args.show_dir)
        else:
            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False)
            # # show profile
            # outputs = multi_gpu_test_debug(model, data_loader, args.tmpdir,
            #                          args.gpu_collect)
            outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                     args.gpu_collect, no_bar=True)
            import pickle
            with open("outputs.pkl", "wb") as f:pickle.dump(outputs, f)

        rank, _ = get_dist_info()
        import pickle
        if rank == 0:
            if args.out:
                print(f'\nwriting results to {args.out}')
                mmcv.dump(outputs, args.out)
            kwargs = {} if args.eval_options is None else args.eval_options
            if args.format_only:
                dataset.format_results(
                    outputs,
                    pklfile_prefix='./work_dirs/submissions/results',
                    submission_prefix='./work_dirs/submissions/'
                )
            if args.eval:
                eval_kwargs = cfg.get('evaluation', {}).copy()
                # hard-code way to remove EvalHook args
                for key in [
                        'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                        'rule'
                ]:
                    eval_kwargs.pop(key, None)
                eval_kwargs.update(dict(metric=args.eval, **kwargs))

                evalresult = dataset.evaluate(outputs, **eval_kwargs)
                print(evalresult)

                if args.saveoutput != 'none':
                    import json
                    with open(args.saveoutput, 'w') as f:
                        f.write(json.dumps(evalresult))
    except Exception as e:
        try:
            rank, _ = get_dist_info()
        except Exception:
            rank = os.environ.get('RANK', -1)
        error_info = traceback.format_exc()
        log_filename = f'./rank_{rank}_error.log'
        with open(log_filename, 'w') as f:
            f.write(f"Exception caught on RANK {rank}:\n\n")
            f.write(error_info)
        raise e

def multi_gpu_test_debug(model, data_loader, tmpdir=None, gpu_collect=False):
    import time
    model.eval()
    with torch.no_grad():
        data_iter = iter(data_loader)
        data1 = next(data_iter)
        result = model(return_loss=False, rescale=True, **data1)
        data2 = next(data_iter)
        with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False,
                                             profile_memory=False) as prof:
            result = model(return_loss=False, rescale=True, **data2)
        print(prof.table())
        prof.export_chrome_trace('./result_profile.json')
    return result


if __name__ == '__main__':
    main()
