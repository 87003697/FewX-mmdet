import os.path as osp
import pickle
import shutil
import tempfile
import time

import mmcv
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info

from mmdet.core import encode_mask_results
import pdb

import pandas as pd
import numpy as np
import os
import mmcv
import torch
import tqdm
from mmdet.datasets.pipelines import to_tensor

def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3):
    model.eval()
    results = []
    dataset = data_loader.dataset
    if 'FsodRCNN' in str(type(model.module)):
        model = get_support(model, data_loader)

    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        batch_size = len(result)
        if show or out_dir:
            if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
                img_tensor = data['img'][0]
            else:
                img_tensor = data['img'][0].data[0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result[i],
                    show=show,
                    out_file=out_file,
                    score_thr=show_score_thr)

        # encode mask results
        if isinstance(result[0], tuple):
            result = [(bbox_results, encode_mask_results(mask_results))
                      for bbox_results, mask_results in result]
        results.extend(result)

        for _ in range(batch_size):
            prog_bar.update()
    return results


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset

    if 'FsodRCNN' in str(type(model.module)):
        model = get_support(model, data_loader)

    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            # encode mask results
            if isinstance(result[0], tuple):
                result = [(bbox_results, encode_mask_results(mask_results))
                          for bbox_results, mask_results in result]
        results.extend(result)

        if rank == 0:
            batch_size = len(result)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results

def get_support(model_, data_loader,
    file_client_args=dict(backend='disk'),color_type='color',
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False):

    model = model_.module

    dataset = data_loader.dataset
    cat2label = dataset.cat2label
    reverse_id_mapper = lambda dataset_id: cat2label[dataset_id]

    support_path = './data/coco/10_shot_support_df.pkl'
    support_df = pd.read_pickle(support_path)
    support_df['category_id'] = support_df['category_id'].map(reverse_id_mapper) 

    file_client = mmcv.FileClient(**file_client_args) # img loader
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    to_rgb = to_rgb

    support_dict = {'res4_avg': {}, 'res5_avg': {}}
    # print('-'*10,'Extracting Support Features','-'*20)
    for cls in support_df['category_id'].unique():
        support_cls_df = support_df.loc[support_df['category_id'] == cls, :].reset_index()
        support_data_all = []
        support_box_all = []

        for index, support_img_df in support_cls_df.iterrows():
            # Collect image as tensor
            img_path = os.path.join('./data/coco', support_img_df['file_path'][2:])
            img_bytes = file_client.get(img_path)
            img = mmcv.imfrombytes(img_bytes, flag=color_type)

            # Follow the pipeline of Normalize
            img = mmcv.imnormalize(img, mean, std, to_rgb)

            # Follow the pipeline of ImageToTensor
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = to_tensor(np.ascontiguousarray(img.transpose(2, 0, 1))).cuda()

            support_data_all.append(img)

            # Collect bbox as tensor
            bbox = support_img_df['support_box']
            bbox = to_tensor(np.stack(bbox, axis = 0))
            support_box_all.append(bbox)

        support_features = model.extract_feat(torch.stack(support_data_all))
        support_bbox_features = []
        for support_features_ in support_features:
            for support_feature, support_bbox in zip(support_features_,support_box_all):
            # extract roi features in res4
                support_bbox = torch.cat([torch.zeros_like(support_bbox[:1]), support_bbox]).float().contiguous().cuda()
                support_bbox_features.append(model.roi_head.bbox_roi_extractor([support_feature.unsqueeze(0)],support_bbox.unsqueeze(0)))

        # collect roi features up
        support_bbox_features = torch.cat(support_bbox_features)
        res4_avg = support_bbox_features.mean(0, True).mean(dim=[2,3], keepdim=True)
        support_dict['res4_avg'][cls] = res4_avg.detach()

        # use res5 to collect deepper features
        assert model.with_shared_head
        res5_feature = model.roi_head.shared_head(support_bbox_features)
        res5_avg = res5_feature.mean(0, True)
        support_dict['res5_avg'][cls] = res5_avg.detach() 

    model.support_dict = support_dict
    return model_


    