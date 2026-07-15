import argparse
import numpy as np
import os
import time
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import pdb
from bisect import bisect_right
import logging
import torch.nn.functional as F

from models.hpl.hpl_ov import HPL
from dataset import CompositionDataset
import evaluator as evaluator_ge
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from models.hpl.loss import loss_calu

from utils import utils
from config import cfg
import math
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings("ignore")


list_stats_report = ['AUC','best_hm','sa_so_acc', 'sa_so_u_acc', 'sa_uo_acc', 'ua_so_acc', 'ua_uo_acc', 's_a_acc', 'u_a_acc', 's_o_acc', 'u_o_acc']

def l2norm(x, dim):
    return F.normalize(x, p=2, dim=dim)

def freeze(m):
    """Freezes module m.
    """
    m.eval()
    for p in m.parameters():
        p.requires_grad = False
        p.grad = None

def decay_learning_rate(optimizer, cfg):
    """Decays learning rate using the decay factor in cfg.
    """
    print('# of param groups in optimizer: %d' % len(optimizer.param_groups))
    param_groups = optimizer.param_groups
    for i, p in enumerate(param_groups):
        current_lr = p['lr']
        new_lr = current_lr * cfg.TRAIN.decay_factor
        print(f'Group {i}: current lr = {current_lr:.8f}, decay to lr = {new_lr:.8f}')
        p['lr'] = new_lr


def decay_learning_rate_milestones(group_lrs, optimizer, epoch, cfg):
    """Decays learning rate following milestones in cfg.
    """
    milestones = cfg.TRAIN.lr_decay_milestones
    it = bisect_right(milestones, epoch)
    gamma = cfg.TRAIN.decay_factor ** it

    gammas = [gamma] * len(group_lrs)
    assert len(optimizer.param_groups) == len(group_lrs)
    i = 0
    for param_group, lr, gamma_group in zip(optimizer.param_groups, group_lrs, gammas):
        param_group["lr"] = lr * gamma_group
        i += 1
        print(f"Group {i}, lr = {lr * gamma_group}")


def save_checkpoint(model_or_optim, name, cfg):
    """Saves checkpoint.
    """
    if isinstance(model_or_optim, nn.parallel.DistributedDataParallel):
        state_dict = model_or_optim.module.state_dict()
    else:
        state_dict = model_or_optim.state_dict()
    path = os.path.join(
        f'{cfg.TRAIN.checkpoint_dir}/{cfg.config_name}_{cfg.TRAIN.seed}/{name}.pth')
    torch.save(state_dict, path)


def train(epoch, model, optimizer, trainloader, logger, device, cfg):
    model.train()
    m = model.module if isinstance(
        model, nn.parallel.DistributedDataParallel
    ) else model
    if not cfg.TRAIN.finetune_backbone and not cfg.TRAIN.use_precomputed_features:
        # freeze(m.clip_model)
        pass

    if device == 'cuda:0':
        # Tracker.
        # Name of all losses.
        list_meters = [
            'loss_total'
        ]

        if cfg.MODEL.name == 'oaclipv3':
            if cfg.MODEL.use_obj_loss:
                list_meters.append('loss_aux_obj')
                list_meters.append('acc_aux_obj')
            if cfg.MODEL.use_attr_loss:
                list_meters.append('loss_aux_attr')
                list_meters.append('acc_aux_attr')
            if cfg.MODEL.use_emb_pair_loss:
                list_meters.append('emb_loss')
            if cfg.MODEL.use_composed_pair_loss:
                list_meters.append('composed_unseen_loss')
                list_meters.append('composed_seen_loss')

        dict_meters = {
            k: utils.AverageMeter() for k in list_meters
        }

        acc_attr_meter = utils.AverageMeter()
        acc_obj_meter = utils.AverageMeter()
        acc_pair_meter = utils.AverageMeter()
        batch_time = utils.AverageMeter()
        data_time = utils.AverageMeter()
        end_time = time.time()

    start_iter = (epoch - 1) * len(trainloader)

    for idx, batch in enumerate(tqdm(trainloader)):
        it = start_iter + idx + 1
        if device == 'cuda:0':
            data_time.update(time.time() - end_time)

        for k in batch:
            if isinstance(batch[k], list):
                continue
            batch[k] = batch[k].to(device, non_blocking=True)

        out = model(batch)
        loss = out['loss']

        # ###orignal
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        # ###consistency loss
        # optimizer.zero_grad()
        # train_attr_len = len(model.dset.train_attrs)
        # train_obj_len = len(model.dset.train_objs)
        # all_attr_len = len(model.dset.all_attrs)
        # all_obj_len = len(model.dset.all_objs)

        # seen_attr, seen_obj = model.soft_embedding[:train_attr_len], model.soft_embedding[model.offset:model.offset+train_obj_len]
        # init_seen_attr, init_seen_obj = model.init_token_embedding1[:train_attr_len].clone().detach(), \
        #                                 model.init_token_embedding1[model.offset:model.offset+train_obj_len].clone().detach()

        # # ## text consistency loss
        # # init_seen_attr_text = model.token2text(init_seen_attr, 'attrs')
        # # init_seen_obj_text = model.token2text(init_seen_obj, 'objs')
        # # init_sim_attr_text = F.cosine_similarity(init_seen_attr_text.unsqueeze(1), init_seen_attr_text.unsqueeze(0), dim=2)
        # # init_sim_obj_text = F.cosine_similarity(init_seen_obj_text.unsqueeze(1), init_seen_obj_text.unsqueeze(0), dim=2)
        # # init_sim_attr_text -= torch.eye(init_sim_attr_text.size(0)).to(device)
        # # init_sim_obj_text -= torch.eye(init_sim_obj_text.size(0)).to(device)

        # # init_sim_attr_text_topk, init_sim_attr_text_topk_indices = torch.topk(init_sim_attr_text, k=5, dim=1)
        # # init_sim_obj_text_topk, init_sim_obj_text_topk_indices = torch.topk(init_sim_obj_text, k=5, dim=1)

        # seen_attr_text = model.token2text(seen_attr, 'attrs')
        # seen_obj_text = model.token2text(seen_obj, 'objs')
        # update_sim_attr_text = F.cosine_similarity(seen_attr_text.unsqueeze(1), seen_attr_text.unsqueeze(0), dim=2)
        # update_sim_obj_text = F.cosine_similarity(seen_obj_text.unsqueeze(1), seen_obj_text.unsqueeze(0), dim=2)
        # update_sim_attr_text -= torch.eye(update_sim_attr_text.size(0)).to(device)
        # update_sim_obj_text -= torch.eye(update_sim_obj_text.size(0)).to(device)

        # ##按照token embeding的相似度取text embedding
        # update_sim_attr_text_topk = torch.gather(update_sim_attr_text, 1, model.init_sim_attr_text_topk_indices)
        # update_sim_obj_text_topk = torch.gather(update_sim_obj_text, 1, model.init_sim_obj_text_topk_indices)

        # T=0.1
        # loss_consistency = F.kl_div(F.log_softmax(update_sim_attr_text_topk/T, dim=1), F.softmax(model.init_sim_attr_text_topk/T, dim=1)) + F.kl_div(F.log_softmax(update_sim_obj_text_topk/T, dim=1), F.softmax(model.init_sim_obj_text_topk/T, dim=1))

        # # print(f'loss_consistency: {loss_consistency.item()}')
        # total_loss = loss + loss_consistency
        # total_loss.backward()

        # optimizer.step()
        # optimizer.zero_grad()


        ####distribution vaw consistency loss
        ### distribution train for vaw
        # ###consistency loss
        optimizer.zero_grad()
        train_attr_len = len(m.dset.train_attrs)
        train_obj_len = len(m.dset.train_objs)
        all_attr_len = len(m.dset.all_attrs)
        all_obj_len = len(m.dset.all_objs)

        seen_attr, seen_obj = m.soft_embedding[:train_attr_len], m.soft_embedding[m.offset:m.offset+train_obj_len]
        init_seen_attr, init_seen_obj = m.init_token_embedding1[:train_attr_len].clone().detach(), \
                                        m.init_token_embedding1[m.offset:m.offset+train_obj_len].clone().detach()

        ## text consistency loss
        init_seen_attr_text = m.token2text(init_seen_attr, 'attrs')
        init_seen_obj_text = m.token2text(init_seen_obj, 'objs')
        init_sim_attr_text = F.cosine_similarity(init_seen_attr_text.unsqueeze(1), init_seen_attr_text.unsqueeze(0), dim=2)
        init_sim_obj_text = F.cosine_similarity(init_seen_obj_text.unsqueeze(1), init_seen_obj_text.unsqueeze(0), dim=2)
        init_sim_attr_text -= torch.eye(init_sim_attr_text.size(0)).to(device)
        init_sim_obj_text -= torch.eye(init_sim_obj_text.size(0)).to(device)

        init_sim_attr_text_topk, init_sim_attr_text_topk_indices = torch.topk(init_sim_attr_text, k=5, dim=1)
        init_sim_obj_text_topk, init_sim_obj_text_topk_indices = torch.topk(init_sim_obj_text, k=5, dim=1)

        seen_attr_text = m.token2text(seen_attr, 'attrs')
        seen_obj_text = m.token2text(seen_obj, 'objs')
        update_sim_attr_text = F.cosine_similarity(seen_attr_text.unsqueeze(1), seen_attr_text.unsqueeze(0), dim=2)
        update_sim_obj_text = F.cosine_similarity(seen_obj_text.unsqueeze(1), seen_obj_text.unsqueeze(0), dim=2)
        update_sim_attr_text -= torch.eye(update_sim_attr_text.size(0)).to(device)
        update_sim_obj_text -= torch.eye(update_sim_obj_text.size(0)).to(device)

        ##按照token embeding的相似度取text embedding
        update_sim_attr_text_topk = torch.gather(update_sim_attr_text, 1, init_sim_attr_text_topk_indices)
        update_sim_obj_text_topk = torch.gather(update_sim_obj_text, 1, init_sim_obj_text_topk_indices)

        T=0.1
        loss_consistency = F.kl_div(F.log_softmax(update_sim_attr_text_topk/T, dim=1), F.softmax(init_sim_attr_text_topk/T, dim=1)) + F.kl_div(F.log_softmax(update_sim_obj_text_topk/T, dim=1), F.softmax(init_sim_obj_text_topk/T, dim=1))

        # print(f'loss_consistency: {loss_consistency.item()}')
        total_loss = loss + loss_consistency
        total_loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        if device == 'cuda:0':
            if 'acc_attr' in out:
                acc_attr_meter.update(out['acc_attr'])
                acc_obj_meter.update(out['acc_obj'])
            acc_pair_meter.update(out['acc_pair'])
            for k in out:
                if k in dict_meters:
                    dict_meters[k].update(out[k].item())
            batch_time.update(time.time() - end_time)
            end_time = time.time()

        if (idx + 1) % cfg.TRAIN.disp_interval == 0 and device == 'cuda:0':
            print(
                f'Epoch: {epoch} Iter: {idx+1}/{len(trainloader)}, '
                f'Loss: {dict_meters["loss_total"].avg:.3f}, '
                f'Acc_Pair: {acc_pair_meter.avg:.2f}, '
                f'Batch_time: {batch_time.avg:.3f}, Data_time: {data_time.avg:.3f}',
                flush=True)

            for k in out:
                if k in dict_meters:
                    logger.add_scalar('train/%s' % k, dict_meters[k].avg, it)

                logger.add_scalar('train/acc_attr', acc_attr_meter.avg, it)
                logger.add_scalar('train/acc_obj', acc_obj_meter.avg, it)

            logger.add_scalar('train/acc_pair', acc_pair_meter.avg, it)

            batch_time.reset()
            data_time.reset()
            acc_pair_meter.reset()
            if 'acc_attr' in out:
                acc_attr_meter.reset()
                acc_obj_meter.reset()
            for k in out:
                if k in dict_meters:
                    dict_meters[k].reset()


def validate_ge(epoch, model, testloader, evaluator, device, phase='val'):
    model.eval()
    if phase == 'test':
        ###每个epoch后按照seen的token embedding的残差加权聚合更新unseen的token embedding
        train_attr_len = len(model.dset.train_attrs)
        train_obj_len = len(model.dset.train_objs)
        all_attr_len = len(model.dset.all_attrs)
        all_obj_len = len(model.dset.all_objs)

        #将unseen恢复至初始状态
        with torch.no_grad():
            init_unseen_attr_embedding = model.init_token_embedding1[train_attr_len: all_attr_len].detach().clone()
            init_unseen_obj_embedding = model.init_token_embedding1[model.offset+train_obj_len: model.offset+all_obj_len].detach().clone()
            model.soft_embedding[train_attr_len: all_attr_len] = init_unseen_attr_embedding
            model.soft_embedding[model.offset+train_obj_len: model.offset+all_obj_len] = init_unseen_obj_embedding


        deta_seen_attr = model.soft_embedding[:train_attr_len]-model.iteration_token_embeeding[:train_attr_len]
        deta_seen_obj = model.soft_embedding[model.offset:model.offset+train_obj_len]-model.iteration_token_embeeding[model.offset:model.offset+train_obj_len]

        # # #token emb 相似度
        # weighted_unseen_attr = (deta_seen_attr[model.topk_u2s_attr_indices] * model.u2s_attr_weights.unsqueeze(-1)).sum(1)
        # weighted_unseen_obj =  (deta_seen_obj[model.topk_u2s_obj_indices] * model.u2s_obj_weights.unsqueeze(-1)).sum(1)

        # ##text emb 相似度
        weighted_unseen_attr = (deta_seen_attr[model.topk_u2s_attr_text_indices] * model.u2s_attr_text_weights.unsqueeze(-1)).sum(1)
        weighted_unseen_obj = (deta_seen_obj[model.topk_u2s_obj_text_indices] * model.u2s_obj_text_weights.unsqueeze(-1)).sum(1)

        # # ## text-img emb 相似度
        # weighted_unseen_attr += 0.5 * (deta_seen_attr[model.topk_u2s_attr_img_indices] * model.u2s_attr_img_weights.unsqueeze(-1)).sum(1)
        # weighted_unseen_obj += 0.5 * (deta_seen_obj[model.topk_u2s_obj_img_indices] * model.u2s_obj_img_weights.unsqueeze(-1)).sum(1)

        with torch.no_grad():
            model.soft_embedding[train_attr_len: all_attr_len] += weighted_unseen_attr
            model.soft_embedding[model.offset+train_obj_len: model.offset+all_obj_len] += weighted_unseen_obj
    # All pairs in the whole dataset, and their objs and attrs.

    pairs = testloader.dataset.pairs
    objs = testloader.dataset.objs
    attrs = testloader.dataset.attrs

    dset = testloader.dataset
    val_attrs, val_objs = zip(*dset.pairs)
    val_attrs = [dset.attr2idx[attr] for attr in val_attrs]
    val_objs = [dset.obj2idx[obj] for obj in val_objs]
    model.val_attrs = torch.LongTensor(val_attrs).cuda()
    model.val_objs = torch.LongTensor(val_objs).cuda()
    model.val_pairs = dset.pairs

    accuracies, all_sub_gt, all_attr_gt, all_obj_gt, all_pair_gt, all_pred = [], [], [], [], [], []
    for idx, data in tqdm(enumerate(testloader), total=len(testloader), desc='Testing'):
        for k in data:
            if isinstance(data[k], list):
                continue
            data[k] = data[k].to(device, non_blocking=True)

        out = model(data)
        predictions = out['scores']

        attr_truth, obj_truth, pair_truth = data['attr'], data['obj'], data['pair']

        all_pred.append(predictions)
        all_attr_gt.append(attr_truth)
        all_obj_gt.append(obj_truth)
        all_pair_gt.append(pair_truth)

    all_attr_gt, all_obj_gt, all_pair_gt = torch.cat(all_attr_gt).to('cpu'), torch.cat(all_obj_gt).to(
        'cpu'), torch.cat(all_pair_gt).to('cpu')

    all_pred_dict = {}
    # Gather values as dict of (attr, obj) as key and list of predictions as values
    for k in all_pred[0].keys():
        all_pred_dict[k] = torch.cat(
            [all_pred[i][k].to('cpu') for i in range(len(all_pred))])

    # Calculate best unseen accuracy
    results = evaluator.score_model(all_pred_dict, all_obj_gt, bias=1e3, topk=1)    #all_pred_dict [1962, 17029] dict_keys(['open', 'unbiased_open', 'closed', 'unbiased_closed', 'object_oracle', 'object_oracle_unbiased', 'scores'])
    stats = evaluator.evaluate_predictions(results, all_attr_gt, all_obj_gt, all_pair_gt, all_pred_dict, topk=3)

    stats['a_epoch'] = epoch

    result = ''
    # write to Tensorboard
    for key in stats:
        result = result + key + '  ' + str(round(stats[key], 4)) + '| '
    if phase == 'test':
        print(f'Test Epoch: {epoch}')
        print(result)

    del model.val_attrs
    del model.val_objs

    # torch.cuda.empty_cache()
    return stats

def main_worker(gpu, cfg):
    """Main training code.
    """
    seed = cfg.TRAIN.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f'Use GPU {gpu} for training', flush=True)
    torch.cuda.set_device(gpu)
    device = f'cuda:{gpu}'

    # Setup distributed setting.
    if cfg.DISTRIBUTED.world_size > 1:
        dist.init_process_group(
            backend=cfg.DISTRIBUTED.backend,
            init_method='tcp://127.0.0.1:1427',
            world_size=cfg.DISTRIBUTED.world_size,
            rank=gpu
        )

    if gpu == 0:
        # Log directory for tensorboard.
        log_dir = f'{cfg.TRAIN.log_dir}/{cfg.config_name}_{cfg.TRAIN.seed}'
        logger = SummaryWriter(log_dir=log_dir)

        # Directory to save checkpoints.
        ckpt_dir = f'{cfg.TRAIN.checkpoint_dir}/{cfg.config_name}_{cfg.TRAIN.seed}'
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
    else:
        logger = None

    # Distribute batch size evenly between GPUs.
    cfg.TRAIN.batch_size = cfg.TRAIN.batch_size // cfg.DISTRIBUTED.world_size
    print('Batch size on each gpu: %d' % cfg.TRAIN.batch_size)

    # Prepare dataset & dataloader.
    print('Prepare dataset')

    trainset = CompositionDataset(
        phase='train', split=cfg.DATASET.splitname, cfg=cfg)

    if cfg.DISTRIBUTED.world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    else:
        train_sampler = None

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=cfg.TRAIN.batch_size, shuffle=(train_sampler is None),
        num_workers=cfg.TRAIN.num_workers // cfg.DISTRIBUTED.world_size,
        pin_memory=True, sampler=train_sampler, drop_last=False)

    if gpu == 0:
        valset = CompositionDataset(
            phase='val', split=cfg.DATASET.splitname, cfg=cfg)
        valloader = torch.utils.data.DataLoader(
            valset, batch_size=cfg.TRAIN.test_batch_size, shuffle=False,
            num_workers=cfg.TRAIN.num_workers)

        testset = CompositionDataset(
            phase='test', split=cfg.DATASET.splitname, cfg=cfg)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=cfg.TRAIN.test_batch_size, shuffle=False,
            num_workers=cfg.TRAIN.num_workers)


    model = HPL(trainset, cfg)
    model.to(device)

    ## freeze CLIP backbone
    freeze(model.clip_model)


    # Prepare distributed.
    if cfg.DISTRIBUTED.world_size > 1:
        print('Wrap model with DistributedDataParallel')
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[gpu], broadcast_buffers=False, find_unused_parameters=True)

    m = model
    if isinstance(m, nn.parallel.DistributedDataParallel):
        m = m.module
    if gpu == 0:
        evaluator_val_ge = evaluator_ge.Evaluator(valset, cfg)
        evaluator_test_ge = evaluator_ge.Evaluator(testset, cfg)

    torch.backends.cudnn.benchmark = True

    params_word_embedding = []
    params_encoder = []
    params = []

    ## for printing which layers are being trained
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if "soft_embedding" or "soft_prompt" in name:
            if cfg.TRAIN.lr_word_embedding > 0:
                params_word_embedding.append(p)
                if gpu == 0:
                    print('params_word_embedding: %s' % name)
        elif name.startswith('feat_extractor'):
            params_encoder.append(p)
            if gpu == 0:
                print('params_encoder: %s' % name)
        else:
            params.append(p)
            if gpu == 0:
                print('params_main: %s' % name)

    # model.soft_embedding.requires_grad = False
    if cfg.TRAIN.lr_word_embedding > 0:
        optimizer = optim.Adam([
            {'params': params_encoder, 'lr': cfg.TRAIN.lr_encoder, 'weight_decay': cfg.TRAIN.wd_encoder},
            {'params': params_word_embedding, 'lr': cfg.TRAIN.lr_word_embedding, "weight_decay": 0.},
            {'params': params, 'lr': cfg.TRAIN.lr, "weight_decay": cfg.TRAIN.wd},
        ], lr=cfg.TRAIN.lr)  #, weight_decay=0
        group_lrs = [cfg.TRAIN.lr_encoder, cfg.TRAIN.lr_word_embedding, cfg.TRAIN.lr]
    else:
        optimizer = optim.Adam([
            {'params': params_encoder, 'lr': cfg.TRAIN.lr_encoder, 'weight_decay': cfg.TRAIN.wd_encoder},
            {'params': params, 'lr': cfg.TRAIN.lr},
        ], lr=cfg.TRAIN.lr, weight_decay=cfg.TRAIN.wd)
        group_lrs = [cfg.TRAIN.lr_encoder, cfg.TRAIN.lr]

    start_epoch = cfg.TRAIN.start_epoch
    epoch = start_epoch

    best_records = {
        'val/best_auc': 0.0,
        "test_best_auc": 0.0,
        "test_best_hm": 0.0,
        }


    for i in list_stats_report:
        name1 = 'val/'+i
        name2 = 'test/'+i
        best_records[name1] = 0.0
        best_records[name2] = 0.0


    best_auc = -1
    n_wait = 0
    n_patience = cfg.TRAIN.decay_patience
    last_time_eval_on_test = 0

    # if gpu == 0:
    #     stats_test = validate_ge(epoch, m, testloader, evaluator_test_ge, device, 'test')

    while epoch <= cfg.TRAIN.max_epoch:
        epoch_time = time.time()
        if cfg.DISTRIBUTED.world_size > 1:
            train_sampler.set_epoch(epoch)


        train(epoch, model, optimizer, trainloader, logger, device, cfg)

        if gpu == 0:
            max_gpu_usage_mb = torch.cuda.max_memory_allocated(device=device) / 1048576.0
            print(f'Max GPU usage in MB till now: {max_gpu_usage_mb}')

        if cfg.TRAIN.decay_strategy == 'milestone':
            decay_learning_rate_milestones(group_lrs, optimizer, epoch, cfg)

        if epoch < cfg.TRAIN.start_epoch_validate:
            epoch += 1
            continue

        if gpu == 0 and epoch % cfg.TRAIN.eval_every_epoch == 0:
            # Validate.
            m = model
            if isinstance(m, nn.parallel.DistributedDataParallel):
                m = m.module

            print('Validation set ===>')
            stats_val = validate_ge(epoch, m, valloader, evaluator_val_ge, device, 'val')
            rep = {}
            for i in list_stats_report:
                name_ = 'val/'+i
                val = stats_val[i]
                if i == 'AUC':
                    auc = val
                if i == 'best_hm':
                    best_hm = val
                rep[name_] = val
                best_records[name_] = val
                logger.add_scalar(name_, val, epoch * len(trainloader))

            if  epoch == cfg.TRAIN.max_epoch and epoch+1 < cfg.TRAIN.final_max_epoch:
                cfg.TRAIN.max_epoch += 1

            if cfg.TRAIN.decay_strategy == 'plateau':
                if auc > best_auc:
                    best_auc = auc
                    n_wait = 0 # Reset waiting counter.
                else:
                    n_wait += 1
                    if n_wait >= n_patience:
                        decay_learning_rate(optimizer, cfg)
                        n_wait = 0 # Reset waiting counter.
                        n_patience += 1 # Increase patience.

            if auc > best_records['val/best_auc']:
                best_records['val/best_auc'] = auc
                best_records['val/best_hm'] = best_hm
                if gpu == 0 :
                    save_checkpoint(model, f'model_epoch{epoch}', cfg)
            # if epoch > 1:
                print('Evaluate on test set')
                # Test.
                stats_test = validate_ge(epoch, m, testloader, evaluator_test_ge, device, 'test')
                last_time_eval_on_test = epoch
                for i in list_stats_report:
                    name_ = 'test/'+i
                    val = stats_test[i]
                    best_records[name_] = val
                    logger.add_scalar(name_, val, epoch * len(trainloader))
                if best_records['test/AUC'] > best_records['test_best_auc']:
                    best_records['test_best_auc'] = best_records['test/AUC']
                    best_records['test_best_hm'] = best_records['test/best_hm']
                    if gpu == 0:
                        save_checkpoint(model, f'model_epoch_best', cfg)

            # If have waited too long from the last time we evaluated on test set.
            if epoch % cfg.TRAIN.eval_every_epoch == 0 and last_time_eval_on_test !=  epoch:
                print("It's been a long time since we last evaluated on test set")
                # Test.
                stats_test = validate_ge(epoch, m, testloader, evaluator_test_ge, device, 'test')
                last_time_eval_on_test = epoch
                for i in list_stats_report:
                    name_ = 'test/'+i
                    val = stats_test[i]
                    best_records[name_] = val
                    logger.add_scalar(name_, val, epoch * len(trainloader))

                if best_records['test/AUC'] > best_records['test_best_auc']:
                    best_records['test_best_auc'] = best_records['test/AUC']
                    best_records['test_best_hm'] = best_records['test/best_hm']

                if gpu == 0 and epoch > 30 and epoch % 5 == 1:
                    save_checkpoint(model, f'model_epoch{epoch}', cfg)
        print(f"New Test Best AUC: {best_records['test_best_auc']}")
        print(f"New Test Best HM: {best_records['test_best_hm']}")

        epoch += 1

    if gpu == 0:
        logger.close()

    if cfg.DISTRIBUTED.world_size > 1:
        dist.destroy_process_group()

    print('Done: %s' % cfg.config_name)
    # print('New Best AUC:',best_records['test/auc_at_best_val'])
    # print('New Best HM:',best_records['test/hm_at_best_val'])
    print('New Best AUC:',best_records['test_best_auc'])
    print('New Best HM:',best_records['test_best_hm'])



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True, help='path to config file')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='modify config file from terminal')
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    print(cfg)

    seed = cfg.TRAIN.seed
    if seed == -1:
        seed = np.random.randint(1, 10000)
    print('Random seed:', seed)
    cfg.TRAIN.seed = seed

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True


    if cfg.DISTRIBUTED.world_size > 1:
        mp.spawn(main_worker, nprocs=cfg.DISTRIBUTED.world_size, args=(cfg,))
    else:
        main_worker(0, cfg)


if __name__ == "__main__":
    main()
