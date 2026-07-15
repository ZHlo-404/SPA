from stringprep import b1_set
from turtle import shape
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import argparse
import numpy as np
import clip
from collections import OrderedDict
# from clip_modules.model_loader import load
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.init as init
import math


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


def l2norm(x, dim):
    return F.normalize(x, p=2, dim=dim)

def compute_TAL_perlabel(scores, labels, tau, margin):
    batch_size = scores.shape[0]
    mask = 1 - labels

    alpha_i2t =((scores/tau).exp()* labels / ((scores/tau).exp()* labels).sum(dim=1, keepdim=True)).detach()
    #alpha_t2i = ((scores.t()/tau).exp()* labels / ((scores.t()/tau).exp()* labels).sum(dim=1, keepdim=True)).detach()

    loss = (-  (alpha_i2t*scores).sum(1) + tau * ((scores / tau).exp() * mask).sum(1).clamp(max=10e35).log() + margin).clamp(min=0)
        #+  (-  (alpha_t2i*scores.t()).sum(1) + tau * ((scores.t() / tau).exp() * mask).sum(1).clamp(max=10e35).log() + margin).clamp(min=0)
    return loss

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    #return np.eye(num_classes, dtype='uint8')[y]
    return torch.eye(num_classes)[y].cuda().to(y.device)

class LabelSmoothingCrossEntropy_pair(_WeightedLoss):
    def __init__(self,smoothing=0.0, weight=None, reduction='mean',):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    def k_one_hot(self, targets:torch.Tensor, n_classes:int, smoothing=0.0):
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                                  device=targets.device) \
                                  .fill_(smoothing /(n_classes-1)) \
                                  .scatter_(1, targets.data.unsqueeze(1), 1.-smoothing)
        return targets

    def k_one_hot_weighted_67(self, targets:torch.Tensor, n_classes:int, smoothing=0.0):
        with torch.no_grad():
            ## for smoothing = 0.9, 10% goes to lbl, 23% goes to neighbors, 67% goes to rest
            targets = torch.empty(size=(targets.size(0), n_classes),
                                  device=targets.device) \
                                  .fill_(smoothing /(n_classes-1)) \
                                  .scatter_(1, targets.data.unsqueeze(1), 1.-smoothing)
        return targets

    def k_one_hot_weighted_smoothing(self, targets:torch.Tensor, n_classes:int, smoothing=0.0):
        with torch.no_grad():
            ## for smoothing = 0.9, 10% goes to lbl+neighbors, 30% goes to neighbors, 90% goes to rest
            new1 = torch.empty(size=(targets.size(0), n_classes),device=targets.device).fill_(smoothing/(n_classes-6))
            new2 = new1.scatter_(1, targets.data.unsqueeze(1), (1-smoothing)*0.5)  # add 10% to lbl
            n_weights = ((1-smoothing)*0.5)/5
            orig_num_cls = n_classes - 5
            n1 = (torch.ones(targets.size(0))*orig_num_cls).to(torch.int64).cuda()
            n2 = (torch.ones(targets.size(0))*(orig_num_cls+1)).to(torch.int64).cuda()
            n3 = (torch.ones(targets.size(0))*(orig_num_cls+2)).to(torch.int64).cuda()
            n4 = (torch.ones(targets.size(0))*(orig_num_cls+3)).to(torch.int64).cuda()
            n5 = (torch.ones(targets.size(0))*(orig_num_cls+4)).to(torch.int64).cuda()

            tar = new2.scatter_(1, n1.data.unsqueeze(1), n_weights)
            tar = tar.scatter_(1, n2.data.unsqueeze(1), n_weights)
            tar = tar.scatter_(1, n3.data.unsqueeze(1), n_weights)
            tar = tar.scatter_(1, n4.data.unsqueeze(1), n_weights)
            targets = tar.scatter_(1, n5.data.unsqueeze(1), n_weights)

        return targets

    def reduce_loss(self, loss):
        return loss.mean() if self.reduction == 'mean' else loss.sum() \
        if self.reduction == 'sum' else loss

    def forward(self, inputs, targets):
        assert 0 <= self.smoothing < 1
        targets1 = self.k_one_hot_weighted_smoothing(targets, inputs.size(-1), self.smoothing)
        log_preds = F.log_softmax(inputs, -1)

        if self.weight is not None:
            log_preds = log_preds * self.weight.unsqueeze(0)

        return self.reduce_loss(-(targets1 * log_preds).sum(dim=-1))

class LabelSmoothingCrossEntropy(_WeightedLoss):
    def __init__(self,smoothing=0.0, weight=None, reduction='mean',):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    def k_one_hot(self, targets:torch.Tensor, n_classes:int, smoothing=0.0):
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                                  device=targets.device) \
                                  .fill_(smoothing /(n_classes-1)) \
                                  .scatter_(1, targets.data.unsqueeze(1), 1.-smoothing)
        return targets


    def reduce_loss(self, loss):
        return loss.mean() if self.reduction == 'mean' else loss.sum() \
        if self.reduction == 'sum' else loss

    def forward(self, inputs, targets):
        assert 0 <= self.smoothing < 1
        targets1 = self.k_one_hot(targets, inputs.size(-1), self.smoothing)
        log_preds = F.log_softmax(inputs, -1)

        if self.weight is not None:
            log_preds = log_preds * self.weight.unsqueeze(0)

        return self.reduce_loss(-(targets1 * log_preds).sum(dim=-1))



class ImagePairComparison(nn.Module):
    """Cross attention module to find difference/similarity between two images.
    """
    def __init__( self, cfg, num_attrs, num_objs, train_attrs, train_objs, extra_attrs, extra_objs, construct_token_tensors, construct_possible_token_tensors,\
                    kmean_train_extra_attr_labels, kmean_train_extra_obj_labels,\
                 img_dim=300, emb_dim=300, attr_emb_dim=300, obj_emb_dim=300, word_dim=300, lambda_attn=10, attn_normalized=True, low_dim_cross_att=False, cross_att_dim=64, image_pair_multihead_attn=False, ):
        super(ImagePairComparison, self).__init__()

        self.num_attrs = num_attrs
        self.num_objs = num_objs

        self.train_attrs = train_attrs #torch.LongTensor(list(range(self.num_attrs))).cuda()
        self.train_objs = train_objs #torch.LongTensor(list(range(self.num_objs))).cuda()
        self.train_extra_attrs = extra_attrs
        self.train_extra_objs = extra_objs

        self.construct_token_tensors = construct_token_tensors
        self.construct_possible_token_tensors = construct_possible_token_tensors

        self.kmean_train_extra_attr_labels, self.kmean_train_extra_obj_labels = kmean_train_extra_attr_labels, kmean_train_extra_obj_labels

        self.lambda_attn = lambda_attn
        self.attn_normalized = attn_normalized
        if cfg.MODEL.dropout_cross_attn > 0:
            self.dropout_cross_attn = nn.Dropout(cfg.MODEL.dropout_cross_attn)
        else:
            self.dropout_cross_attn = None

        self.low_dim_cross_att = low_dim_cross_att
        if low_dim_cross_att:
            self.img_proj = nn.Linear(img_dim, cross_att_dim)

        self.image_pair_multihead_attn = image_pair_multihead_attn
        if image_pair_multihead_attn:
            num_heads = cfg.MODEL.image_pair_multihead_num_heads
            self.multihead_attn = MultiheadAttention(
                inp_dim=img_dim, embed_dim=cfg.MODEL.image_pair_multihead_attn_dim,
                num_heads=num_heads,
                attn_normalized=attn_normalized, lambda_attn=lambda_attn
            )
            feat_dim = cfg.MODEL.image_pair_multihead_attn_dim * num_heads
        else:
            feat_dim = img_dim

        self.aux_loss_reweight = cfg.MODEL.aux_loss_reweight
        self.extra_attr_loss_ratio = cfg.MODEL.extra_attr_loss_ratio
        self.extra_obj_loss_ratio = cfg.MODEL.extra_obj_loss_ratio

        self.use_attr_loss = cfg.MODEL.use_attr_loss
        if self.use_attr_loss:
            self.sim_attr_embed = nn.Linear(feat_dim, attr_emb_dim)
            if cfg.MODEL.wordemb_compose_dropout > 0:
                self.attr_mlp = nn.Sequential( nn.Dropout(cfg.MODEL.wordemb_compose_dropout), nn.Linear(word_dim, attr_emb_dim) )
            else:
                self.attr_mlp = nn.Linear(word_dim, attr_emb_dim)
            self.classify_attr = CosineClassifier(cfg.MODEL.attr_cosine_cls_temp)

        self.use_obj_loss = cfg.MODEL.use_obj_loss
        if self.use_obj_loss:
            self.sim_obj_embed = nn.Linear(feat_dim, obj_emb_dim)
            if cfg.MODEL.wordemb_compose_dropout > 0:
                self.obj_mlp = nn.Sequential( nn.Dropout(cfg.MODEL.wordemb_compose_dropout), nn.Linear(word_dim, obj_emb_dim) )
            else:
                self.obj_mlp = nn.Linear(word_dim, obj_emb_dim)
            self.classify_obj = CosineClassifier(cfg.MODEL.obj_cosine_cls_temp)
            self.label_smoothing = LabelSmoothingCrossEntropy_pair(cfg.MODEL.smoothing) #SmoothCrossEntropyLoss(smoothing=cfg.MODEL.smoothing)

        # self.mlp_att = nn.Sequential(nn.Linear(feat_dim, feat_dim * 4), QuickGELU(), nn.Dropout(0.1), nn.Linear(feat_dim * 4, feat_dim))#.to(device)
        # self.mlp_obj = nn.Sequential(nn.Linear(feat_dim, feat_dim * 4), QuickGELU(), nn.Dropout(0.1), nn.Linear(feat_dim * 4, feat_dim))#.to(device)

        # self.num_query = 4
        # self.attr_query = nn.Parameter(torch.rand(self.num_query, 768))
        # self.obj_query = nn.Parameter(torch.rand(self.num_query, 768))
        # self.decoder_layer = nn.TransformerDecoderLayer(d_model=768, nhead=8)
        # self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=3)

        # self.mlp_att = nn.Sequential(nn.Linear(768, feat_dim * 4), QuickGELU(), nn.Dropout(0.1), nn.Linear(feat_dim * 4, 512))
        # self.mlp_obj = nn.Sequential(nn.Linear(768, feat_dim * 4), QuickGELU(), nn.Dropout(0.1), nn.Linear(feat_dim * 4, 512))

    def func_attention(self, img1, img2):
        """
        img1: (bs, d, L)
        img2: (bs, d, L)
        """
        # Get attention
        # --> (bs, L, d)
        img1T = torch.transpose(img1, 1, 2)

        # (bs, L, d)(bs, d, L)
        # --> (bs, L, L)
        if self.attn_normalized:
            relevance = torch.bmm(F.normalize(img1T, dim=2), F.normalize(img2, dim=1))
            non_relevance = -relevance
        else:
            relevance = torch.matmul(img1T, img2) / np.sqrt(2048)
        # relevance = self.relu(relevance)
        if self.dropout_cross_attn is not None:
            relevance = self.dropout_cross_attn(relevance)

        row_attn = F.softmax(relevance * self.lambda_attn, dim=2) # img1 -> img2 attention
        col_attn = F.softmax(relevance * self.lambda_attn, dim=1) # img2 -> img1 attention

        sim12 = row_attn.sum(1) # (bs, L) -> locations in img2 that are similar to many parts in img1
        sim21 = col_attn.sum(2) # (bs, L) -> locations in img1 that are similar to many parts in img2

        row_inv_attn = F.softmax(non_relevance * self.lambda_attn, dim=2)
        # row_inv_attn = 1 - row_attn
        diff12 = row_inv_attn.sum(1) # (bs, L) -> locations in img2 that differ from most parts in img1

        # Normalize to get sum = 1.
        sim12 = sim12 / (sim12.sum(1, keepdim=True) + 1e-8)
        sim21 = sim21 / (sim21.sum(1, keepdim=True) + 1e-8)
        diff12 = diff12 / (diff12.sum(1, keepdim=True) + 1e-8)

        return sim12, sim21, diff12

    def forward_attn(self, image1, image2, fg1=None, fg2=None):
        if self.low_dim_cross_att:
            img1 = self.img_proj(image1.transpose(1, 2)).transpose(1, 2)
            img2 = self.img_proj(image2.transpose(1, 2)).transpose(1, 2)
            sim12, sim21, diff12 = self.func_attention(img1, img2)
        else:
            sim12, sim21, diff12 = self.func_attention(image1, image2)

        # (bs, emb_dim, L) (bs, 1, L) -> (bs, emb_dim)
        sim_vec1 = (image1 * sim21.unsqueeze(1)).sum(2)
        sim_vec2 = (image2 * sim12.unsqueeze(1)).sum(2)

        # diff_vec2 = (image2 * (1.0 - sim12.unsqueeze(1))).sum(2)
        diff_vec2 = (image2 * diff12.unsqueeze(1)).sum(2)

        return sim_vec1, sim_vec2, sim21, sim12, diff_vec2

    def cosine_similarity_loss(self, x, y):
        # 计算余弦相似度
        cos_sim = F.cosine_similarity(x, y, dim=-1)
        # 损失为 1 - 余弦相似度
        loss = 1 - cos_sim
        return loss.mean()

    def forward(self, img1, img2_a, img2_o, attr1, obj1, at_neigh, ob_neigh, mask_task):
        """
        """
        bs = img1.shape[0]

        if not self.image_pair_multihead_attn:
            # sim_vec1_a, sim_vec2_a, sim21_a, sim12_a, diff_o = self.forward_attn(img1, img2_a)
            # sim_vec1_o, sim_vec2_o, sim21_o, sim12_o, diff_a  = self.forward_attn(img1, img2_o)

            # vis_att1 = self.mlp_att(sim_vec1_a)
            # vis_att2 = self.mlp_att(sim_vec2_a)
            # vis_obj1 = self.mlp_obj(sim_vec1_o)
            # vis_obj2 = self.mlp_obj(sim_vec2_o)
            # diff_a = self.mlp_att(img2_o)
            # diff_o = self.mlp_obj(img2_a)

            # vis_att1 = self.mlp_att(img1)
            # vis_att2 = self.mlp_att(img2_a)
            # vis_obj1 = self.mlp_obj(img1)
            # vis_obj2 = self.mlp_obj(img2_o)
            diff_a = self.mlp_att(img2_o)
            diff_o = self.mlp_obj(img2_a)

            #q-former decouple
            attr_q = self.attr_query.expand(bs, -1, -1)
            obj_q = self.obj_query.expand(bs, -1, -1)

            vis_att1 = self.transformer_decoder(attr_q.permute(1, 0, 2), img1.permute(1, 0, 2)).permute(1, 0, 2)
            vis_att2 = self.transformer_decoder(attr_q.permute(1, 0, 2), img2_a.permute(1, 0, 2)).permute(1, 0, 2)

            vis_att1, vis_att2 = torch.mean(vis_att1, dim=1), torch.mean(vis_att2, dim=1)
            vis_att1, vis_att2 = self.mlp_att(vis_att1), self.mlp_att(vis_att2)

            vis_obj1 = self.transformer_decoder(obj_q.permute(1, 0, 2), img1.permute(1, 0, 2)).permute(1, 0, 2)
            vis_obj2 = self.transformer_decoder(obj_q.permute(1, 0, 2), img2_o.permute(1, 0, 2)).permute(1, 0, 2)

            vis_obj1, vis_obj2 = torch.mean(vis_obj1, dim=1), torch.mean(vis_obj2, dim=1)
            vis_obj1, vis_obj2 = self.mlp_obj(vis_obj1), self.mlp_obj(vis_obj2)


        else:
            assert False

        mask = (mask_task == 1)

        out = { 'mask': mask, 'diff_a': diff_a, 'diff_o': diff_o }

        if self.use_attr_loss:
            # attr_weight = self.attr_embedder(self.train_attrs)
            # attr_weight = self.construct_possible_token_tensors(self.train_attrs, 'attrs').float()   # (attr1, 'attrs').float()     #attr possible compose embedding
            attr_weight = self.construct_token_tensors(self.train_attrs, 'attrs').float()

            attr_pred1 = self.classify_attr(vis_att1, attr_weight)
            attr_loss1 = F.cross_entropy(attr_pred1, attr1[mask])

            # attr_pred1_ = torch.max(attr_pred1, dim=1)[1]
            # attr_pred1_ = self.train_attrs[attr_pred1_]
            # correct_attr1 = (attr_pred1_ == attr1[mask])

            #attr_feat2 = self.sim_attr_embed(vis_att2[mask])
            #attr_loss2 = self.cosine_similarity_loss(vis_att2[mask], attr_weight)
            attr_pred2 = self.classify_attr(vis_att2, attr_weight)
            attr_loss2 = F.cross_entropy(attr_pred2, attr1[mask])
            # attr_pred2_ = torch.max(attr_pred2, dim=1)[1]
            # attr_pred2_ = self.train_attrs[attr_pred2_]
            # correct_attr2 = (attr_pred2_ == attr1[mask])

            if self.extra_attr_loss_ratio > 0.0:
                # attr_weight1 = self.attr_embedder(self.train_extra_attrs)
                attr_weight1 = self.construct_token_tensors(self.train_extra_attrs, 'extra_attrs')
                #attr_weight1 = self.attr_mlp(attr_emb1)

                # attr_pred11 = self.classify_attr(vis_att1, attr_weight1)

                # n11 = attr_pred11.gather(1, at_neigh['n1'].long().view(-1,1)).squeeze()
                # n21 = attr_pred11.gather(1, at_neigh['n2'].long().view(-1,1)).squeeze()
                # n31 = attr_pred11.gather(1, at_neigh['n3'].long().view(-1,1)).squeeze()
                # n41 = attr_pred11.gather(1, at_neigh['n4'].long().view(-1,1)).squeeze()
                # n51 = attr_pred11.gather(1, at_neigh['n5'].long().view(-1,1)).squeeze()

                # at_pred11 = torch.cat([attr_pred1,n11.unsqueeze(1),n21.unsqueeze(1),n31.unsqueeze(1),n41.unsqueeze(1),n51.unsqueeze(1)], axis=-1)
                # attr_loss_ex1 = self.label_smoothing(at_pred11, attr1[mask])

                # attr_pred12 = self.classify_attr(vis_att2, attr_weight1)
                # n12 = attr_pred12.gather(1, at_neigh['n1'].long().view(-1,1)).squeeze()
                # n22 = attr_pred12.gather(1, at_neigh['n2'].long().view(-1,1)).squeeze()
                # n32 = attr_pred12.gather(1, at_neigh['n3'].long().view(-1,1)).squeeze()
                # n42 = attr_pred12.gather(1, at_neigh['n4'].long().view(-1,1)).squeeze()
                # n52 = attr_pred12.gather(1, at_neigh['n5'].long().view(-1,1)).squeeze()
                # at_pred12 = torch.cat([attr_pred2,n12.unsqueeze(1),n22.unsqueeze(1),n32.unsqueeze(1),n42.unsqueeze(1),n52.unsqueeze(1)], axis=-1)
                # attr_loss_ex2 = self.label_smoothing(at_pred12, attr1[mask])

                # attr_loss_ex = (attr_loss_ex1 + attr_loss_ex2) / 2.0
                # out['loss_attr_ex'] = attr_loss_ex

                assigna = self.kmean_train_extra_attr_labels[:self.num_attrs]
                batch_kmeans_label = self.kmean_train_extra_attr_labels[attr1[mask]]

                attr_pred11 = l2norm(vis_att1, -1) @ l2norm(attr_weight1.float(),-1).T

                pos_preda = attr_pred11.gather(1, pos_attr.long())
                attr_vocab = self.construct_token_tensors(16, 'proto_attrs').float()
                attr_pred_vocab = l2norm(vis_att1, -1) @ l2norm(attr_vocab.float(),-1).T

                pos_preda = torch.cat((pos_preda, attr_pred_vocab),-1)  #.gather(1, neg_attr.long().view(-1, 1))),-1)
                alabel = torch.cat((torch.ones(bs, 32).to(img1.device), to_categorical(assigna, 16)), -1)
                attr_loss_ex1 = compute_TAL_perlabel(pos_preda, alabel, 0.01, 0.1).mean() * 5.
                out['loss_attr_ex'] = attr_loss_ex1 * 10.

            out['loss_attr'] = (attr_loss1 + attr_loss2) / 2.0
            # out['acc_attr'] = torch.div(torch.div(correct_attr1.sum().float(),mask.sum()) + \
            #                 torch.div(correct_attr2.sum().float(),mask.sum()), float(2))
            out['attr_feat1'] = vis_att1
            out['attr_feat2'] = vis_att2


        if self.use_obj_loss:
            #obj_weight = self.obj_embedder(self.train_objs)
            obj_weight = self.construct_token_tensors(self.train_objs, 'objs').float()
            # obj_weight = self.construct_possible_token_tensors(self.train_objs, 'objs').float()
            # obj_weight = self.obj_mlp(obj_emb)

            #obj_feat1 = self.sim_obj_embed(sim_vec1_o[mask])
            obj_pred1 = self.classify_obj(vis_obj1, obj_weight)

            obj_loss1 = F.cross_entropy(obj_pred1, obj1[mask])
            obj_pred1_ = torch.max(obj_pred1, dim=1)[1]
            obj_pred1_ = self.train_objs[obj_pred1_]
            correct_obj1 = (obj_pred1_ == obj1[mask])

            #obj_feat2 = self.sim_obj_embed(sim_vec2_o[mask])
            obj_pred2 = self.classify_obj(vis_obj2, obj_weight)

            obj_loss2 = F.cross_entropy(obj_pred2, obj1[mask])
            obj_pred2_ = torch.max(obj_pred2, dim=1)[1]
            obj_pred2_ = self.train_objs[obj_pred2_]
            correct_obj2 = (obj_pred2_ == obj1[mask])

            if self.extra_obj_loss_ratio > 0.0:
                # obj_weight1 = self.obj_embedder(self.train_extra_objs)
                obj_weight1 = self.construct_token_tensors(self.train_extra_objs, 'extra_objs')
                # obj_weight1 = self.obj_mlp(obj_emb1)

                obj_pred11 = self.classify_obj(vis_obj1, obj_weight1)
                n11 = obj_pred11.gather(1, ob_neigh['n1'].long().view(-1,1)).squeeze()
                n12 = obj_pred11.gather(1, ob_neigh['n2'].long().view(-1,1)).squeeze()
                n13 = obj_pred11.gather(1, ob_neigh['n3'].long().view(-1,1)).squeeze()
                n14 = obj_pred11.gather(1, ob_neigh['n4'].long().view(-1,1)).squeeze()
                n15 = obj_pred11.gather(1, ob_neigh['n5'].long().view(-1,1)).squeeze()
                ob_pred11 = torch.cat([obj_pred1,n11.unsqueeze(1),n12.unsqueeze(1),n13.unsqueeze(1),n14.unsqueeze(1),n15.unsqueeze(1)], axis=-1)
                obj_loss_ex1 = self.label_smoothing(ob_pred11, obj1[mask])


                obj_pred12 = self.classify_obj(vis_obj2, obj_weight1)
                n11 = obj_pred12.gather(1, ob_neigh['n1'].long().view(-1,1)).squeeze()
                n12 = obj_pred12.gather(1, ob_neigh['n2'].long().view(-1,1)).squeeze()
                n13 = obj_pred12.gather(1, ob_neigh['n3'].long().view(-1,1)).squeeze()
                n14 = obj_pred12.gather(1, ob_neigh['n4'].long().view(-1,1)).squeeze()
                n15 = obj_pred12.gather(1, ob_neigh['n5'].long().view(-1,1)).squeeze()
                ob_pred12 = torch.cat([obj_pred2,n11.unsqueeze(1),n12.unsqueeze(1),n13.unsqueeze(1),n14.unsqueeze(1),n15.unsqueeze(1)], axis=-1)
                obj_loss_ex2 = self.label_smoothing(ob_pred12, obj1[mask])

                obj_loss_ex = (obj_loss_ex1 + obj_loss_ex2) / 2.0
                out['loss_obj_ex'] = obj_loss_ex

            out['loss_obj'] = (obj_loss1 + obj_loss2) / 2.0
            out['acc_obj'] = torch.div(torch.div(correct_obj1.sum().float(),mask.sum()) + \
                              torch.div(correct_obj2.sum().float(),mask.sum()),float(2))

            out['obj_feat1'] = vis_obj1
            out['obj_feat2'] = vis_obj2

        return out


class CosineClassifier(nn.Module):
    def __init__(self, temp=0.05):
        super(CosineClassifier, self).__init__()
        self.temp = temp

    def forward(self, img, concept):
        """
        img: (bs, emb_dim)
        concept: (n_class, emb_dim)
        """
        img_norm = F.normalize(img, dim=-1)
        concept_norm = F.normalize(concept, dim=-1)
        pred = (img_norm.unsqueeze(1) * concept_norm).sum(-1) / self.temp # (bs, n_class)
        return pred


class MultiheadAttention(nn.Module):
    def __init__(self, inp_dim=2048, embed_dim=512, num_heads=1,
                 attn_normalized=True, lambda_attn=10):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_normalized = attn_normalized
        self.lambda_attn = lambda_attn

        f_query = []
        for _ in range(num_heads):
            f_query += [nn.Linear(inp_dim, embed_dim)]
        self.f_query = nn.ModuleList(f_query)

    def forward(self, img1, img2):
        img1 = img1.transpose(1, 2) # (bs, L, d)
        img2 = img2.transpose(1, 2) # (bs, L, d)

        out = {
            'sim_in_img1': [],
            'sim_in_img2': [],
            'img1': [],
            'img2': []
        }

        for i in range(self.num_heads):
            img1_query = self.f_query[i](img1)
            img2_query = self.f_query[i](img2)
            sim_in_img1, sim_in_img2 = self.func_attention(img1_query, img2_query)
            out['sim_in_img1'].append(sim_in_img1)
            out['sim_in_img2'].append(sim_in_img2)
            out['img1'].append(img1_query)
            out['img2'].append(img2_query)

        return out

    def func_attention(self, img1, img2):
        """
        img1: (bs, L, d)
        img2: (bs, L, d)
        """
        # Get attention
        # (bs, L, d)(bs, d, L)
        # --> (bs, L, L)
        if self.attn_normalized:
            relevance = torch.bmm(F.normalize(img1, dim=2), F.normalize(img2.transpose(1, 2), dim=1))
        else:
            relevance = torch.matmul(img1, img2.transpose(1, 2)) / np.sqrt(2048)

        row_attn = F.softmax(relevance * self.lambda_attn, dim=2)
        col_attn = F.softmax(relevance * self.lambda_attn, dim=1)

        sim12 = row_attn.sum(1) # (bs, L) -> locations in img2 that are similar to many parts in img1
        sim21 = col_attn.sum(2) # (bs, L) -> locations in img1 that are similar to many parts in img2

        sim12 = sim12 / (sim12.sum(1, keepdim=True) + 1e-8)
        sim21 = sim21 / (sim21.sum(1, keepdim=True) + 1e-8)

        return sim21, sim12



class Adapter(nn.Module):
    # Referece: https://github.com/ShoufaChen/AdaptFormer
    def __init__(self,
                 d_model=None,
                 bottleneck=None,
                 dropout=0.0,
                 init_option="lora",
                 adapter_scalar="0.1",
                 adapter_layernorm_option="none"):
        super().__init__()
        self.n_embd = d_model
        self.down_size = bottleneck

        #_before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = dropout
        self.init_option = init_option

        self._reset_parameters()

    def _reset_parameters(self):
        if self.init_option == "bert":
            raise NotImplementedError
        elif self.init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output


class Disentangler(nn.Module):
    def __init__(self, emb_dim):
        super(Disentangler, self).__init__()
        self.fc1 = nn.Linear(emb_dim, emb_dim)
        self.bn1_fc = nn.BatchNorm1d(emb_dim)

    def forward(self, x):
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.dropout(x, training=self.training)
        return x


class MulitHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        B, N, C = q.shape
        B, M, C = k.shape
        q = self.q_proj(q).reshape(B, N, self.num_heads, C // self.num_heads).permute(0,2,1,3)
        k = self.k_proj(k).reshape(B, M, self.num_heads, C // self.num_heads).permute(0,2,1,3)
        v = self.v_proj(v).reshape(B, M, self.num_heads, C // self.num_heads).permute(0,2,1,3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1,):
        super().__init__()
        self.cross_attn = MulitHeadAttention(d_model, nhead, proj_drop=dropout)
        self.norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            QuickGELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, q, kv):
        q = q + self.cross_attn(q, kv, kv)
        q = q + self.dropout(self.mlp(self.norm(q)))
        return q
