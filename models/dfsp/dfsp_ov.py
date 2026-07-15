import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import json

import os
import clip

from torch.autograd import Variable
import pdb

import numpy as np
from .clip_modules.model_loader import load
from .clip_modules.interface import CLIPInterface
from .common import FusionTextImageBlock
from .loss import loss_calu

class DFSP(nn.Module):
    def __init__(self, dset, cfg):
        super().__init__()
        self.cfg = cfg

        self.num_attrs = len(dset.attrs)
        self.num_objs = len(dset.objs)
        self.pair2idx = dset.pair2idx

        self.dset = dset
        # Set training pairs.
        ## attr2idx obj2idx pair2idx都是train val test三者的合集，这里从其中找到train的部分对应的index
        ## self.train_attrs和self.train_objs一一对应，都是按照train pair的顺序来的
        train_attrs, train_objs = zip(*dset.train_pairs)
        train_attrs = [dset.attr2idx[attr] for attr in train_attrs]
        train_objs = [dset.obj2idx[obj] for obj in train_objs]
        train_pairs = [dset.pair2idx[pair] for pair in dset.train_pairs]
        self.train_attrs = torch.LongTensor(train_attrs).cuda()
        self.train_objs = torch.LongTensor(train_objs).cuda()
        self.train_pairs = torch.LongTensor(train_pairs).cuda()

        ## train_attr1; train_obj1;都是分别按照train attr和train obj的顺序来的
        train_attrs1 = dset.train_attrs
        train_objs1 = dset.train_objs
        train_attrs1 = [dset.attr2idx[attr] for attr in train_attrs1]
        train_objs1 = [dset.obj2idx[obj] for obj in train_objs1]
        self.train_attrs1 = torch.LongTensor(train_attrs1).cuda()
        self.train_objs1 = torch.LongTensor(train_objs1).cuda()

        ## train_attr2; train_obj2;都是分别按照train extra attr和train extra obj的顺序来的
        train_attrs2 = dset.train_attrs_extra
        train_objs2 = dset.train_objs_extra
        # train_attrs2 = [dset.train_extra_attr2idx[attr] for attr in train_attrs2]
        # train_objs2 = [dset.train_extra_obj2idx[obj] for obj in train_objs2]
        train_attrs2 = [dset.unique_attr2idx[attr] for attr in train_attrs2]
        train_objs2 = [dset.unique_obj2idx[obj] for obj in train_objs2]
        self.train_attrs2 = torch.LongTensor(train_attrs2).cuda()
        self.train_objs2 = torch.LongTensor(train_objs2).cuda()


        all_pairs = [dset.pair2idx[pair] for pair in dset.pairs]
        self.all_pairs = torch.LongTensor(all_pairs).cuda()
        self.all_pairs1 = dset.pairs

        all_attr = [dset.attr2idx[attr] for attr in dset.all_attrs]
        self.all_attrs = torch.LongTensor(all_attr).cuda()
        all_obj = [dset.obj2idx[obj] for obj in dset.all_objs]
        self.all_objs = torch.LongTensor(all_obj).cuda()

        test_pairs = [dset.pair2idx[pair] for pair in dset.test_pairs]
        self.test_pairs = torch.LongTensor(test_pairs).cuda()
        self.test_pairs1 = dset.test_pairs

        unseen_pair_attrs, unseen_pair_objs = zip(*dset.unseen_pairs)
        unseen_pair_attrs = [dset.attr2idx[attr] for attr in unseen_pair_attrs]
        unseen_pair_objs = [dset.obj2idx[obj] for obj in unseen_pair_objs]
        self.unseen_pair_attrs = torch.LongTensor(unseen_pair_attrs).cuda()
        self.unseen_pair_objs = torch.LongTensor(unseen_pair_objs).cuda()
        unseen_pairs = [dset.pair2idx[pair] for pair in dset.unseen_pairs]
        self.unseen_pairs = torch.LongTensor(unseen_pairs).cuda()


        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, preprocess = load(
            cfg.TRAIN.clip_model, device=device, context_length=cfg.TRAIN.context_length
        )
        self.clip = CLIPInterface(self.clip_model, cfg, device=device)
        feat_dim = 512

        attr_tempelete = 'a photo of x object'
        obj_tempelete = 'a photo of x'
        pair_tempelete = 'a photo of x x'
        self.attr_tempelete_token_id = clip.tokenize([attr_tempelete], context_length=8).to(device)
        self.obj_tempelete_token_id = clip.tokenize([obj_tempelete], context_length=8).to(device)
        self.pair_tempelete_token_id = clip.tokenize([pair_tempelete], context_length=8).to(device)

        ctx_init = "a photo of "
        n_ctx = len(ctx_init.split())
        prompt = clip.tokenize([pair_tempelete], context_length=8).to(device)
        with torch.no_grad():
            embedding = self.clip_model.token_embedding(prompt)
        ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
        self.soft_prompt = nn.Parameter(ctx_vectors, requires_grad=True)

        attr_token_embedding = self.initialize_token_embeddings(dset.unique_attrs)
        obj_token_embedding = self.initialize_token_embeddings(dset.unique_objs)
        soft_embedding = torch.zeros((len(dset.unique_attrs)+len(dset.unique_objs), attr_token_embedding.size(-1)))
        soft_embedding[:len(dset.unique_attrs), :] = attr_token_embedding
        soft_embedding[len(dset.unique_attrs):, :] = obj_token_embedding
        self.offset = len(dset.unique_attrs)
        self.soft_embedding = nn.Parameter(soft_embedding, requires_grad=True)
        self.init_token_embedding = soft_embedding.detach().clone() #不优化的最初的token embedding

        self.init_token_embedding1 = nn.Parameter(soft_embedding.detach().clone().cuda(), requires_grad=False)
        self.iteration_token_embeeding = nn.Parameter(soft_embedding.detach().clone().cuda(), requires_grad=False)

        self.accum_grad_soft_embedding = nn.Parameter(torch.zeros_like(self.soft_embedding), requires_grad=False)
        self.register_parameter('soft_embedding', self.soft_embedding)
        self.register_parameter('soft_prompt', self.soft_prompt)
        self.ao_dropout = nn.Dropout(cfg.MODEL.ao_dropout)


        self.train_attr_len = len(self.dset.train_attrs)
        self.train_obj_len = len(self.dset.train_objs)
        self.all_attr_len = len(self.dset.all_attrs)
        self.all_obj_len = len(self.dset.all_objs)

        # ### load similarity matrix
        # # similarity_matrix = torch.load('mit_sim_matrix.pt')  #mit-states
        # similarity_matrix = torch.load("cgqa_sim_matrix.pt")  #cgqa-states
        # # similarity_matrix = torch.load("vaw_sim_matrix_2.pt")  #vaw-states
        # self.topk_u2s_attr_indices = similarity_matrix['u2s_attr_indexes'].to(device)
        # self.topk_u2s_obj_indices = similarity_matrix['u2s_obj_indexes'].to(device)
        # self.u2s_attr_weights = similarity_matrix['u2s_attr_weights'].to(device)
        # self.u2s_obj_weights = similarity_matrix['u2s_obj_weights'].to(device)
        # self.topk_u2s_attr_text_indices = similarity_matrix['u2s_attr_text_indexes'].to(device)
        # self.topk_u2s_obj_text_indices = similarity_matrix['u2s_obj_text_indexes'].to(device)
        # self.u2s_attr_text_weights = similarity_matrix['u2s_attr_text_weights'].to(device)
        # self.u2s_obj_text_weights = similarity_matrix['u2s_obj_text_weights'].to(device)
        # self.topk_u2s_attr_img_indices = similarity_matrix['u2s_attr_img_indexes'].to(device)
        # self.topk_u2s_obj_img_indices = similarity_matrix['u2s_obj_img_indexes'].to(device)
        # self.u2s_attr_img_weights = similarity_matrix['u2s_attr_img_weights'].to(device)
        # self.u2s_obj_img_weights = similarity_matrix['u2s_obj_img_weights'].to(device)

    #     ###save similarity matrix
    #     sim_matrix = {}
        train_attr_len = len(self.dset.train_attrs)
        train_obj_len = len(self.dset.train_objs)
        all_attr_len = len(self.dset.all_attrs)
        all_obj_len = len(self.dset.all_objs)

        ### unseen 和 seen 的token embedding相似度矩阵
        seen_attr, seen_obj = self.init_token_embedding1[:train_attr_len].clone().detach(), self.init_token_embedding1[self.offset:self.offset+train_obj_len].clone().detach()
        unseen_attr, unseen_obj = self.init_token_embedding1[train_attr_len: all_attr_len].clone().detach(), self.init_token_embedding1[self.offset+train_obj_len: self.offset+all_obj_len].clone().detach()
        self.u2s_attr_sim = F.cosine_similarity(unseen_attr.unsqueeze(1), seen_attr.unsqueeze(0), dim=2)
        self.u2s_obj_sim = F.cosine_similarity(unseen_obj.unsqueeze(1), seen_obj.unsqueeze(0), dim=2)

        topk_attr_sim, self.topk_u2s_attr_indices = torch.topk(self.u2s_attr_sim, k=5, dim=1)
        topk_obj_sim, self.topk_u2s_obj_indices = torch.topk(self.u2s_obj_sim, k=5, dim=1)

        self.u2s_attr_weights = F.softmax(topk_attr_sim, dim=1)
        self.u2s_obj_weights = F.softmax(topk_obj_sim, dim=1)

    # #    # 存储到 sim_matrix
    # #     sim_matrix.update({
    # #         'u2s_attr_indexes': self.topk_u2s_attr_indices,
    # #         'u2s_obj_indexes': self.topk_u2s_obj_indices,
    # #         'u2s_attr_weights': self.u2s_attr_weights,
    # #         'u2s_obj_weights': self.u2s_obj_weights
    # #     })

        ### unseen 和seen的text embedding相似度矩阵
        seen_attr_text_emb = self.token2text(seen_attr, 'attrs')
        seen_obj_text_emb = self.token2text(seen_obj, 'objs')
        unseen_attr_text_emb = self.token2text(unseen_attr, 'attrs')
        unseen_obj_text_emb = self.token2text(unseen_obj, 'objs')

        self.u2s_attr_text_sim = F.cosine_similarity(unseen_attr_text_emb.unsqueeze(1), seen_attr_text_emb.unsqueeze(0), dim=2)
        self.u2s_obj_text_sim = F.cosine_similarity(unseen_obj_text_emb.unsqueeze(1), seen_obj_text_emb.unsqueeze(0), dim=2)

        topk_attr_text_sim, self.topk_u2s_attr_text_indices = torch.topk(self.u2s_attr_text_sim, k=5, dim=1)
        topk_obj_text_sim, self.topk_u2s_obj_text_indices = torch.topk(self.u2s_obj_text_sim, k=5, dim=1)

        self.u2s_attr_text_weights = F.softmax(topk_attr_text_sim, dim=1)
        self.u2s_obj_text_weights = F.softmax(topk_obj_text_sim, dim=1)

    #     init_sim_attr_text = F.cosine_similarity(seen_attr_text_emb.unsqueeze(1), seen_attr_text_emb.unsqueeze(0), dim=2)
    #     init_sim_obj_text = F.cosine_similarity(seen_obj_text_emb.unsqueeze(1), seen_obj_text_emb.unsqueeze(0), dim=2)
    #     init_sim_attr_text -= torch.eye(init_sim_attr_text.size(0)).to(device)
    #     init_sim_obj_text -= torch.eye(init_sim_obj_text.size(0)).to(device)

    #     self.init_sim_attr_text_topk, self.init_sim_attr_text_topk_indices = torch.topk(init_sim_attr_text, k=5, dim=1)
    #     self.init_sim_obj_text_topk, self.init_sim_obj_text_topk_indices = torch.topk(init_sim_obj_text, k=5, dim=1)

        #   # 存储到 sim_matrix
        # sim_matrix.update({
        #     'u2s_attr_text_indexes': self.topk_u2s_attr_text_indices,
        #     'u2s_obj_text_indexes': self.topk_u2s_obj_text_indices,
        #     'u2s_attr_text_weights': self.u2s_attr_text_weights,
        #     'u2s_obj_text_weights': self.u2s_obj_text_weights
        # })

    #      ## unseen 和 seen 经过 image作用的相似度矩阵
    #     sample_img_feats = sample_img_feats.to(device)
    #     # **优化显存占用：逐步计算 image 相似度**
    #     batch_size = 100  # 调整 batch size 以控制显存
    #     num_unseen_attr = unseen_attr_text_emb.shape[0]
    #     num_unseen_obj = unseen_obj_text_emb.shape[0]

    #     self.topk_u2s_attr_img_indices, self.topk_u2s_obj_img_indices = [], []
    #     self.u2s_attr_img_weights, self.u2s_obj_img_weights = [], []

    #     for i in range(0, num_unseen_attr, batch_size):
    #         batch_unseen_attr = unseen_attr_text_emb[i:i+batch_size]

    #         u2i_attr_img_sim = F.cosine_similarity(batch_unseen_attr.unsqueeze(1), sample_img_feats.unsqueeze(0), dim=2)
    #         batch_s2i_attr_img_sim = []
    #         for j in range(0, len(seen_attr_text_emb), batch_size):
    #             batch_seen_attr = seen_attr_text_emb[j:j+batch_size]
    #             batch_sim = F.cosine_similarity(batch_seen_attr.unsqueeze(1), sample_img_feats.unsqueeze(0), dim=2)
    #             batch_s2i_attr_img_sim.append(batch_sim)
    #         # 合并所有 batch 结果
    #         s2i_attr_img_sim = torch.cat(batch_s2i_attr_img_sim, dim=0)  # [num_seen_obj, num_sample_img]
    #         u2s_attr_img_sim = F.cosine_similarity(u2i_attr_img_sim.unsqueeze(1), s2i_attr_img_sim.unsqueeze(0), dim=2)

    #         topk_attr_img_sim, topk_indices = torch.topk(u2s_attr_img_sim, k=5, dim=1)
    #         self.topk_u2s_attr_img_indices.append(topk_indices)
    #         self.u2s_attr_img_weights.append(F.softmax(topk_attr_img_sim, dim=1))

    #         del u2i_attr_img_sim, s2i_attr_img_sim, u2s_attr_img_sim
    #         torch.cuda.empty_cache()

    #     batch_size = 50
    #     for i in range(0, num_unseen_obj, batch_size):
    #         batch_unseen_obj = unseen_obj_text_emb[i:i+batch_size]

    #         u2i_obj_img_sim = F.cosine_similarity(batch_unseen_obj.unsqueeze(1), sample_img_feats.unsqueeze(0), dim=2)
    #          # 分批计算 seen_obj_text_emb 和 sample_img_feats 之间的相似度
    #         batch_s2i_obj_img_sim = []
    #         for j in range(0, len(seen_obj_text_emb), batch_size):
    #             batch_seen_obj = seen_obj_text_emb[j:j+batch_size]
    #             batch_sim = F.cosine_similarity(batch_seen_obj.unsqueeze(1), sample_img_feats.unsqueeze(0), dim=2)
    #             batch_s2i_obj_img_sim.append(batch_sim)

    #         # 合并所有 batch 结果
    #         s2i_obj_img_sim = torch.cat(batch_s2i_obj_img_sim, dim=0)  # [num_seen_obj, num_sample_img]
    #         u2s_obj_img_sim = F.cosine_similarity(u2i_obj_img_sim.unsqueeze(1), s2i_obj_img_sim.unsqueeze(0), dim=2)

    #         topk_obj_img_sim, topk_indices = torch.topk(u2s_obj_img_sim, k=5, dim=1)
    #         self.topk_u2s_obj_img_indices.append(topk_indices)
    #         self.u2s_obj_img_weights.append(F.softmax(topk_obj_img_sim, dim=1))

    #         del u2i_obj_img_sim, s2i_obj_img_sim, u2s_obj_img_sim
    #         torch.cuda.empty_cache()

    #     # 拼接所有 batch 结果
    #     self.topk_u2s_attr_img_indices = torch.cat(self.topk_u2s_attr_img_indices, dim=0)
    #     self.topk_u2s_obj_img_indices = torch.cat(self.topk_u2s_obj_img_indices, dim=0)
    #     self.u2s_attr_img_weights = torch.cat(self.u2s_attr_img_weights, dim=0)
    #     self.u2s_obj_img_weights = torch.cat(self.u2s_obj_img_weights, dim=0)

    #     # 存储到 sim_matrix
    #     sim_matrix.update({
    #         'u2s_attr_img_indexes': self.topk_u2s_attr_img_indices,
    #         'u2s_obj_img_indexes': self.topk_u2s_obj_img_indices,
    #         'u2s_attr_img_weights': self.u2s_attr_img_weights,
    #         'u2s_obj_img_weights': self.u2s_obj_img_weights
    #     })
    #     # Save the similarity matrix to a file
    #     torch.save(sim_matrix, 'vaw_sim_matrix.pt')  #
    # #     ###end

        self.weight = 0.8
        self.fusion = FusionTextImageBlock(width_img=768, width_txt=512, attributes=self.train_attr_len, classes=self.train_obj_len, layers=1, context_length=8, fusion='txt2img').cuda()


        self.pair2attr_obj, self.train_extra_pair2attr_obj = self.construct_pair_to_ao_idx(dset.unique_pairs, dset.train_pairs_extra)

    def token2text(self, token_emb, type):
        if 'attrs' in type:
            eos_idx = int(self.attr_tempelete_token_id.argmax())
            token_id = self.attr_tempelete_token_id.repeat(token_emb.size(0), 1)
        elif 'objs' in type:
            eos_idx = int(self.obj_tempelete_token_id.argmax())
            token_id = self.obj_tempelete_token_id.repeat(token_emb.size(0), 1)

        token_embedding = self.clip_model.token_embedding(token_id)

        if type == 'attrs':
            token_embedding[:, eos_idx-2, :] = token_emb
        elif type =='objs':
            token_embedding[:, eos_idx-1, :] = token_emb

        text_emb, _ = self.clip.text_encoder(token_id, token_embedding, enable_pos_emb=True)

        return text_emb

    def construct_pair_to_ao_idx(self, unique_pairs, train_extra_pairs):
        pair2attr_obj = []
        #构造pair到attr和obj的映射
        for idx, pair in enumerate(unique_pairs):
            pair2attr_obj.append((self.dset.unique_attr2idx[pair[0]], self.dset.unique_obj2idx[pair[1]]))

        train_extra_pair2attr_obj = []
        for idx, pair in enumerate(train_extra_pairs):
            train_extra_pair2attr_obj.append((self.dset.unique_attr2idx[pair[0]], self.dset.unique_obj2idx[pair[1]]))

        return pair2attr_obj, train_extra_pair2attr_obj

    def initialize_token_embeddings(self, dset):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenized = clip.tokenize(dset, context_length=16).to(device)
        eos_ids = tokenized.argmax(dim=-1)

        tokenize_embedding = self.clip_model.token_embedding(tokenized)

        mean_embeddings = []

        for i, eos_id in enumerate(eos_ids):
            mean_embedding = torch.mean(tokenize_embedding[i, 1:eos_id, :], dim=0)
            mean_embeddings.append(mean_embedding)

        mean_embeddings = torch.stack(mean_embeddings)

        return mean_embeddings

    def construct_token_tensors(self, idx, type):

        train_attr_len = len(self.dset.train_attrs)
        train_obj_len = len(self.dset.train_objs)
        train_attr = self.ao_dropout(self.soft_embedding[:train_attr_len])
        train_obj = self.ao_dropout(self.soft_embedding[self.offset:self.offset+train_obj_len])
        soft_embedding = torch.cat(
            (train_attr, self.soft_embedding[train_attr_len:self.offset],train_obj, self.soft_embedding[self.offset+train_obj_len:]), dim=0
        )

        eos_idx = int(self.pair_tempelete_token_id.argmax())
        token_id = self.pair_tempelete_token_id.repeat(len(idx), 1)

        token_embedding = self.clip_model.token_embedding(token_id)

        attr_id = [self.pair2attr_obj[pair][0] for pair in idx]
        obj_id = [self.pair2attr_obj[pair][1] for pair in idx]
        token_embedding[:, eos_idx-2, :] = soft_embedding[attr_id]
        token_embedding[:, eos_idx-1, :] = soft_embedding[[id + self.offset for id in obj_id]]

        # token_embedding[:, 1 : len(self.soft_prompt)+1, :] = self.soft_prompt

        text_emb, text_ft = self.clip.text_encoder(token_id, token_embedding, enable_pos_emb=True)

        return text_emb, text_ft

    def ft_to_logit(self, img, txt):
        img_feature = img.permute(1, 0, 2)  # LND -> NLD

        img_feature = self.clip.image_encoder.ln_post(img_feature[:, 0, :])
        if self.clip.image_encoder.proj is not None:
            img_feature = img_feature @ self.clip.image_encoder.proj

        if self.cfg.TRAIN.fusion in ["BiFusion", "img2txt"]:
            txt_feature = txt.permute(0, 2, 1, 3)
            txt_feature = self.clip.text_encoder.ln_final(txt_feature)
            txt_tf = (
                txt_feature[
                    :, torch.arange(txt_feature.shape[1]), self.token_ids.argmax(dim=-1)
                ]  # POS of <EOS>
                @ self.clip.text_encoder.text_projection
            )
        else:
            txt_feature = txt.permute(1, 0, 2)
            txt_feature = self.clip.text_encoder.ln_final(txt_feature)
            txt_tf = (
                txt_feature[
                    torch.arange(txt_feature.shape[0]), self.pair_tempelete_token_id.argmax(dim=-1)
                ]  # POS of <EOS>
                @ self.clip.text_encoder.text_projection
            )
        return img_feature, txt_tf
    def decompose_logits(self, logits, idx):
        att_idx, obj_idx = idx[:, 0], idx[:, 1]
        logits_att = torch.zeros(logits.shape[0], self.train_attr_len).cuda()
        logits_obj = torch.zeros(logits.shape[0], self.train_obj_len).cuda()
        for i in range(self.train_attr_len):
            logits_att[:, i] = logits[:, np.where(att_idx==i)[0]].mean(-1)
        for i in range(self.train_obj_len):
            logits_obj[:, i] = logits[:, np.where(obj_idx==i)[0]].mean(-1)
        return logits_att, logits_obj


    def train_forward(self, batch):
        img1 = batch['img']

        # Labels of 1st image.
        attr_labels = batch['attr']
        obj_labels = batch['obj']
        pair_labels = batch['pair']

        bs = img1.shape[0]

        batch_img, img_ft = self.clip.encode_image(img1.half())
        batch_img, img_ft = batch_img.float(), img_ft.float()

        text_features, text_ft= self.construct_token_tensors(self.train_pairs, 'pairs')

        batch_img_soft_prompt = batch_img / batch_img.norm(dim=-1, keepdim=True)
        text_features_soft_prompt = text_features / text_features.norm(dim=-1, keepdim=True)

        idx = [[self.pair2attr_obj[item][0],self.pair2attr_obj[item][1]] for item in self.train_pairs]
        idx = np.array(idx)

        img_ft, text_ft = self.fusion(img_ft.type(torch.float), text_ft.type(torch.float), idx, bs)
        img_ft, text_ft = self.ft_to_logit(img_ft.type(self.clip.dtype), text_ft.type(self.clip.dtype))

        batch_img = self.weight * batch_img + (1 - self.weight) * img_ft
        normalized_img = batch_img / batch_img.norm(dim=-1, keepdim=True)

        if self.cfg.TRAIN.fusion in ["BiFusion", "img2txt"]:
            text_features = self.weight * text_features.repeat(bs, 1, 1) + (1 - self.weight) * text_ft
        else:
            text_features = self.weight * text_features + (1 - self.weight) * text_ft
        idx_text_features = text_features / text_features.norm(
            dim=-1, keepdim=True
        )
        if self.cfg.TRAIN.fusion in ["BiFusion", "img2txt"]:
            logits = (
                self.clip.logit_scale.exp()
                * normalized_img.unsqueeze(1)
                @ idx_text_features.permute(0,2,1)
            ).squeeze()     ###     48 * 1262
        else:
            logits = (
                self.clip.logit_scale.exp()
                * normalized_img
                @ idx_text_features.float().t()
            )

        logits_soft_prompt = (
            self.clip.logit_scale.exp()
            * batch_img_soft_prompt
            @ text_features_soft_prompt.float().t()
        )

        logits_att, logits_obj = self.decompose_logits(logits_soft_prompt, idx)

        predict = (logits, logits_att, logits_obj, logits_soft_prompt)
        loss = loss_calu(predict, batch)
        pred = torch.max(logits, dim = 1)[1]
        attr_pred = self.train_attrs[pred]
        obj_pred = self.train_objs[pred]

        correct_attr = (attr_pred == attr_labels)
        correct_obj = (obj_pred == obj_labels)
        correct_pair = (pred == pair_labels)

        out = {
            'loss': loss,
            'acc_attr': torch.div(correct_attr.sum(),float(bs)),
            'acc_obj': torch.div(correct_obj.sum(),float(bs)),
            'acc_pair': torch.div(correct_pair.sum(),float(bs)),

        }

        return out

    def val_forward(self, batch):
        img = batch['img']

        bs = img.shape[0]

        batch_img, img_ft = self.clip.encode_image(img.half())
        batch_img, img_ft = batch_img.float(), img_ft.float()

        text_features, text_ft= self.construct_token_tensors(self.all_pairs, 'pairs')

        idx = [[self.pair2attr_obj[item][0],self.pair2attr_obj[item][1]] for item in self.all_pairs]
        idx = np.array(idx)

        img_ft, text_ft = self.fusion(img_ft.type(torch.float), text_ft.type(torch.float), idx, bs)
        img_ft, text_ft = self.ft_to_logit(img_ft.type(self.clip.dtype), text_ft.type(self.clip.dtype))

        batch_img = self.weight * batch_img + (1 - self.weight) * img_ft
        normalized_img = batch_img / batch_img.norm(dim=-1, keepdim=True)

        if self.cfg.TRAIN.fusion in ["BiFusion", "img2txt"]:
            text_features = self.weight * text_features.repeat(bs, 1, 1) + (1 - self.weight) * text_ft
        else:
            text_features = self.weight * text_features + (1 - self.weight) * text_ft
        idx_text_features = text_features / text_features.norm(
            dim=-1, keepdim=True
        )
        if self.cfg.TRAIN.fusion in ["BiFusion", "img2txt"]:
            logits = (
                self.clip.logit_scale.exp()
                * normalized_img.unsqueeze(1)
                @ idx_text_features.permute(0,2,1)
            ).squeeze()     ###     48 * 1262
        else:
            logits = (
                self.clip.logit_scale.exp()
                * normalized_img
                @ idx_text_features.float().t()
            )

        pred = F.softmax(logits, dim=1)
        out = {}
        out['pred'] = pred

        out['scores'] = {}
        for _, pair in enumerate(self.all_pairs1):  # all-pair id, all-pairs name
            out['scores'][pair] = pred[:,self.pair2idx[pair]]

        return out

    def forward(self, x):
        if self.training:
            out = self.train_forward(x)
        else:
            with torch.no_grad():
                out = self.val_forward(x)
        return out
