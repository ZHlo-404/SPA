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
from torch.nn.modules.loss import CrossEntropyLoss
from .clip_modules.interface import CLIPInterface
from .common import Adapter, Disentangler, CrossAttentionLayer
from .loss import loss_calu
import torch.utils.checkpoint as checkpoint

class Troika(nn.Module):
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

        self.cross_attn_dropout = cfg.TRAIN.cross_attn_dropout if hasattr(cfg, 'cross_attn_dropout') else 0.1
        self.prim_loss_weight = cfg.TRAIN.prim_loss_weight if hasattr(cfg, 'prim_loss_weight') else 1

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, preprocess = load(
            cfg.TRAIN.clip_model, device=device, context_length=cfg.TRAIN.context_length
        )
        self.clip = CLIPInterface(self.clip_model, cfg, device=device)

        attr_tempelete = 'a photo of x object'
        obj_tempelete = 'a photo of x'
        pair_tempelete = 'a photo of x x'
        self.attr_tempelete_token_id = clip.tokenize(
            [attr_tempelete], context_length=cfg.TRAIN.context_length
        ).to(device)
        self.obj_tempelete_token_id = clip.tokenize(
            [obj_tempelete], context_length=cfg.TRAIN.context_length
        ).to(device)
        self.pair_tempelete_token_id = clip.tokenize(
            [pair_tempelete], context_length=cfg.TRAIN.context_length
        ).to(device)

        ctx_init = "a photo of "
        n_ctx = len(ctx_init.split())
        prompt = clip.tokenize(
            [pair_tempelete], context_length=cfg.TRAIN.context_length
        ).to(device)
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

        # self.additional_visual_params = self.add_visual_tunable_params().half()

        output_dim = 512
        self.attr_disentangler = Disentangler(output_dim)
        self.obj_disentangler = Disentangler(output_dim)

        self.cmt = nn.ModuleList([CrossAttentionLayer(output_dim, output_dim//64, self.cross_attn_dropout) for _ in range(cfg.TRAIN.cmt_layers)])
        self.lamda = nn.Parameter(torch.ones(output_dim) * cfg.TRAIN.init_lamda)
        self.patch_norm = nn.LayerNorm(output_dim)

        self.train_attr_len = len(self.dset.train_attrs)
        self.train_obj_len = len(self.dset.train_objs)
        self.all_attr_len = len(self.dset.all_attrs)
        self.all_obj_len = len(self.dset.all_objs)

        seen_attr, seen_obj = self.init_token_embedding1[:self.train_attr_len].clone().detach(), self.init_token_embedding1[self.offset:self.offset+self.train_obj_len].clone().detach()
        unseen_attr, unseen_obj = self.init_token_embedding1[self.train_attr_len: self.all_attr_len], self.init_token_embedding1[self.offset+self.train_obj_len: self.offset+self.all_obj_len]
        self.u2s_attr_sim = F.cosine_similarity(unseen_attr.unsqueeze(1), seen_attr.unsqueeze(0), dim=2)
        self.u2s_obj_sim = F.cosine_similarity(unseen_obj.unsqueeze(1), seen_obj.unsqueeze(0), dim=2)

        topk_attr_sim, self.topk_u2s_attr_indices = torch.topk(self.u2s_attr_sim, k=5, dim=1)
        topk_obj_sim, self.topk_u2s_obj_indices = torch.topk(self.u2s_obj_sim, k=5, dim=1)

        self.u2s_attr_weights = F.softmax(topk_attr_sim, dim=1)
        self.u2s_obj_weights = F.softmax(topk_obj_sim, dim=1)

        self.weight = 0.8

        self.pair2attr_obj, self.train_extra_pair2attr_obj = self.construct_pair_to_ao_idx(dset.unique_pairs, dset.train_pairs_extra)


        # self.train_attr_len = len(self.dset.train_attrs)
        # self.train_obj_len = len(self.dset.train_objs)
        # self.all_attr_len = len(self.dset.all_attrs)
        # self.all_obj_len = len(self.dset.all_objs)
        # # ### load similarity matrix
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


    #      ###save similarity matrix
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

        init_sim_attr_text = F.cosine_similarity(seen_attr_text_emb.unsqueeze(1), seen_attr_text_emb.unsqueeze(0), dim=2)
        init_sim_obj_text = F.cosine_similarity(seen_obj_text_emb.unsqueeze(1), seen_obj_text_emb.unsqueeze(0), dim=2)
        init_sim_attr_text -= torch.eye(init_sim_attr_text.size(0)).to(device)
        init_sim_obj_text -= torch.eye(init_sim_obj_text.size(0)).to(device)

        self.init_sim_attr_text_topk, self.init_sim_attr_text_topk_indices = torch.topk(init_sim_attr_text, k=5, dim=1)
        self.init_sim_obj_text_topk, self.init_sim_obj_text_topk_indices = torch.topk(init_sim_obj_text, k=5, dim=1)
        #   # 存储到 sim_matrix
        # sim_matrix.update({
        #     'u2s_attr_text_indexes': self.topk_u2s_attr_text_indices,
        #     'u2s_obj_text_indexes': self.topk_u2s_obj_text_indices,
        #     'u2s_attr_text_weights': self.u2s_attr_text_weights,
        #     'u2s_obj_text_weights': self.u2s_obj_text_weights
        # })

        #  ## unseen 和 seen 经过 image作用的相似度矩阵
        # sample_img_feats = sample_img_feats.to(device)
        # # **优化显存占用：逐步计算 image 相似度**
        # batch_size = 100  # 调整 batch size 以控制显存
        # num_unseen_attr = unseen_attr_text_emb.shape[0]
        # num_unseen_obj = unseen_obj_text_emb.shape[0]

        # self.topk_u2s_attr_img_indices, self.topk_u2s_obj_img_indices = [], []
        # self.u2s_attr_img_weights, self.u2s_obj_img_weights = [], []

        # for i in range(0, num_unseen_attr, batch_size):
        #     batch_unseen_attr = unseen_attr_text_emb[i:i+batch_size]

        #     u2i_attr_img_sim = F.cosine_similarity(batch_unseen_attr.unsqueeze(1), sample_img_feats.unsqueeze(0), dim=2)
        #     batch_s2i_attr_img_sim = []
        #     for j in range(0, len(seen_attr_text_emb), batch_size):
        #         batch_seen_attr = seen_attr_text_emb[j:j+batch_size]
        #         batch_sim = F.cosine_similarity(batch_seen_attr.unsqueeze(1), sample_img_feats.unsqueeze(0), dim=2)
        #         batch_s2i_attr_img_sim.append(batch_sim)
        #     # 合并所有 batch 结果
        #     s2i_attr_img_sim = torch.cat(batch_s2i_attr_img_sim, dim=0)  # [num_seen_obj, num_sample_img]
        #     u2s_attr_img_sim = F.cosine_similarity(u2i_attr_img_sim.unsqueeze(1), s2i_attr_img_sim.unsqueeze(0), dim=2)

        #     topk_attr_img_sim, topk_indices = torch.topk(u2s_attr_img_sim, k=5, dim=1)
        #     self.topk_u2s_attr_img_indices.append(topk_indices)
        #     self.u2s_attr_img_weights.append(F.softmax(topk_attr_img_sim, dim=1))

        #     del u2i_attr_img_sim, s2i_attr_img_sim, u2s_attr_img_sim
        #     torch.cuda.empty_cache()

        # batch_size = 50
        # for i in range(0, num_unseen_obj, batch_size):
        #     batch_unseen_obj = unseen_obj_text_emb[i:i+batch_size]

        #     u2i_obj_img_sim = F.cosine_similarity(batch_unseen_obj.unsqueeze(1), sample_img_feats.unsqueeze(0), dim=2)
        #      # 分批计算 seen_obj_text_emb 和 sample_img_feats 之间的相似度
        #     batch_s2i_obj_img_sim = []
        #     for j in range(0, len(seen_obj_text_emb), batch_size):
        #         batch_seen_obj = seen_obj_text_emb[j:j+batch_size]
        #         batch_sim = F.cosine_similarity(batch_seen_obj.unsqueeze(1), sample_img_feats.unsqueeze(0), dim=2)
        #         batch_s2i_obj_img_sim.append(batch_sim)

        #     # 合并所有 batch 结果
        #     s2i_obj_img_sim = torch.cat(batch_s2i_obj_img_sim, dim=0)  # [num_seen_obj, num_sample_img]
        #     u2s_obj_img_sim = F.cosine_similarity(u2i_obj_img_sim.unsqueeze(1), s2i_obj_img_sim.unsqueeze(0), dim=2)

        #     topk_obj_img_sim, topk_indices = torch.topk(u2s_obj_img_sim, k=5, dim=1)
        #     self.topk_u2s_obj_img_indices.append(topk_indices)
        #     self.u2s_obj_img_weights.append(F.softmax(topk_obj_img_sim, dim=1))

        #     del u2i_obj_img_sim, s2i_obj_img_sim, u2s_obj_img_sim
        #     torch.cuda.empty_cache()

        # # 拼接所有 batch 结果
        # self.topk_u2s_attr_img_indices = torch.cat(self.topk_u2s_attr_img_indices, dim=0)
        # self.topk_u2s_obj_img_indices = torch.cat(self.topk_u2s_obj_img_indices, dim=0)
        # self.u2s_attr_img_weights = torch.cat(self.u2s_attr_img_weights, dim=0)
        # self.u2s_obj_img_weights = torch.cat(self.u2s_obj_img_weights, dim=0)

        # # 存储到 sim_matrix
        # sim_matrix.update({
        #     'u2s_attr_img_indexes': self.topk_u2s_attr_img_indices,
        #     'u2s_obj_img_indexes': self.topk_u2s_obj_img_indices,
        #     'u2s_attr_img_weights': self.u2s_attr_img_weights,
        #     'u2s_obj_img_weights': self.u2s_obj_img_weights
        # })
        # # Save the similarity matrix to a file
        # torch.save(sim_matrix, 'vaw_sim_matrix.pt')  #
        # #     ###end

    def add_visual_tunable_params(self):
        adapter_num = 2 * self.clip.image_encoder.visual_transformer.layers
        params = nn.ModuleList([Adapter(d_model=self.clip.image_encoder.visual_transformer.width,
                                    bottleneck=self.cfg.TRAIN.adapter_dim,
                                    dropout=self.cfg.TRAIN.adapter_dropout
                                ) for _ in range(adapter_num)])
        return params

    def construct_pair_to_ao_idx(self, unique_pairs, train_extra_pairs):
        pair2attr_obj = []
        #构造pair到attr和obj的映射
        for idx, pair in enumerate(unique_pairs):
            pair2attr_obj.append((self.dset.unique_attr2idx[pair[0]], self.dset.unique_obj2idx[pair[1]]))

        train_extra_pair2attr_obj = []
        for idx, pair in enumerate(train_extra_pairs):
            train_extra_pair2attr_obj.append((self.dset.unique_attr2idx[pair[0]], self.dset.unique_obj2idx[pair[1]]))

        return pair2attr_obj, train_extra_pair2attr_obj

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

        text_emb = self.clip.text_encoder(token_id, token_embedding, enable_pos_emb=True)

        return text_emb

    def initialize_token_embeddings(self, dset):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenized = clip.tokenize(
            dset, context_length=self.cfg.TRAIN.context_length
        ).to(device)
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

        def text_encoder_forward(token_id, token_embedding):
            return self.clip.text_encoder(token_id, token_embedding, enable_pos_emb=True)

        ### pair text embedding
        eos_idx = int(self.pair_tempelete_token_id.argmax())
        token_id = self.pair_tempelete_token_id.repeat(len(idx), 1)

        token_embedding = self.clip_model.token_embedding(token_id)

        attr_id = [self.pair2attr_obj[pair][0] for pair in idx]
        obj_id = [self.pair2attr_obj[pair][1] for pair in idx]
        token_embedding[:, eos_idx-2, :] = soft_embedding[attr_id]
        token_embedding[:, eos_idx-1, :] = soft_embedding[[id + self.offset for id in obj_id]]

        ## use ctx prompt
        # token_embedding[:, 1 : len(self.soft_prompt)+1, :] = self.soft_prompt

        # text_emb = self.clip.text_encoder(token_id, token_embedding, enable_pos_emb=True)
        text_emb = checkpoint.checkpoint(
            text_encoder_forward,  # 传递函数而非模块
            token_id,
            token_embedding
        )

        ### attr text embedding
        attr_eos_idx = int(self.attr_tempelete_token_id.argmax())
        attr_token_id = self.attr_tempelete_token_id.repeat(len(idx), 1)
        attr_token_embedding = self.clip_model.token_embedding(attr_token_id)
        attr_token_embedding[:, attr_eos_idx-2, :] = soft_embedding[attr_id]
        # attr_text_emb = self.clip.text_encoder(attr_token_id, attr_token_embedding, enable_pos_emb=True)
        attr_text_emb = checkpoint.checkpoint(
            text_encoder_forward,  # 正确
            attr_token_id,
            attr_token_embedding
        )

        ### obj text embedding
        obj_eos_idx = int(self.obj_tempelete_token_id.argmax())
        obj_token_id = self.obj_tempelete_token_id.repeat(len(idx), 1)
        obj_token_embedding = self.clip_model.token_embedding(obj_token_id)
        obj_token_embedding[:, obj_eos_idx-1, :] = soft_embedding[[id + self.offset for id in obj_id]]
        # obj_text_emb = self.clip.text_encoder(obj_token_id, obj_token_embedding, enable_pos_emb=True)
        obj_text_emb = checkpoint.checkpoint(
            text_encoder_forward,
            obj_token_id,
            obj_token_embedding,
        )

        return text_emb, attr_text_emb, obj_text_emb

    # def encode_image(self, x: torch.Tensor):
    #     return self.encode_image_with_adapter(x)

    # def encode_image_with_adapter(self, x: torch.Tensor):
    #     x = self.clip.image_encoder.conv1(x)  # shape = [*, width, grid, grid]
    #     x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
    #     x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
    #     x = torch.cat([self.clip.image_encoder.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
    #     x = x + self.clip.image_encoder.visual_positional_embedding.to(x.dtype)
    #     x = self.clip.image_encoder.ln_pre(x)

    #     x = x.permute(1, 0, 2)  # NLD -> LND
    #     # img_feature = self.clip.visual.transformer(x)
    #     for i_block in range(self.clip.image_encoder.visual_transformer.layers):
    #         # MHA
    #         adapt_x = self.additional_visual_params[i_block](x, add_residual=False)
    #         residual = x
    #         x = self.clip.image_encoder.visual_transformer.resblocks[i_block].attention(
    #             self.clip.image_encoder.visual_transformer.resblocks[i_block].ln_1(x)
    #         )
    #         x = x + adapt_x + residual

    #         # FFN
    #         i_adapter = i_block + self.clip.image_encoder.visual_transformer.layers
    #         adapt_x = self.additional_visual_params[i_adapter](x, add_residual=False)
    #         residual = x
    #         x = self.clip.image_encoder.visual_transformer.resblocks[i_block].mlp(
    #             self.clip.image_encoder.visual_transformer.resblocks[i_block].ln_2(x)
    #         )
    #         x = x + adapt_x + residual

    #     img_feature = x.permute(1, 0, 2)  # LND -> NLD

    #     img_feature = self.clip.image_encoder.ln_post(img_feature)
    #     if self.clip.image_encoder.proj is not None:
    #         img_feature = img_feature @ self.clip.image_encoder.proj
    #     return img_feature[:, 0, :], img_feature

    def loss_calu(self, predict, target):
        loss_fn = CrossEntropyLoss()
        _, batch_attr, batch_obj, batch_target = target
        comp_logits, attr_logits, obj_logits = predict
        batch_attr = batch_attr.cuda()
        batch_obj = batch_obj.cuda()
        batch_target = batch_target.cuda()
        loss_comp = loss_fn(comp_logits, batch_target)
        loss_attr = loss_fn(attr_logits, batch_attr)
        loss_obj = loss_fn(obj_logits, batch_obj)
        loss = loss_comp * self.cfg.TRAIN.pair_loss_weight +\
               loss_attr * self.cfg.TRAIN.attr_loss_weight +\
               loss_obj * self.cfg.TRAIN.obj_loss_weight
        return loss

    def logit_infer(self, predict, pairs):
        comp_logits, attr_logits, obj_logits = predict
        # print(f"comp_logits: {comp_logits.shape}, attr_logits: {attr_logits.shape}, obj_logits: {obj_logits.shape}")
        #comp_logits: torch.Size([32, 1962]), attr_logits: torch.Size([32, 230]), obj_logits: torch.Size([32, 490])
        attr_pred = F.softmax(attr_logits, dim=-1)
        obj_pred = F.softmax(obj_logits, dim=-1)
        for i_comp in range(comp_logits.shape[-1]):
            weighted_attr_pred = 1 if self.cfg.TRAIN.attr_inference_weight == 0 else attr_pred[:, self.pair2attr_obj[pairs[i_comp]][0]] * self.cfg.TRAIN.attr_inference_weight
            weighted_obj_pred = 1 if self.cfg.TRAIN.obj_inference_weight == 0 else obj_pred[:, self.pair2attr_obj[pairs[i_comp]][1]] * self.cfg.TRAIN.obj_inference_weight
            # weighted_attr_pred: bs*1, weighted_obj_pred: bs*1
            comp_logits[:, i_comp] = comp_logits[:, i_comp] * self.cfg.TRAIN.pair_inference_weight + weighted_attr_pred * weighted_obj_pred
        # print(f"comp_logits: {comp_logits.shape}")
        #comp_logits: torch.Size([32, 1962])
        return comp_logits

    def train_forward(self, batch):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            img1 = batch['img']

            # Labels of 1st image.
            attr_labels = batch['attr']
            obj_labels = batch['obj']
            pair_labels = batch['pair']

            bs = img1.shape[0]

            # batch_img, batch_patch = self.encode_image(img1.half())
            batch_img, batch_patch = self.clip.encode_image(img1.half())
            batch_img, batch_patch = batch_img, batch_patch

                # 图像特征提取全用 FP16
            batch_img_features = [
                batch_img,
                self.attr_disentangler(batch_img).half(),
                self.obj_disentangler(batch_img).half()
            ]
            normalized_img_features = [feats / feats.norm(dim=-1, keepdim=True) for feats in batch_img_features]


            ##original text embedding
            text_tuple = self.construct_token_tensors(self.train_pairs, 'pairs')
            text_tuple = [t.half() for t in text_tuple]
            logits = list()

            batch_patch = self.patch_norm(batch_patch)  # ✅ 预处理 `batch_patch`

            # ##original code
            # logits = list()
            # for i_element in range(len(text_tuple)):
            #     _text_features = text_tuple[i_element]
            #     idx_text_features = _text_features / _text_features.norm(
            #         dim=-1, keepdim=True
            #     )
            #     # CMT
            #     cmt_text_features = idx_text_features.unsqueeze(0).expand(bs, -1, -1)
            #     cmt_text_features = cmt_text_features.float()
            #     for layer in self.cmt:
            #         cmt_text_features = layer(cmt_text_features, batch_patch)
            #     cmt_text_features = idx_text_features + self.lamda * cmt_text_features.squeeze(1)

            #     cmt_text_features = cmt_text_features / cmt_text_features.norm(
            #         dim=-1, keepdim=True
            #     )

            #     logits.append(
            #         torch.einsum(
            #             "bd, bkd->bk",
            #             normalized_img_features[i_element],
            #             cmt_text_features * self.clip.logit_scale.exp()
            #     ))
            logits = list()

            def compute_cmt_text_features(idx_text_features, batch_patch):
                """ 计算 CMT Text Features，用于 checkpoint 计算 """
                cmt_text_features = idx_text_features.unsqueeze(0).expand(bs, -1, -1)
                cmt_text_features = cmt_text_features.float()

                for layer in self.cmt:
                    cmt_text_features = layer(cmt_text_features, batch_patch)

                cmt_text_features = idx_text_features + self.lamda * cmt_text_features.squeeze(1)
                return cmt_text_features / cmt_text_features.norm(dim=-1, keepdim=True)

            for i_element in range(len(text_tuple)):
                _text_features = text_tuple[i_element]
                idx_text_features = _text_features / _text_features.norm(dim=-1, keepdim=True)

                # **使用 checkpoint 计算 CMT 传播，减少显存占用**
                cmt_text_features = checkpoint.checkpoint(
                    compute_cmt_text_features,
                    idx_text_features.requires_grad_(),
                    batch_patch.requires_grad_()
                )

                logits.append(
                    torch.einsum(
                        "bd, bkd->bk",
                        normalized_img_features[i_element],
                        cmt_text_features * self.clip.logit_scale.exp()
                    )
                )

            multi_logits = self.logit_infer(logits, self.train_pairs)
            pred = torch.max(multi_logits, dim = 1)[1]
            attr_pred = self.train_attrs[pred]
            obj_pred = self.train_objs[pred]

            correct_attr = (attr_pred == attr_labels)
            correct_obj = (obj_pred == obj_labels)
            correct_pair = (pred == pair_labels)

            loss = self.loss_calu(logits, (batch_img, attr_labels, obj_labels, pair_labels))

            out = {
                'loss': loss,
                'acc_attr': torch.div(correct_attr.sum(),float(bs)),
                'acc_obj': torch.div(correct_obj.sum(),float(bs)),
                'acc_pair': torch.div(correct_pair.sum(),float(bs)),
            }

            return out


    def val_forward(self, batch):
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                img = batch['img']
                bs = img.shape[0]

                # batch_img, batch_patch = self.encode_image(img.half())
                batch_img, batch_patch = self.clip.encode_image(img.half())
                batch_img, batch_patch = batch_img.half(), batch_patch.half()
                batch_img_features = [
                    batch_img,
                    self.attr_disentangler(batch_img).half(),
                    self.obj_disentangler(batch_img).half()
                ]
                normalized_img_features = [feats / feats.norm(dim=-1, keepdim=True) for feats in batch_img_features]

                text_tuple = self.construct_token_tensors(self.all_pairs, 'pairs')
                text_tuple = [t.half() for t in text_tuple]


                chunk_size = 512
                num_chunks = (text_tuple[0].size(0) + chunk_size - 1) // chunk_size  # 计算 chunk 数量


                logits = list()
                batch_patch = self.patch_norm(batch_patch)
                for i_element in range(len(text_tuple)):
                    _text_features = text_tuple[i_element]
                    idx_text_features = _text_features / _text_features.norm(dim=-1, keepdim=True)

                    # CMT 计算
                    trunk_cmt_text = []
                    cmt_text_features = idx_text_features.unsqueeze(0).expand(bs, -1, -1)
                    cmt_text_features = cmt_text_features.float()

                    for chunk_idx in range(num_chunks):
                        start = chunk_idx * chunk_size
                        end = min((chunk_idx + 1) * chunk_size, cmt_text_features.shape[1])
                        chunk_cmt_text_features = cmt_text_features[:, start:end, :]
                        for layer in self.cmt:
                            chunk_cmt_text_features = layer( chunk_cmt_text_features, batch_patch)
                        trunk_cmt_text.append(chunk_cmt_text_features)
                    trunk_cmt_text = torch.cat(trunk_cmt_text, dim=1)
                    cmt_text_features = trunk_cmt_text
                    cmt_text_features = idx_text_features + self.lamda * cmt_text_features.squeeze(1)
                    cmt_text_features = cmt_text_features / cmt_text_features.norm(dim=-1, keepdim=True)

                    logits.append(
                        torch.einsum(
                            "bd, bkd->bk",
                            normalized_img_features[i_element],
                            cmt_text_features * self.clip.logit_scale.exp()
                    ))


                # logits = list()
                # for i_element in range(len(text_tuple)):
                #     _text_features = text_tuple[i_element]
                #     idx_text_features = _text_features / _text_features.norm(
                #         dim=-1, keepdim=True
                #     )
                #     # print(idx_text_features.shape)
                #     # CMT
                #     cmt_text_features = idx_text_features.unsqueeze(0).expand(bs, -1, -1)
                #     cmt_text_features = cmt_text_features.float()
                #     batch_patch = self.patch_norm(batch_patch)
                #     for layer in self.cmt:
                #         cmt_text_features = layer(cmt_text_features, batch_patch)
                #     cmt_text_features = idx_text_features + self.lamda * cmt_text_features.squeeze(1)

                #     cmt_text_features = cmt_text_features / cmt_text_features.norm(
                #         dim=-1, keepdim=True
                #     )

                #     logits.append(
                #         torch.einsum(
                #             "bd, bkd->bk",
                #             normalized_img_features[i_element],
                #             cmt_text_features * self.clip.logit_scale.exp()
                #     ))


                multi_logits = self.logit_infer(logits, self.all_pairs)
                pred = F.softmax(multi_logits, dim=1)
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
