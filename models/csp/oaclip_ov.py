import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import json

import os
import clip
from torch.autograd import Variable
from torch.nn.modules.loss import _WeightedLoss
import pdb
from sklearn.cluster import KMeans, DBSCAN


from clip_modules.model_loader import load
from clip_modules.interface import CLIPInterface
from .common import CosineClassifier, ImagePairComparison, LabelSmoothingCrossEntropy_pair, LabelSmoothingCrossEntropy, \
    GradientCorrectionMLP, DeepGradientCorrectionMLP, CustomTransformerDecoder, CustomTransformerDecoderLayer
## Label Smoothing using manual weights, seems to work for training labels only, not sure if we add neighbors

class OACLIPv3(nn.Module):
    """Object-Attribute Compositional Learning from Image Pair.
    """
    def __init__(self, dset, cfg):
        super(OACLIPv3, self).__init__()
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

        # ## extra pairs
        # self.extra_obj2idx = {obj: idx for idx, obj in enumerate(dset.extra_objs)}
        # self.extra_attr2idx = {attr: idx for idx, attr in enumerate(dset.extra_attrs)}
        # self.extra_pair2idx = {pair: idx for idx, pair in enumerate(dset.extra_pairs)}
        # extra_objs = [self.extra_obj2idx[obj] for obj in dset.extra_objs]
        # extra_attrs = [self.extra_attr2idx[attr] for attr in dset.extra_attrs]
        # extra_pairs = [self.extra_pair2idx[pair] for pair in dset.extra_pairs]
        # self.extra_attrs = torch.LongTensor(extra_attrs).cuda()
        # self.extra_objs = torch.LongTensor(extra_objs).cuda()
        # self.extra_pairs = torch.LongTensor(extra_pairs).cuda()

        #train extra attrs objs pairs
        self.train_extra_attrs = [dset.unique_attr2idx[attr] for attr in dset.train_attrs_extra]
        self.train_extra_objs = [dset.unique_obj2idx[obj] for obj in dset.train_objs_extra]
        train_extra_pairs = [dset.unique_pair2idx[pair] for pair in dset.train_pairs_extra]

        self.train_extra_attrs = torch.LongTensor(self.train_extra_attrs).cuda()
        self.train_extra_objs = torch.LongTensor(self.train_extra_objs).cuda()
        self.train_extra_pairs = torch.LongTensor(train_extra_pairs).cuda()


        #seen attrs objs pairs
        self.seen_attrs = dset.seen_attr_neighbors
        self.seen_objs = dset.seen_obj_neighbors
        self.seen_pairs = dset.seen_pairs_neighbors
        # self.train_attr_neigh, self.train_attr_neigh_sim = self.neighbor2index('attr',k=5)
        # self.train_obj_neigh, self.train_obj_neigh_sim = self.neighbor2index('obj', k=5)
        # self.train_pair_neigh, self.train_pair_neigh_sim = self.neighbor2index('pair', k=5)

        #feat_dim = 512
        #load CLIP model
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

        # self.ensemble_token_embeeding = torch.load("soft_embedding.pt").cuda()  #加载之前训练好的soft embedding

        self.accum_grad_soft_embedding = nn.Parameter(torch.zeros_like(self.soft_embedding), requires_grad=False)
        self.register_parameter('soft_embedding', self.soft_embedding)
        self.ao_dropout = nn.Dropout(cfg.MODEL.ao_dropout)

        # self.pair2attr_obj, self.attr_possible_compose, self.obj_possible_compose = self.construct_pair_to_ao_idx(dset.unique_pairs, dset.train_extra_pairs)
        self.pair2attr_obj, self.train_extra_pair2attr_obj = self.construct_pair_to_ao_idx(dset.unique_pairs, dset.train_pairs_extra)

        #构造attr和obj可能存在的组合关系
        attr_possible_compose = {}
        objs_possible_compose = {}
        for attr, obj in self.train_extra_pair2attr_obj:
            if attr not in attr_possible_compose:
                attr_possible_compose[attr] = []
            attr_possible_compose[attr].append(obj)

            if obj not in objs_possible_compose:
                objs_possible_compose[obj] = []
            objs_possible_compose[obj].append(attr)

        self.attr_possible_compose = attr_possible_compose
        self.obj_possible_compose = objs_possible_compose

        self.attr_compose_emb, self.obj_compose_emb = self.possible_compose_embeddings()

        self.all_pair2attr_obj = []
        for id, pair in enumerate(self.dset.pairs):
            self.all_pair2attr_obj.append((self.dset.attr2idx[pair[0]], self.dset.obj2idx[pair[1]]))


        self.classifier = CosineClassifier(temp=cfg.MODEL.cosine_cls_temp)
        self.label_smoothing = LabelSmoothingCrossEntropy_pair(cfg.MODEL.smoothing) # reduction='sum' #SmoothCrossEntropyLoss(smoothing=cfg.MODEL.smoothing)
        # self.image_pair_comparison = ImagePairComparison(
        #     cfg, self.num_attrs, self.num_objs, self.train_attrs1, self.train_objs1,
        #     self.train_attrs2, self.train_objs2, self.construct_token_tensors, self.construct_possible_token_tensors,
        #     img_dim=feat_dim,
        # )

        # ##引入 image feats (visual encoder is freezed)
        # sample_img_feats = torch.load('mit_average_sample_img_feats_10.pt')    ##(256*40, 512)
        # sample_img_feats = sample_img_feats.to(device)

        # ### extra 和 seen 的token embedding相似度矩阵
        train_attr_len = len(self.dset.train_attrs)
        train_obj_len = len(self.dset.train_objs)
        all_attr_len = len(self.dset.all_attrs)
        all_obj_len = len(self.dset.all_objs)

        seen_attr, seen_obj = self.init_token_embedding1[:train_attr_len].clone().detach(), self.init_token_embedding1[self.offset:self.offset+train_obj_len].clone().detach()
        # extra_attr, extra_obj = self.init_token_embedding1[all_attr_len: self.offset].clone().detach(), self.init_token_embedding1[self.offset+all_obj_len:].clone().detach()

        # # self.e2s_attr_sim = F.cosine_similarity(extra_attr.unsqueeze(1), seen_attr.unsqueeze(0), dim=2)
        # # self.e2s_obj_sim = F.cosine_similarity(extra_obj.unsqueeze(1), seen_obj.unsqueeze(0), dim=2)

        # # topk_attr_sim, self.topk_e2s_attr_indices = torch.topk(self.e2s_attr_sim, k=5, dim=1)
        # # topk_obj_sim, self.topk_e2s_obj_indices = torch.topk(self.e2s_obj_sim, k=5, dim=1)

        # # self.e2s_attr_weights = F.softmax(topk_attr_sim, dim=1)
        # # self.e2s_obj_weights = F.softmax(topk_obj_sim, dim=1)

        # # # ### extra 和 seen 的text embedding相似度矩阵
        # # # extra_attr_text_emb = self.token2text(extra_attr, 'attrs')
        # # # extra_obj_text_emb = self.token2text(extra_obj, 'objs')
        seen_attr_text_emb = self.token2text(seen_attr, 'attrs')
        seen_obj_text_emb = self.token2text(seen_obj, 'objs')

        # # self.e2s_attr_text_sim = F.cosine_similarity(extra_attr_text_emb.unsqueeze(1), seen_attr_text_emb.unsqueeze(0), dim=2)
        # # self.e2s_obj_text_sim = F.cosine_similarity(extra_obj_text_emb.unsqueeze(1), seen_obj_text_emb.unsqueeze(0), dim=2)

        # # topk_attr_text_sim, self.topk_e2s_attr_text_indices = torch.topk(self.e2s_attr_text_sim, k=5, dim=1)
        # # topk_obj_text_sim, self.topk_e2s_obj_text_indices = torch.topk(self.e2s_obj_text_sim, k=5, dim=1)

        # # self.e2s_attr_text_weights = F.softmax(topk_attr_text_sim, dim=1)
        # # self.e2s_obj_text_weights = F.softmax(topk_obj_text_sim, dim=1)

        # ### unseen 和 seen 的token embedding相似度矩阵
        unseen_attr, unseen_obj = self.init_token_embedding1[train_attr_len: all_attr_len], self.init_token_embedding1[self.offset+train_obj_len: self.offset+all_obj_len]
        self.u2s_attr_sim = F.cosine_similarity(unseen_attr.unsqueeze(1), seen_attr.unsqueeze(0), dim=2)
        self.u2s_obj_sim = F.cosine_similarity(unseen_obj.unsqueeze(1), seen_obj.unsqueeze(0), dim=2)

        topk_attr_sim, self.topk_u2s_attr_indices = torch.topk(self.u2s_attr_sim, k=5, dim=1)
        topk_obj_sim, self.topk_u2s_obj_indices = torch.topk(self.u2s_obj_sim, k=5, dim=1)

        self.u2s_attr_weights = F.softmax(topk_attr_sim, dim=1)
        self.u2s_obj_weights = F.softmax(topk_obj_sim, dim=1)

        # ### unseen 和 extra 的text embedding相似度矩阵
        # self.u2e_attr_sim = F.cosine_similarity(unseen_attr.unsqueeze(1), extra_attr.unsqueeze(0), dim=2)
        # self.u2e_obj_sim = F.cosine_similarity(unseen_obj.unsqueeze(1), extra_obj.unsqueeze(0), dim=2)

        # topk_attr_sim, self.topk_u2e_attr_indices = torch.topk(self.u2e_attr_sim, k=5, dim=1)
        # topk_obj_sim, self.topk_u2e_obj_indices = torch.topk(self.u2e_obj_sim, k=5, dim=1)

        # self.u2e_attr_weights = F.softmax(topk_attr_sim, dim=1)
        # self.u2e_obj_weights = F.softmax(topk_obj_sim, dim=1)
        self.tau = 0.1

        # ### unseen 和seen的text embedding相似度矩阵
        unseen_attr_text_emb = self.token2text(unseen_attr, 'attrs')
        unseen_obj_text_emb = self.token2text(unseen_obj, 'objs')

        self.u2s_attr_text_sim = F.cosine_similarity(unseen_attr_text_emb.unsqueeze(1), seen_attr_text_emb.unsqueeze(0), dim=2)
        self.u2s_obj_text_sim = F.cosine_similarity(unseen_obj_text_emb.unsqueeze(1), seen_obj_text_emb.unsqueeze(0), dim=2)

        self.topk_attr_text_sim, self.topk_u2s_attr_text_indices = torch.topk(self.u2s_attr_text_sim, k=3, dim=1)
        self.topk_obj_text_sim, self.topk_u2s_obj_text_indices = torch.topk(self.u2s_obj_text_sim, k=3, dim=1)

        self.u2s_attr_text_weights = F.softmax(self.topk_attr_text_sim / self.tau, dim=1)
        self.u2s_obj_text_weights = F.softmax(self.topk_obj_text_sim / self.tau, dim=1)



        init_sim_attr_text = F.cosine_similarity(seen_attr_text_emb.unsqueeze(1), seen_attr_text_emb.unsqueeze(0), dim=2)
        init_sim_obj_text = F.cosine_similarity(seen_obj_text_emb.unsqueeze(1), seen_obj_text_emb.unsqueeze(0), dim=2)
        init_sim_attr_text -= torch.eye(init_sim_attr_text.size(0)).to(device)
        init_sim_obj_text -= torch.eye(init_sim_obj_text.size(0)).to(device)

        self.init_sim_attr_text_topk, self.init_sim_attr_text_topk_indices = torch.topk(init_sim_attr_text, k=5, dim=1)
        self.init_sim_obj_text_topk, self.init_sim_obj_text_topk_indices = torch.topk(init_sim_obj_text, k=5, dim=1)





        # ## unseen 和 seen 经过 image作用的相似度矩阵
        # u2i_attr_img_sim = F.cosine_similarity(unseen_attr_text_emb.unsqueeze(1), sample_img_feats.unsqueeze(0) , dim=2)
        # u2i_obj_img_sim = F.cosine_similarity(unseen_obj_text_emb.unsqueeze(1), sample_img_feats.unsqueeze(0) , dim=2)
        # s2i_attr_img_sim = F.cosine_similarity(seen_attr_text_emb.unsqueeze(1), sample_img_feats.unsqueeze(0) , dim=2)
        # s2i_obj_img_sim = F.cosine_similarity(seen_obj_text_emb.unsqueeze(1), sample_img_feats.unsqueeze(0) , dim=2)

        # self.u2s_attr_img_sim = F.cosine_similarity(u2i_attr_img_sim.unsqueeze(1), s2i_attr_img_sim.unsqueeze(0), dim=2)
        # self.u2s_obj_img_sim = F.cosine_similarity(u2i_obj_img_sim.unsqueeze(1), s2i_obj_img_sim.unsqueeze(0), dim=2)

        # topk_attr_img_sim, self.topk_u2s_attr_img_indices = torch.topk(self.u2s_attr_img_sim, k=5, dim=1)
        # topk_obj_img_sim, self.topk_u2s_obj_img_indices = torch.topk(self.u2s_obj_img_sim, k=5, dim=1)

        # self.u2s_attr_img_weights = F.softmax(topk_attr_img_sim, dim=1)
        # self.u2s_obj_img_weights = F.softmax(topk_obj_img_sim, dim=1)

        # del u2i_attr_img_sim, u2i_obj_img_sim, s2i_attr_img_sim, s2i_obj_img_sim
        # torch.cuda.empty_cache()

        # ## extra 和 seen经过 image作用的相似度矩阵
        # e2i_attr_img_sim = F.cosine_similarity(extra_attr_text_emb.unsqueeze(1), sample_img_feats.unsqueeze(0) , dim=2)
        # e2i_obj_img_sim = F.cosine_similarity(extra_obj_text_emb.unsqueeze(1), sample_img_feats.unsqueeze(0) , dim=2)
        # s2i_attr_img_sim = F.cosine_similarity(seen_attr_text_emb.unsqueeze(1), sample_img_feats.unsqueeze(0) , dim=2)
        # s2i_obj_img_sim = F.cosine_similarity(seen_obj_text_emb.unsqueeze(1), sample_img_feats.unsqueeze(0) , dim=2)

        # self.e2s_attr_img_sim = F.cosine_similarity(e2i_attr_img_sim.unsqueeze(1)**5, s2i_attr_img_sim.unsqueeze(0)**5, dim=2)
        # self.e2s_obj_img_sim = F.cosine_similarity(e2i_obj_img_sim.unsqueeze(1)**5, s2i_obj_img_sim.unsqueeze(0)**5, dim=2)

        # topk_attr_img_sim, self.topk_e2s_attr_img_indices = torch.topk(self.e2s_attr_img_sim, k=5, dim=1)
        # topk_obj_img_sim, self.topk_e2s_obj_img_indices = torch.topk(self.e2s_obj_img_sim, k=5, dim=1)

        # self.e2s_attr_img_weights = F.softmax(topk_attr_img_sim, dim=1)
        # self.e2s_obj_img_weights = F.softmax(topk_obj_img_sim, dim=1)

        # del e2i_attr_img_sim, e2i_obj_img_sim

        # ##image 和text 混合作用
        # self.u2s_attr_cross_sim = self.u2s_attr_img_sim + self.u2s_attr_text_sim
        # self.u2s_obj_cross_sim = self.u2s_obj_img_sim + self.u2s_obj_text_sim

        # topk_attr_cross_sim, self.topk_u2s_attr_cross_indices = torch.topk(self.u2s_attr_cross_sim, k=5, dim=1)
        # topk_obj_cross_sim, self.topk_u2s_obj_cross_indices = torch.topk(self.u2s_obj_cross_sim, k=5, dim=1)

        # self.u2s_attr_cross_weights = F.softmax(topk_attr_cross_sim, dim=1)
        # self.u2s_obj_cross_weights = F.softmax(topk_obj_cross_sim, dim=1)

        # #imgae 和token 混合作用
        # self.u2s_attr_cross_sim = self.u2s_attr_img_sim + self.u2s_attr_sim
        # self.u2s_obj_cross_sim = self.u2s_obj_img_sim + self.u2s_obj_sim

        # topk_attr_cross_sim, self.topk_u2s_attr_cross_indices = torch.topk(self.u2s_attr_cross_sim, k=5, dim=1)
        # topk_obj_cross_sim, self.topk_u2s_obj_cross_indices = torch.topk(self.u2s_obj_cross_sim, k=5, dim=1)

        # self.u2s_attr_cross_weights = F.softmax(topk_attr_cross_sim, dim=1)
        # self.u2s_obj_cross_weights = F.softmax(topk_obj_cross_sim, dim=1)


        # del sample_img_feats, u2i_attr_img_sim, u2i_obj_img_sim, s2i_attr_img_sim, s2i_obj_img_sim
        # torch.cuda.empty_cache()

        # self.average_sample_img_feats = torch.zeros(len(self.train_pairs)*2, 512).cuda()
        # self.img_labels = torch.zeros(len(self.train_pairs)).cuda()


        # ### load similarity matrix
        # # similarity_matrix = torch.load('mit_sim_matrix.pt')  #mit-states
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


    def neighbor2index(self, mode, k=5):
        neighbors = {}
        neighbors_sim ={}
        def get_top_k_neighbors_and_softmax(sim_dict, index_mapping, k):
            # 将邻居属性和相似度值组成列表并按相似度排序（从大到小）
            sorted_neighbors = sorted(sim_dict.items(), key=lambda x: x[1], reverse=True)
            # 只保留前k个邻居
            top_k_neighbors = sorted_neighbors[:k]
            # 分别取出邻居和相似度
            top_k_neigh, top_k_sim = zip(*top_k_neighbors)
            # 获取邻居的索引
            neighbor_indices = [index_mapping[neigh] for neigh in top_k_neigh]
            # 对前k个相似度计算 softmax
            top_k_sim_softmax = np.exp(top_k_sim) / np.sum(np.exp(top_k_sim))

            return neighbor_indices, top_k_sim_softmax

        if mode == 'attr':
            for item in self.seen_attrs.keys():
                index = self.dset.unique_attr2idx[item]
                neighbors[index], neighbors_sim[index] = get_top_k_neighbors_and_softmax(
                    self.seen_attrs[item], self.dset.unique_attr2idx, k
                )
        elif mode == 'obj':
            for item in self.seen_objs.keys():
                index = self.dset.unique_obj2idx[item]
                neighbors[index], neighbors_sim[index] = get_top_k_neighbors_and_softmax(
                    self.seen_objs[item], self.dset.unique_obj2idx, k
                )
        elif mode == 'pair':
            for item in self.seen_pairs.keys():
                index = self.dset.unique_pair2idx[item]
                neighbors[index], neighbors_sim[index] = get_top_k_neighbors_and_softmax(
                    self.seen_pairs[item], self.dset.unique_pair2idx, k
                )

        return neighbors, neighbors_sim

    def initialize_token_embeddings(self, dset):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenized = clip.tokenize(dset, context_length=32).to(device)
        eos_ids = tokenized.argmax(dim=-1)

        tokenize_embedding = self.clip_model.token_embedding(tokenized)

        mean_embeddings = []

        for i, eos_id in enumerate(eos_ids):
            mean_embedding = torch.mean(tokenize_embedding[i, 1:eos_id, :], dim=0)
            mean_embeddings.append(mean_embedding)

        mean_embeddings = torch.stack(mean_embeddings)

        return mean_embeddings

    def possible_compose_embeddings(self):   # 115 or more
        attr_composed_token_embeddings = torch.zeros(len(self.dset.unique_attrs), self.soft_embedding.size(-1))
        obj_composed_token_embeddings = torch.zeros(len(self.dset.unique_objs), self.soft_embedding.size(-1))
        for attr_idx, possible_objs in self.attr_possible_compose.items():
            if len(possible_objs) > 0:
                attr_composed_token_embeddings[attr_idx] = torch.mean(self.init_token_embedding[possible_objs], dim=0)  #采用固定的token embedding
        for obj_idx, possible_attrs in self.obj_possible_compose.items():
            if len(possible_attrs) > 0:
                obj_composed_token_embeddings[obj_idx] = torch.mean(self.init_token_embedding[[item + self.offset for item in possible_attrs]], dim=0)
        return attr_composed_token_embeddings, obj_composed_token_embeddings

    def construct_pair_to_ao_idx(self, unique_pairs, train_extra_pairs):
        pair2attr_obj = []
        #构造pair到attr和obj的映射
        for idx, pair in enumerate(unique_pairs):
            pair2attr_obj.append((self.dset.unique_attr2idx[pair[0]], self.dset.unique_obj2idx[pair[1]]))

        train_extra_pair2attr_obj = []
        for idx, pair in enumerate(train_extra_pairs):
            train_extra_pair2attr_obj.append((self.dset.unique_attr2idx[pair[0]], self.dset.unique_obj2idx[pair[1]]))

        return pair2attr_obj, train_extra_pair2attr_obj
        # return pair2attr_obj, attr_possible_compose, objs_possible_compose


    def construct_token_tensors(self, idx, type):
        # soft_embedding = self.ao_dropout(self.soft_embedding)
        train_attr_len = len(self.dset.train_attrs)
        train_obj_len = len(self.dset.train_objs)
        train_attr = self.ao_dropout(self.soft_embedding[:train_attr_len])
        train_obj = self.ao_dropout(self.soft_embedding[self.offset:self.offset+train_obj_len])
        soft_embedding = torch.cat(
            (train_attr, self.soft_embedding[train_attr_len:self.offset],train_obj, self.soft_embedding[self.offset+train_obj_len:]), dim=0
        )

        # soft_embedding = self.soft_embedding
        if 'attrs' in type:
            eos_idx = int(self.attr_tempelete_token_id.argmax())
            token_id = self.attr_tempelete_token_id.repeat(len(idx), 1)
        elif 'objs' in type:
            eos_idx = int(self.obj_tempelete_token_id.argmax())
            token_id = self.obj_tempelete_token_id.repeat(len(idx), 1)
        elif 'pairs' in type:
            eos_idx = int(self.pair_tempelete_token_id.argmax())
            token_id = self.pair_tempelete_token_id.repeat(len(idx), 1)
        elif "attr+obj" in type or "obj+attr" in type:
            eos_idx = int(self.pair_tempelete_token_id.argmax())
            token_id = self.pair_tempelete_token_id.repeat(len(idx[0]), 1)

        token_embedding = self.clip_model.token_embedding(token_id)

        if type == 'attrs':
            token_embedding[:, eos_idx-2, :] = soft_embedding[idx]
        elif type =='extra_attrs':
            token_embedding[:, eos_idx-2, :] = self.init_token_embedding[idx]
        elif type =='proto_attrs':
            token_embedding[:, eos_idx-2, :] = self.kmean_attr_centers
        elif type =='objs':
            token_embedding[:, eos_idx-1, :] = soft_embedding[idx+self.offset]
        elif type == 'proto_objs':
            token_embedding[:, eos_idx-1, :] = self.kmean_obj_centers
        elif type == 'extra_objs':
            token_embedding[:, eos_idx-1, :] = self.init_token_embedding[idx+self.offset]
        elif type == 'pairs':
            #todo：change the pair_id to attr_id and obj_id
            attr_id = [self.pair2attr_obj[pair][0] for pair in idx]
            obj_id = [self.pair2attr_obj[pair][1] for pair in idx]
            token_embedding[:, eos_idx-2, :] = soft_embedding[attr_id]
            token_embedding[:, eos_idx-1, :] = soft_embedding[[id + self.offset for id in obj_id]]
        elif type == 'extra_pairs':
            attr_id = [self.pair2attr_obj[pair][0] for pair in idx]
            obj_id = [self.pair2attr_obj[pair][1] for pair in idx]
            token_embedding[:, eos_idx-2, :] = self.init_token_embedding[attr_id]
            token_embedding[:, eos_idx-1, :] = self.init_token_embedding[[id + self.offset for id in obj_id]]
        elif type == "attr+obj":
            attr_id, obj_id = idx
            token_embedding[:, eos_idx-2, :] = self.soft_embedding[attr_id]
            token_embedding[:, eos_idx-1, :] = self.init_token_embedding[[id + self.offset for id in obj_id]]
        elif type == "obj+attr":
            attr_id, obj_id = idx
            token_embedding[:, eos_idx-2, :] = self.init_token_embedding[attr_id]
            token_embedding[:, eos_idx-1, :] = self.soft_embedding[[id + self.offset for id in obj_id]]

        text_emb = self.clip.text_encoder(token_id, token_embedding, enable_pos_emb=True)

        return text_emb

    # def ensemble_construct_token_tensors(self, idx):
    #     soft_embedding = self.ensemble_token_embeeding
    #     eos_idx = int(self.pair_tempelete_token_id.argmax())
    #     token_id = self.pair_tempelete_token_id.repeat(len(idx), 1)

    #     token_embedding = self.clip_model.token_embedding(token_id)

    #     attr_id = [self.pair2attr_obj[pair][0] for pair in idx]
    #     obj_id = [self.pair2attr_obj[pair][1] for pair in idx]
    #     token_embedding[:, eos_idx-2, :] = soft_embedding[attr_id]
    #     token_embedding[:, eos_idx-1, :] = soft_embedding[[id + self.offset for id in obj_id]]

    #     text_emb = self.clip.text_encoder(token_id, token_embedding, enable_pos_emb=True)

    #     return text_emb

    def extract_train_extra_embeddings(self):

        train_extra_attr = self.dset.train_attrs_extra
        train_extra_obj = self.dset.train_objs_extra
        #map train_extra to unique
        train_extra_attr = [self.dset.unique_attr2idx[attr] for attr in train_extra_attr]
        train_extra_obj = [self.dset.unique_obj2idx[obj] for obj in train_extra_obj]
        train_extra_attr = torch.LongTensor(train_extra_attr).cuda()
        train_extra_obj = torch.LongTensor(train_extra_obj).cuda()

        #map train_extra to token embedding
        train_extra_attr_token_emb = self.soft_embedding[train_extra_attr]
        train_extra_obj_token_emb = self.soft_embedding[train_extra_obj + self.offset]

        #extract train_extra word embedding
        attr_eos_idx = int(self.attr_tempelete_token_id.argmax())
        attr_token_id = self.attr_tempelete_token_id.repeat(train_extra_attr_token_emb.shape[0], 1)

        obj_eos_idx = int(self.obj_tempelete_token_id.argmax())
        obj_token_id = self.obj_tempelete_token_id.repeat(train_extra_obj_token_emb.shape[0], 1)

        attr_token_embedding = self.clip_model.token_embedding(attr_token_id)
        obj_token_embedding = self.clip_model.token_embedding(obj_token_id)

        #exchange x in tempelete to train_extra token embedding
        attr_token_embedding[:, attr_eos_idx-2, :] = train_extra_attr_token_emb
        obj_token_embedding[:, obj_eos_idx-1, :] = train_extra_obj_token_emb

        attr_text_emb = self.clip.text_encoder(attr_token_id, attr_token_embedding, enable_pos_emb=True)
        obj_text_emb = self.clip.text_encoder(obj_token_id, obj_token_embedding, enable_pos_emb=True)

        if not os.path.exists('outputs'):
            os.makedirs('outputs')

        torch.save(train_extra_attr_token_emb, 'outputs/train_extra_attr_token_emb.pt')
        torch.save(train_extra_obj_token_emb, 'outputs/train_extra_obj_token_emb.pt')
        torch.save(attr_text_emb, 'outputs/train_extra_attr_text_emb.pt')
        torch.save(obj_text_emb, 'outputs/train_extra_obj_text_emb.pt')

        #extract unique word embedding
        attr_token_id = self.attr_tempelete_token_id.repeat(self.offset, 1)
        obj_token_id = self.obj_tempelete_token_id.repeat(self.soft_embedding.shape[0]-self.offset, 1)
        attr_token_embedding = self.clip_model.token_embedding(attr_token_id)
        obj_token_embedding = self.clip_model.token_embedding(obj_token_id)

        attr_token_embedding[:, attr_eos_idx-2, :] = self.soft_embedding[:self.offset, :]
        obj_token_embedding[:, obj_eos_idx-1, :] = self.soft_embedding[self.offset:, :]

        unique_attr_text_emb = self.clip.text_encoder(attr_token_id, attr_token_embedding, enable_pos_emb=True)
        unique_obj_text_emb = self.clip.text_encoder(obj_token_id, obj_token_embedding, enable_pos_emb=True)

        torch.save(self.soft_embedding, 'outputs/soft_embedding.pt')
        torch.save(unique_attr_text_emb, 'outputs/unique_attr_text_emb.pt')
        torch.save(unique_obj_text_emb, 'outputs/unique_obj_text_emb.pt')

        return

    def extract_unique_pair_text_emb(self):
        # 获取 eos 的索引
        unique_pair_eos_idx = int(self.pair_tempelete_token_id.argmax())
        # 获取所有 unique pair 的 token id
        unique_pair_token_id = self.pair_tempelete_token_id.repeat(len(self.dset.unique_pairs), 1)
        # 提前生成所有的 token embedding
        unique_pair_token_embedding = self.clip_model.token_embedding(unique_pair_token_id)

        # 获取 attr 和 obj ids
        attr_ids = [item[0] for item in self.pair2attr_obj]
        obj_ids = [item[1] for item in self.pair2attr_obj]

        # 替换指定位置的 embedding
        unique_pair_token_embedding[:, unique_pair_eos_idx-2, :] = self.soft_embedding[attr_ids]
        unique_pair_token_embedding[:, unique_pair_eos_idx-1, :] = self.soft_embedding[[item + self.offset for item in obj_ids]]

        all_unique_pair_text_embs = []
        batch_size = 1024
        num_batches = (len(self.dset.unique_pairs) + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(self.dset.unique_pairs))

            # 从预先生成的 unique_pair_token_id 和 unique_pair_token_embedding 中获取当前 batch 的数据
            batch_token_id = unique_pair_token_id[start_idx:end_idx]
            batch_token_embedding = unique_pair_token_embedding[start_idx:end_idx]

            # 生成当前 batch 的 text embedding
            batch_unique_pair_text_emb = self.clip.text_encoder(batch_token_id, batch_token_embedding, enable_pos_emb=True)

            # 将当前 batch 的结果保存到列表中
            all_unique_pair_text_embs.append(batch_unique_pair_text_emb)

        # 将所有 batch 的 embedding 拼接在一起
        unique_pair_text_emb = torch.cat(all_unique_pair_text_embs, dim=0)

        # 保存结果
        torch.save(unique_pair_text_emb, 'outputs/unique_pair_text_emb.pt')

        return

    def construct_possible_token_tensors(self, idx, type):
        #构造可能的组合的token tensor
        eos_idx = int(self.pair_tempelete_token_id.argmax())
        token_id = self.pair_tempelete_token_id.repeat(len(idx), 1)
        token_embedding = self.clip_model.token_embedding(token_id)
        if type == 'attrs':
            token_embedding[:, eos_idx-2, :] = self.soft_embedding[idx] #self.init_token_embedding[idx]
            token_embedding[:, eos_idx-1, :] = self.attr_compose_emb[idx]
        elif type =='objs':
            token_embedding[:, eos_idx-1, :] = self.soft_embedding[idx] #self.init_token_embedding[idx+self.offset]
            token_embedding[:, eos_idx-2, :] = self.obj_compose_emb[idx]
        else:
            raise ValueError(f"Invalid type {type}")
        text_emb = self.clip.text_encoder(token_id, token_embedding, enable_pos_emb=True)

        return text_emb


    def extra_neigh_concept(self):
        # soft_embedding = self.ao_dropout(self.soft_embedding)
        soft_embedding = self.soft_embedding
        eos_idx = int(self.pair_tempelete_token_id.argmax())
        token_id = self.pair_tempelete_token_id.repeat(len(self.train_pairs), 1)
        token_embedding = self.clip_model.token_embedding(token_id)

        attr_id = [self.pair2attr_obj[pair][0] for pair in self.train_pairs]
        obj_id = [self.pair2attr_obj[pair][1] for pair in self.train_pairs]
        # attr_id = self.train_attrs
        # obj_id = self.train_objs
        extra_attr_token_embedding = torch.zeros((len(self.train_attr_neigh), self.soft_embedding.size(-1)))
        extra_obj_token_embedding = torch.zeros((len(self.train_obj_neigh), self.soft_embedding.size(-1)))
        for i in self.train_attr_neigh.keys():
            extra_attr_token_embedding[i,:] = sum(
                [self.init_token_embedding[it] * sim \
                    for it,sim in zip(self.train_attr_neigh[i], self.train_attr_neigh_sim[i])
                    ]
            )
        for i in self.train_obj_neigh.keys():
            extra_obj_token_embedding[i,:] = sum(
                [self.init_token_embedding[it + self.offset] * sim \
                    for it,sim in zip(self.train_obj_neigh[i], self.train_obj_neigh_sim[i])
                    ]
            )

        if not os.path.exists(self.cfg.config_name+'_extra_pair_word_emb.pkl'):
            extra_pair_token_embedding = torch.zeros((len(self.train_pair_neigh), 2, self.soft_embedding.size(-1)))

            for i,item in enumerate(self.train_pair_neigh.keys()):
                extra_pair_token_embedding[i, 0, :] = sum(
                    [self.init_token_embedding[self.pair2attr_obj[it][0]] * sim \
                        for it,sim in zip(self.train_pair_neigh[item], self.train_pair_neigh_sim[item])]
                )
                extra_pair_token_embedding[i, 1, :] = sum(
                    [self.init_token_embedding[self.pair2attr_obj[it][1] + self.offset]* sim \
                        for it,sim in zip(self.train_pair_neigh[item], self.train_pair_neigh_sim[item])]
                )

            extra_pair_token_embedding = extra_pair_token_embedding.to(soft_embedding.device)
            extra_pair_token_embedding = torch.cat([token_embedding[:, :eos_idx-2, :], \
                                                    extra_pair_token_embedding, \
                                                    token_embedding[:, eos_idx: , :]], \
                                                    dim=1)
            extra_text_emb = self.clip.text_encoder(token_id, extra_pair_token_embedding, enable_pos_emb=True)
            torch.save(extra_text_emb, self.cfg.config_name+'_extra_pair_word_emb.pkl')
        else:
            extra_text_emb = torch.load(self.cfg.config_name+'_extra_pair_word_emb.pkl')
        self.extra_text_emb = extra_text_emb

        token_embedding[:, eos_idx-2, :] = soft_embedding[attr_id]
        token_embedding[:, eos_idx-1, :] = soft_embedding[[id + self.offset for id in obj_id]]

        extra_attr_token_embedding = extra_attr_token_embedding[attr_id].to(soft_embedding.device)
        extra_obj_token_embedding = extra_obj_token_embedding[obj_id].to(soft_embedding.device)

        token_embedding[:, eos_idx-2, :] = 0.9*token_embedding[:, eos_idx-2, :] + 0.1 * extra_attr_token_embedding
        token_embedding[:, eos_idx-1, :] = 0.9*token_embedding[:, eos_idx-1, :] + 0.1 * extra_obj_token_embedding

        text_emb = self.clip.text_encoder(token_id, token_embedding, enable_pos_emb=True)

        text_emb = 0.9*text_emb + 0.1*extra_text_emb

        return text_emb
    def compute_TAL_perlabel(self, scores, labels, tau, margin):
        mask = 1 - labels

        alpha_i2t =((scores/tau).exp()* labels / ((scores/tau).exp()* labels).sum(dim=1, keepdim=True)).detach()
        #alpha_t2i = ((scores.t()/tau).exp()* labels / ((scores.t()/tau).exp()* labels).sum(dim=1, keepdim=True)).detach()

        loss = (-  (alpha_i2t*scores).sum(1) + tau * ((scores / tau).exp() * mask).sum(1).clamp(max=10e35).log() + margin).clamp(min=0)
            #+  (-  (alpha_t2i*scores.t()).sum(1) + tau * ((scores.t() / tau).exp() * mask).sum(1).clamp(max=10e35).log() + margin).clamp(min=0)
        return loss

    def compute_TRL_per(self, scores, pid, margin = 0.2, tau=0.02):
        batch_size = scores.shape[0]
        pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
        pid_dist = pid - pid.t()
        labels = (pid_dist == 0).float().cuda()

        mask = 1 - labels

        alpha_1 =((scores/tau).exp()* labels / ((scores/tau).exp()* labels).sum(dim=1, keepdim=True)).detach()
        alpha_2 = ((scores.t()/tau).exp()* labels / ((scores.t()/tau).exp()* labels).sum(dim=1, keepdim=True)).detach()

        pos_1 = (alpha_1 * scores).sum(1)
        pos_2 = (alpha_2 * scores.t()).sum(1)

        neg_1 = (mask*scores).max(1)[0]
        neg_2 = (mask*scores.t()).max(1)[0]

        cost_1 = (margin + neg_1 - pos_1).clamp(min=0)
        cost_2 = (margin + neg_2 - pos_2).clamp(min=0)
        return cost_1 + cost_2

    def l2norm(self, x, dim):
        return F.normalize(x, p=2, dim=dim)

    def train_forward(self, batch):

        img1 = batch['img']
        # img2_a = batch['img1_a'] # Image that shares the same attribute
        # img2_o = batch['img1_o'] # Image that shares the same object

        # Labels of 1st image.
        attr_labels = batch['attr']
        obj_labels = batch['obj']
        pair_labels = batch['pair']

        # attr2_labels_a = batch['attr1_a'] # attr labels of 2nd image
        # obj2_labels_a = batch['obj1_a'] # obj labels of 2nd image

        # attr2_labels_o = batch['attr1_o'] # attr labels of 3rd image
        # obj2_labels_o = batch['obj1_o'] # obj labels of 3rd image

        # composed_unseen_pair = batch['composed_unseen_pair']
        # composed_seen_pair = batch['composed_seen_pair']

        # #use for text embedding
        # at_neigh = {'n1': batch['at1'], 'n2': batch['at2'], 'n3': batch['at3'], 'n4':batch['at4'], 'n5': batch['at5']}
        # ob_neigh = {'n1':batch['ob1'], 'n2':batch['ob2'], 'n3': batch['ob3'], 'n4':batch['ob4'], 'n5': batch['ob5']}
        # pair_neigh = {'n1':batch['lbl1'], 'n2':batch['lbl2'], 'n3': batch['lbl3'], 'n4':batch['lbl4'], 'n5': batch['lbl5']}
        # #use for label
        # at_neigh_L = {'n1': batch['at1_L'], 'n2': batch['at2_L'], 'n3': batch['at3_L'], 'n4':batch['at4_L'], 'n5': batch['at5_L']}
        # ob_neigh_L = {'n1':batch['ob1_L'], 'n2':batch['ob2_L'], 'n3': batch['ob3_L'], 'n4':batch['ob4_L'], 'n5': batch['ob5_L']}
        # pair_neigh_L = {'n1':batch['lbl1_L'], 'n2':batch['lbl2_L'], 'n3': batch['lbl3_L'], 'n4':batch['lbl4_L'], 'n5': batch['lbl5_L']}

        # mask_task = batch['mask_task']
        bs = img1.shape[0]

        if self.cfg.TRAIN.sample_negative_pairs != -1:
            pool_of_pairs = batch['pool_of_pairs'] # [bs, n_pool].
            # We explicitly set positive label at index 0 (look at DataLoader code).
            pair_labels = torch.zeros(bs).to(img1.device).long()
        else:
            pool_of_pairs = None
        if self.cfg.MODEL.use_extra_pair_loss:
            # concept = self.compose_word_embeddings(mode='train_extra', pool_of_pairs=pool_of_pairs)
            #[6148,300]
            # concept_train_only = self.compose_word_embeddings(mode='train', pool_of_pairs=pool_of_pairs)

            concept = self.construct_token_tensors(self.train_extra_pairs, 'extra_pairs')
            concept_train_only = self.extra_neigh_concept()
            # concept_train_only = concept
        else:
            # concept = self.compose_word_embeddings( mode='train', pool_of_pairs=pool_of_pairs) # [501,512,300](n_pairs, emb_dim) or (bs, n_pairs, emb_dim)

            concept = self.construct_token_tensors(self.train_pairs, 'pairs')
            # concept = self.extra_neigh_concept()
            concept_train_only = concept
        # pool_of_pairs = None
        # concept = self.compose_word_embeddings(
        #         mode='train', pool_of_pairs=pool_of_pairs) # [501,512,300](n_pairs, emb_dim) or (bs, n_pairs, emb_dim)
        # concept_train_only = concept

        #CLIP image encoder
        img1, patch1 = self.clip.encode_image(img1.half())   #B x 49 x768
        img1, patch1 = img1.float(), patch1.float()

        # ###sample image features
        # for i,it in enumerate(pair_labels):
        #     if self.img_labels[it]<1:
        #         idx = int(it + self.img_labels[it].item())
        #         self.average_sample_img_feats[idx] = img1[i]
        #         self.img_labels[it] += 1
        #     else:
        #         continue
        # ### end

        # img2_a, patch2_a = self.clip.encode_image(img2_a.half())
        # img2_a, patch2_a = img2_a.float(), patch2_a.float()

        # img2_o, patch2_o = self.clip.encode_image(img2_o.half())
        # img2_o, patch2_o = img2_o.float(), patch2_o.float()


        # aux_loss = self.image_pair_comparison(img1, img2_a, img2_o, attr_labels, obj_labels, at_neigh_L, ob_neigh_L, mask_task)
        # aux_loss = self.image_pair_comparison(patch1, patch2_a, patch2_o, attr_labels, obj_labels, at_neigh_L, ob_neigh_L, mask_task)
        aux_loss = None

        pred = self.classifier(img1, concept_train_only)
        # pred_extra = self.classifier(img1, concept_train_only) #self.classifier(img1, concept)
        # pred_extra1 = self.classifier(img1, concept)

        zs_concept = self.construct_token_tensors(self.train_pairs, 'extra_pairs')
        zs_pred = self.classifier(img1, zs_concept)
        t = 1.0
        zs_pred = F.softmax(zs_pred / t, dim=-1)
        kl_loss = -zs_pred * F.log_softmax(pred / t, dim=-1) * t * t
        kl_loss = kl_loss.sum(1).mean()

        # pred_prob = pred / pred.sum(dim=-1, keepdim=True)
        # target_prob = zs_pred / zs_pred.sum(dim=-1, keepdim=True)
        # kl_loss = F.kl_div(pred_prob.log(), target_prob, reduction='batchmean')  # 对每一行求平均


        if pool_of_pairs is None:
            pair_loss = F.cross_entropy(pred, pair_labels)
            # loss1 = pair_loss * (1.0 - self.cfg.MODEL.extra_pair_loss_ratio)
            loss1 = pair_loss

            pred = torch.max(pred, dim=1)[1]
            attr_pred = self.train_attrs[pred]
            obj_pred = self.train_objs[pred]

            correct_attr = (attr_pred == attr_labels)
            correct_obj = (obj_pred == obj_labels)
            correct_pair = (pred == pair_labels)

            # out = { 'loss_total': pair_loss, 'acc_pair': torch.div(correct_pair.sum(),float(bs)) }

            #return out
        else:
            pair_loss = F.cross_entropy(pred, pair_labels)
            loss1 = pair_loss * (1.0 - self.cfg.MODEL.extra_pair_loss_ratio) #* self.cfg.MODEL.w_loss_main

            pred = torch.max(pred, dim=1)[1] # (bs)
            true_pair_labels = torch.gather(pool_of_pairs, 1, pred.unsqueeze(1)).squeeze(1) # (bs)
            attr_pred = self.train_attrs[true_pair_labels]
            obj_pred = self.train_objs[true_pair_labels]

            correct_attr = (attr_pred == attr_labels)
            correct_obj = (obj_pred == obj_labels)
            correct_pair = (pred == pair_labels)


        if self.cfg.MODEL.use_extra_pair_loss:
            sel_ind = pred_extra1.gather(1, pair_labels.long().view(-1,1)).squeeze()
            n1 = pred_extra1.gather(1, batch['lbl1_L'].long().view(-1,1)).squeeze()
            n2 = pred_extra1.gather(1, batch['lbl2_L'].long().view(-1,1)).squeeze()
            n3 = pred_extra1.gather(1, batch['lbl3_L'].long().view(-1,1)).squeeze()
            n4 = pred_extra1.gather(1, batch['lbl4_L'].long().view(-1,1)).squeeze()
            n5 = pred_extra1.gather(1, batch['lbl5_L'].long().view(-1,1)).squeeze()

            #5neigh
            new_pred = torch.cat([pred_extra,n1.unsqueeze(1),n2.unsqueeze(1),n3.unsqueeze(1),n4.unsqueeze(1),n5.unsqueeze(1)], axis=-1)

            #3neigh
            # new_pred = torch.cat([pred_extra,n1.unsqueeze(1),n2.unsqueeze(1),n3.unsqueeze(1)], axis=-1)
            # 1 neigh
            # new_pred = torch.cat([pred_extra,n1.unsqueeze(1)], axis=-1)


            pair_loss_ex = self.label_smoothing(new_pred, pair_labels)

            loss1 += pair_loss_ex * self.cfg.MODEL.extra_pair_loss_ratio

        loss = loss1 * self.cfg.MODEL.w_loss_main


        # extra_attr_idx = torch.load("train_extra_sim_data/topk_attr_idx.pt").to(attr_labels.device)
        # extra_obj_idx = torch.load("train_extra_sim_data/topk_obj_idx.pt").to(obj_labels.device)

        # sample_num = 5
        # extra_attr_idx = extra_attr_idx[:,:5*sample_num]
        # extra_obj_idx = extra_obj_idx[:,:5*sample_num]
        # extra_attr_idx = extra_attr_idx[:, torch.randperm(extra_attr_idx.size(1))[:sample_num]]
        # extra_obj_idx = extra_obj_idx[:, torch.randperm(extra_attr_idx.size(1))[:sample_num]]
        # extra_attr = extra_attr_idx[attr_labels].reshape(bs*sample_num)
        # extra_obj= extra_obj_idx[obj_labels].reshape(bs*sample_num)

        # extra_concept_attr = self.construct_token_tensors((attr_labels.repeat_interleave(sample_num), extra_obj), "attr+obj")
        # extra_concept_objs = self.construct_token_tensors((extra_attr, obj_labels.repeat_interleave(sample_num)), "obj+attr")
        # pair_concept = self.construct_token_tensors(pair_labels, "pairs")
        # sim1 = torch.einsum("ad,bd->ab", self.l2norm(pair_concept,-1), self.l2norm(extra_concept_attr,-1))
        # sim2 = torch.einsum("ad,bd->ab", self.l2norm(pair_concept,-1), self.l2norm(extra_concept_objs,-1))
        # extra_label = torch.ones(sim1.shape[0]).long().to(sim1.device)
        # extra_attr_loss  = F.cross_entropy(sim1, extra_label)
        # extra_obj_loss = F.cross_entropy(sim2, extra_label)

        # loss += 0.001*extra_attr_loss
        # loss += 0.001*extra_obj_loss
        # print(f"loss:{loss} extra_attr_loss:{0.001*extra_attr_loss} extra_obj_loss:{0.001*extra_obj_loss}")

        # ###pair extra loss
        # extra_pair_idx = torch.load("train_extra_sim_data/topk_pair_idx.pt").to(pair_labels.device)
        # pos_neg_num = 10
        # pos_pair_idx, neg_pair_idx = extra_pair_idx[:,:5*pos_neg_num], extra_pair_idx[:,5*pos_neg_num:10*pos_neg_num]
        # pos_pair_idx = pos_pair_idx[:, torch.randperm(pos_pair_idx.size(1))[:pos_neg_num]]
        # neg_pair_idx = neg_pair_idx[:, torch.randperm(neg_pair_idx.size(1))[:pos_neg_num]]
        # train_extra_feats = self.construct_token_tensors(self.train_extra_pairs, 'extra_pairs')
        # extra_feats = train_extra_feats[len(self.train_pairs):]
        # pos_extra_feats, neg_extra_feats = extra_feats[pos_pair_idx], extra_feats[neg_pair_idx]
        # pos_neg_pair = torch.cat((pos_extra_feats, neg_extra_feats), dim=1)
        # pos_neg_pair = torch.einsum("ad,abd->ab", self.l2norm(concept,-1), self.l2norm(pos_neg_pair,-1))
        # pos_neg_pair_labels = torch.cat((torch.ones(len(self.train_pairs), pos_neg_num), torch.zeros(len(self.train_pairs), pos_neg_num)), dim=1).to(pair_labels.device)
        # # pos_neg_pair_labels = torch.cat((torch.zeros(len(self.train_pairs), pos_neg_num), torch.zeros(len(self.train_pairs), pos_neg_num)), dim=1).to(pair_labels.device)
        # pair_loss_extra = torch.mean(F.binary_cross_entropy_with_logits(pos_neg_pair, pos_neg_pair_labels, reduction='none'))
        # # pair_loss_extra = self.compute_TAL_perlabel(pos_neg_pair, pos_neg_pair_labels, tau=0.1, margin=0.3).mean()
        # print(f"loss:{loss} pair_loss_extra:{pair_loss_extra}")
        # loss += 0.001*pair_loss_extra
        # ### END

        # ###attr obj extra loss
        # train_attr_len = len(self.dset.train_attrs)
        # train_obj_len = len(self.dset.train_objs)

        # all_attr_len = len(self.dset.all_attrs)
        # all_obj_len = len(self.dset.all_objs)

        # unique_attr_len = len(self.dset.unique_attrs)
        # unique_obj_len = len(self.dset.unique_objs)

        # extra_attr_idx = torch.load("train_extra_sim_data/topk_attr_idx.pt").to(attr_labels.device)
        # extra_obj_idx = torch.load("train_extra_sim_data/topk_obj_idx.pt").to(obj_labels.device)

        # pos_neg_num = 10
        # pos_attr_idx, neg_attr_idx = extra_attr_idx[:,:5*pos_neg_num], extra_attr_idx[:,5*pos_neg_num:10*pos_neg_num]
        # pos_obj_idx, neg_obj_idx = extra_obj_idx[:,:5*pos_neg_num], extra_obj_idx[:,5*pos_neg_num:10*pos_neg_num]

        # pos_attr_idx, neg_attr_idx = pos_attr_idx[:, torch.randperm(pos_attr_idx.size(1))[:pos_neg_num]], neg_attr_idx[:, torch.randperm(neg_attr_idx.size(1))[:pos_neg_num]]
        # pos_obj_idx, neg_obj_idx = pos_obj_idx[:, torch.randperm(pos_obj_idx.size(1))[:pos_neg_num]], neg_obj_idx[:, torch.randperm(neg_obj_idx.size(1))[:pos_neg_num]]

        # train_attrs = torch.arange(train_attr_len).to(attr_labels.device)
        # train_attr_feats = self.construct_token_tensors(train_attrs, 'attrs')
        # # train_extra_attr_feats = self.construct_token_tensors(self.train_extra_attrs, 'extra_attrs')
        # train_extra_attr_feats = self.construct_token_tensors(self.train_extra_attrs, 'attrs')

        # train_objs = torch.arange(train_obj_len).to(obj_labels.device)
        # train_obj_feats = self.construct_token_tensors(train_objs, 'objs')
        # # train_extra_obj_feats = self.construct_token_tensors(self.train_extra_objs, 'extra_objs')
        # train_extra_obj_feats = self.construct_token_tensors(self.train_extra_objs, 'objs')

        # extra_attr_feat = train_extra_attr_feats[train_attr_len:]
        # extra_obj_feat = train_extra_obj_feats[train_obj_len:]
        # pos_extra_attr_feat, neg_extra_attr_feat = extra_attr_feat[pos_attr_idx], extra_attr_feat[neg_attr_idx]
        # pos_extra_obj_feat, neg_extra_obj_feat = extra_obj_feat[pos_obj_idx], extra_obj_feat[neg_obj_idx]

        # pos_neg_attr = torch.cat((pos_extra_attr_feat, neg_extra_attr_feat), dim=1)
        # pos_neg_attr = torch.einsum("ad,abd->ab", self.l2norm(train_attr_feats.float(), -1), self.l2norm(pos_neg_attr.float(),-1))

        # pos_neg_obj = torch.cat((pos_extra_obj_feat, neg_extra_obj_feat), dim=1)
        # pos_neg_obj = torch.einsum("ad,abd->ab", self.l2norm(train_obj_feats.float(), -1), self.l2norm(pos_neg_obj.float(),-1))

        # pos_neg_attr_labels = torch.cat((torch.ones(train_attr_len, 10), torch.zeros(train_attr_len, 10)), dim=1).to(attr_labels.device)
        # pos_neg_obj_labels = torch.cat((torch.ones(train_obj_len, 10), torch.zeros(train_obj_len, 10)), dim=1).to(obj_labels.device)

        # # pos_neg_attr_labels = torch.cat((torch.zeros(train_attr_len, 10), torch.zeros(train_attr_len, 10)), dim=1).to(attr_labels.device)
        # # pos_neg_obj_labels = torch.cat((torch.zeros(train_obj_len, 10), torch.zeros(train_obj_len, 10)), dim=1).to(obj_labels.device)

        # attr_loss_extra = torch.mean(F.binary_cross_entropy_with_logits(pos_neg_attr, pos_neg_attr_labels, reduction='none'))
        # obj_loss_extra = torch.mean(F.binary_cross_entropy_with_logits(pos_neg_obj, pos_neg_obj_labels, reduction='none'))
        # # attr_loss_extra = self.compute_TAL_perlabel(pos_neg_attr, pos_neg_attr_labels, tau=0.1, margin=0.3).mean()
        # # obj_loss_extra = self.compute_TAL_perlabel(pos_neg_obj, pos_neg_obj_labels, tau=0.1, margin=0.3).mean()

        # # print(f"loss:{loss} attr_loss_extra:{attr_loss_extra} obj_loss_extra:{obj_loss_extra}")
        # loss += attr_loss_extra
        # loss += obj_loss_extra
        # ###END


        if self.cfg.MODEL.use_attr_loss:
            loss_attr = (1.0 - self.cfg.MODEL.extra_attr_loss_ratio) * aux_loss['loss_attr']
            if self.cfg.MODEL.use_extra_attr_loss and self.cfg.MODEL.extra_attr_loss_ratio > 0.0:
                loss_attr += self.cfg.MODEL.extra_attr_loss_ratio * aux_loss['loss_attr_ex']
            loss += loss_attr * self.cfg.MODEL.w_loss_attr

        if self.cfg.MODEL.use_obj_loss:
            loss_obj = (1.0 - self.cfg.MODEL.extra_obj_loss_ratio) * aux_loss['loss_obj']
            if self.cfg.MODEL.use_extra_obj_loss and self.cfg.MODEL.extra_obj_loss_ratio > 0.0:
                loss_obj += aux_loss['loss_obj_ex'] * self.cfg.MODEL.extra_obj_loss_ratio
            loss += loss_obj * self.cfg.MODEL.w_loss_obj


        out = {
            'loss_total': loss,
            'ce_loss': pair_loss,
            'kl_loss': kl_loss,
            'acc_attr': torch.div(correct_attr.sum(),float(bs)),
            'acc_obj': torch.div(correct_obj.sum(),float(bs)),
            'acc_pair': torch.div(correct_pair.sum(),float(bs)),
            'img_feats': img1.clone().detach(),
        }

        if self.cfg.MODEL.use_attr_loss:
            out['loss_aux_attr'] = aux_loss['loss_attr']
            # out['acc_aux_attr'] = aux_loss['acc_attr']

        if self.cfg.MODEL.use_obj_loss:
            out['loss_aux_obj'] = aux_loss['loss_obj']
            # out['acc_aux_obj'] = aux_loss['acc_obj']
        return out

    def construct_token_tensors_eval(self, idx, type):
        # soft_embedding = self.ao_dropout(self.soft_embedding)
        soft_embedding = self.init_token_embedding1
        if 'attrs' in type:
            eos_idx = int(self.attr_tempelete_token_id.argmax())
            token_id = self.attr_tempelete_token_id.repeat(len(idx), 1)
        elif 'objs' in type:
            eos_idx = int(self.obj_tempelete_token_id.argmax())
            token_id = self.obj_tempelete_token_id.repeat(len(idx), 1)
        elif 'pairs' in type:
            eos_idx = int(self.pair_tempelete_token_id.argmax())
            token_id = self.pair_tempelete_token_id.repeat(len(idx), 1)

        token_embedding = self.clip_model.token_embedding(token_id)

        if type == 'attrs':
            token_embedding[:, eos_idx-2, :] = soft_embedding[idx]
        elif type =='extra_attrs':
            token_embedding[:, eos_idx-2, :] = self.init_token_embedding[idx]
        elif type =='objs':
            token_embedding[:, eos_idx-1, :] = soft_embedding[idx+self.offset]
        elif type == 'extra_objs':
            token_embedding[:, eos_idx-1, :] = self.init_token_embedding[idx+self.offset]
        elif type == 'pairs':
            #todo：change the pair_id to attr_id and obj_id
            attr_id = [self.pair2attr_obj[pair][0] for pair in idx]
            obj_id = [self.pair2attr_obj[pair][1] for pair in idx]
            token_embedding[:, eos_idx-2, :] = soft_embedding[attr_id]
            token_embedding[:, eos_idx-1, :] = soft_embedding[[id + self.offset for id in obj_id]]
        elif type == 'extra_pairs':
            attr_id = [self.pair2attr_obj[pair][0] for pair in idx]
            obj_id = [self.pair2attr_obj[pair][1] for pair in idx]
            token_embedding[:, eos_idx-2, :] = self.init_token_embedding[attr_id]
            token_embedding[:, eos_idx-1, :] = self.init_token_embedding[[id + self.offset for id in obj_id]]

        text_emb = self.clip.text_encoder(token_id, token_embedding, enable_pos_emb=True)

        return text_emb

    def val_forward(self, batch):
        img = batch['img']
        bs = img.shape[0]

        concept = self.construct_token_tensors(self.all_pairs, 'pairs').float()   #使用梯度时

        # concept = self.construct_token_tensors_eval(self.all_pairs, 'pairs').float()    #使用残差时

        # attr_weight = self.construct_token_tensors(self.all_attrs, 'attrs').float()
        # obj_weight = self.construct_token_tensors(self.all_objs, 'objs').float()

        # attr_weight = self.construct_possible_token_tensors(self.all_attrs, 'attrs').float()
        # obj_weight = self.construct_possible_token_tensors(self.all_objs, 'objs').float()


        # ###用seen text feats的残差加权聚合加到unseen text feats上
        # 因为unseen pair中含有(AO)*,A*O,AO*,A*O*,所以其实可以更加细化各种情况,但是此处没有
        # # self.dset.test_data_split
        # only_train_pairs = [self.dset.unique_pair2idx[item] for item in self.dset.train_pairs]
        # only_train_pairs = torch.LongTensor(only_train_pairs).to(concept.device)
        # mask = ~torch.isin(self.all_pairs, only_train_pairs)
        # unseen_pairs = self.all_pairs[mask]

        # unseen_pairs_feats = self.construct_token_tensors(unseen_pairs, 'extra_pairs').float()  #用extra为了获得最初始的text feats
        # init_seen_pairs_feats = self.construct_token_tensors(only_train_pairs, 'extra_pairs').float() #初始值
        # seen_pairs_feats = self.construct_token_tensors(only_train_pairs, 'pairs').float()  #当前值

        # sim_matrix = F.cosine_similarity(unseen_pairs_feats.unsqueeze(1), init_seen_pairs_feats.unsqueeze(0), dim=2)
        # topk_sim, topk_indices = torch.topk(sim_matrix, k=5, dim=1)
        # residual = (seen_pairs_feats[topk_indices] - init_seen_pairs_feats[topk_indices]) * topk_sim.unsqueeze(-1)
        # concept[mask] = concept[mask] + residual.mean(dim=1)
        # ### END

        img, patch = self.clip.encode_image(img.half())
        img, patch = img.float(), patch.float()


        pred = self.classifier(img, concept)

        # #ensemble logits
        # ###每个epoch后按照seen的token embedding的残差加权聚合更新unseen的token embedding
        # train_attr_len = len(self.dset.train_attrs)
        # train_obj_len = len(self.dset.train_objs)
        # all_attr_len = len(self.dset.all_attrs)
        # all_obj_len = len(self.dset.all_objs)

        # concept_SAS = self.ensemble_construct_token_tensors(self.all_pairs).float()

        # pred_SAS = self.classifier(img, concept_SAS)

        # pred = (pred + pred_SAS) / 2.0



        # def find_unseen_ids(all_pairs, train_pairs):
        #     # 先转换为 CPU 上的集合再计算差集
        #     unseen_mask = ~torch.isin(all_pairs, train_pairs)
        #     seen_mask = torch.isin(all_pairs, train_pairs)
        #     unseen_ids = all_pairs[unseen_mask]
        #     seen_ids = all_pairs[seen_mask]
        #     return unseen_ids, seen_ids

        # # 用法示例
        # unseen_ids, seen_ids = find_unseen_ids(self.all_pairs, self.train_pairs)
        # unseen_concept = concept[unseen_ids]
        # seen_concept = concept[seen_ids]

        ### F.cosine_similarity(unseen_attr_text_emb.unsqueeze(1), seen_attr_text_emb.unsqueeze(0), dim=2)

        ###使用q-former解耦合视觉属性和物体
        # vis_att = self.image_pair_comparison.mlp_att(img)
        # vis_obj = self.image_pair_comparison.mlp_obj(img)

        # attr_q = self.image_pair_comparison.attr_query
        # attr_q = attr_q.unsqueeze(0).expand(bs, -1, -1)
        # obj_q = self.image_pair_comparison.obj_query
        # obj_q = obj_q.unsqueeze(0).expand(bs, -1, -1)


        # vis_att = self.image_pair_comparison.transformer_decoder(attr_q.permute(1, 0, 2), patch.permute(1, 0, 2)).permute(1, 0, 2)
        # vis_att = torch.mean(vis_att, dim=1)
        # vis_obj = self.image_pair_comparison.transformer_decoder(obj_q.permute(1, 0, 2), patch.permute(1, 0, 2)).permute(1, 0, 2)
        # vis_obj = torch.mean(vis_obj, dim=1)

        # vis_att, vis_obj = self.image_pair_comparison.mlp_att(vis_att), self.image_pair_comparison.mlp_obj(vis_obj)

        # pred_attr = self.classifier(vis_att, attr_weight)
        # pred_obj = self.classifier(vis_obj, obj_weight)

        # factor = 0.1
        # for i in range(pred.shape[-1]):
        #     pred_ao = pred_attr[:, self.all_pair2attr_obj[self.all_pairs[i]][0]] + pred_obj[:, self.all_pair2attr_obj[self.all_pairs[i]][1]]
        #     pred[:, i] = (1- factor) * pred[:, i] + factor * pred_ao
        # ### END
        pred = F.softmax(pred, dim=1)
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
