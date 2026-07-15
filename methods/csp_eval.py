import argparse
import torch
import os
from models.csp.oaclip_ov import OACLIPv3  # 根据你的模型文件
from dataset import CompositionDataset  # 根据你的数据集文件
import evaluator as evaluator_ge  # 评估模块
from utils import utils  # 包含工具函数
from torch.utils.tensorboard import SummaryWriter
import warnings
from config import cfg
from methods.csp_train import validate_ge
import numpy as np
import time
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from thop import profile, clever_format

warnings.filterwarnings("ignore")

list_stats_report = ['AUC', 'best_hm', 'sa_so_acc', 'sa_so_u_acc', 'sa_uo_acc', 'ua_so_acc', 'ua_uo_acc', 's_a_acc', 'u_a_acc', 's_o_acc', 'u_o_acc']

def freeze(m):
    """冻结模型参数，不更新梯度"""
    m.eval()
    for p in m.parameters():
        p.requires_grad = False
        p.grad = None

def load_model(cfg, device, checkpoint_path):
    """加载训练好的模型"""
    # 初始化模型
    trainset = CompositionDataset(
        phase='train', split=cfg.DATASET.splitname, cfg=cfg)

    model = OACLIPv3(trainset, cfg)  # None 因为不需要训练集
    model = model.to(device)

    if os.path.exists(checkpoint_path):
        print(f"Loading model from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    return model

def evaluate(model, cfg, device):
    """评估模型在测试集上的表现"""

    # 加载测试集
    testset = CompositionDataset(phase='test', split=cfg.DATASET.splitname, cfg=cfg)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=cfg.TRAIN.test_batch_size, shuffle=False,
        num_workers=cfg.TRAIN.num_workers)


    # 评估器初始化
    evaluator_test_ge = evaluator_ge.Evaluator(testset, cfg)

    # 开始评估
    start_time = time.perf_counter()
    model.eval()
    # inputs = next(iter(testloader))  # 获取一个批次的数据
    # print("len(inputs): ", len(inputs))
    # for k in inputs:
    #     if isinstance(inputs[k], list):
    #         continue
    #     inputs[k] = inputs[k].to(device, non_blocking=True)

    # ###计算 FLOPs 和参数量
    # flops = FlopCountAnalysis(model, inputs)
    # print("FLOPs: ", flops.total()/1e9) # .total() 方法直接给出总数

    # macs, params = profile(model, inputs=(inputs, ))

    # # 使用 clever_format 格式化输出，会自动加上 G, M 等单位
    # macs, params = clever_format([macs, params], "%.3f")
    # print(f"MACs: {macs}")
    # print(f"Params: {params}")

    stats_test = validate_ge(0, model, testloader, evaluator_test_ge, device, phase='test')  # 调用 train.py 中的 validate_ge
    end_time = time.perf_counter()

    # 计算时间差（毫秒）
    elapsed_ms = (end_time - start_time) * 1000

    print(f"代码执行耗时: {elapsed_ms:.3f} 毫秒")



def main_worker(cfg, checkpoint_path):
    """评估脚本的主函数"""
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 加载模型
    model = load_model(cfg, device, checkpoint_path)

    # 冻结特定层（如不需要更新）
    # freeze(model.attr_embedder)
    # freeze(model.obj_embedder)
    # freeze(model.pair_embedder)
    freeze(model.clip_model)

    # before_train_text_embedding_1 = model.token2text(model.init_token_embedding1[:model.offset], "attrs")
    # before_train_text_embedding_2 = model.token2text(model.init_token_embedding1[model.offset:], "objs")
    # before_train_text_embedding = torch.cat((before_train_text_embedding_1, before_train_text_embedding_2), dim=0)
    # after_train_text_embedding_1 = model.token2text(model.soft_embedding[:model.offset], "attrs")
    # after_train_text_embedding_2 = model.token2text(model.soft_embedding[model.offset:], "objs")
    # after_train_text_embedding = torch.cat((after_train_text_embedding_1, after_train_text_embedding_2), dim=0)

    # torch.save(before_train_text_embedding, 'before_train_text_embedding.pt')
    # torch.save(after_train_text_embedding, 'after_train_text_embedding.pt')
    # freeze(model.feat_extractor)
    # 评估模型
    evaluate(model, cfg, device)

def main():
    """主函数入口"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True, help='path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='path to model checkpoint')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER, help='modify config file from terminal')
    args = parser.parse_args()

    # 加载配置文件
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    print(cfg)

    # 设置随机种子
    seed = cfg.TRAIN.seed
    if seed == -1:
        seed = np.random.randint(1, 10000)
    print('Random seed:', seed)
    cfg.TRAIN.seed = seed

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    # 运行评估
    main_worker(cfg, args.checkpoint)

if __name__ == "__main__":
    main()
