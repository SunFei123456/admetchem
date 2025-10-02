from argparse import Namespace

from torch import nn
import numpy as np
import torch
from sklearn.metrics import auc, mean_absolute_error, mean_squared_error, precision_recall_curve, r2_score,\
    roc_auc_score, accuracy_score, log_loss

from model.MoleculeModel import MoleculeModel


def remove_nan_label(pred,truth):
    nan = torch.isnan(truth)
    truth = truth[~nan]
    pred = pred[~nan]

    return pred,truth
def remove_nan_line(encoder,truth):
    if truth.dim() == 1:
        truth = truth.unsqueeze(1)
    nan = torch.isnan(truth).any(dim=1)
    truth = truth[~nan]
    encoder = encoder[~nan]

    return encoder,truth


def remove_nan_eval(encoder, truth):
    # 确保 truth 是二维的
    if truth.dim() == 1:
        truth = truth.unsqueeze(1)

    # 检测每一行是否有 NaN
    nan_mask = torch.isnan(truth).any(dim=1)

    # 过滤 truth 中的行，移除包含 NaN 的行
    truth = truth[~nan_mask]

    # 使用列表推导式过滤 encoder 中的元素，保持与 truth 的过滤一致
    encoder = [enc for enc, is_nan in zip(encoder, nan_mask) if not is_nan]

    return encoder, truth
def roc_auc(pred,truth):
    return roc_auc_score(truth,pred)

def rmse(pred,truth):
    return nn.functional.mse_loss(pred,truth)**0.5

def mae(pred,truth):
    return mean_absolute_error(truth,pred)

func_dict={'relu':nn.ReLU(),
           'sigmoid':nn.Sigmoid(),
           'mse':nn.MSELoss(),
           'rmse':rmse,
           'mae':mae,
           'crossentropy':nn.CrossEntropyLoss(),
           'bce':nn.BCEWithLogitsLoss(),
           'auc':roc_auc,
           'leakyrelu':nn.LeakyReLU(),
           'prelu':nn.PReLU(),
           'tanh':nn.Tanh(),
           'selu':nn.SELU(),
           'elu':nn.ELU(),
           'gelu':nn.GELU()
           }

def get_func(fn_name):
    """
        获取激活函数.

        :param fn_name: 激活函数的名称.
        :return: 该激活函数
    """
    fn_name = fn_name.lower()
    return func_dict[fn_name]

def build_model(args: Namespace, encoder_name) -> nn.Module:
    """
    构建一个分子模型. 构建的分子适用于下游任务微调，output_size是num_tasks的大小

    :param args: 参数.
    :return: 一个包含消息传递网络（MPN）编码器以及初始化了参数的最终线性层的分子模型
    """
    output_size = args["num_tasks"]

    args["output_size"] = output_size
    if args["dataset_type"] == 'multiclass':
        args["output_size"] *= args["multiclass_num_classes"]

    model = MoleculeModel(classification=args["dataset_type"] == 'classification', multiclass=args["dataset_type"] == 'multiclass', pretrain=args["pre_train"])
    model.create_encoder(args, encoder_name)
    model.create_ffn(args)

    model.initialize_weights()

    return model


def build_pretrain_model(args: Namespace, encoder_name) -> nn.Module:
    """
    构建一个分子模型. 构建的分子适用于预训练，output_size是hidden_size的大小

    :param args: 参数.
    :return: 一个分子模型.
    """
    args["ffn_hidden_size"] = args["hid_dim"] // 2
    args["output_size"] = args["hid_dim"]

    model = MoleculeModel(classification=args["dataset_type"] == 'classification',
                          multiclass=args["dataset_type"] == 'multiclass', pretrain=True)
    model.create_encoder(args, encoder_name)
    model.create_ffn(args)

    model.initialize_weights()

    return model
