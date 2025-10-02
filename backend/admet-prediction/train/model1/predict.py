from typing import Union
import os
from typing import Tuple, List

import numpy as np
import json
import torch

from model.util import  build_model
dic = {
        "LogS": "reg",
        "LogP": "reg",
        "LogD": "reg",
        "Caco-2": "reg",
        "CL": "reg",
        "Drug Half-Life": "reg",
        "ROA": "reg",
        "PPBR": "reg",
        "VDss": "reg",
        "Hydration Free Energy" : "hfe",
        "CYP1A2-inh": "class",
        "CYP2C9-inh": "class",
        "CYP2C9-sub": "class",
        "CYP2C19-inh": "class",
        "CYP2D6-inh": "class",
        "CYP2D6-sub": "class",
        "CYP3A4-inh": "class",
        "CYP3A4-sub": "class",
        "hERG": "class",
        "Pgp-inh": "class",
        "Ames": "class",
        "BBB": "class",
        "DILI": "class",
        "HIA": "class",
        "SkinSen": "class",
        "NR-AR-LBD": "class",
        "NR-AR": "class",
        "NR-AhR": "class",
        "NR-Aromatase": "class",
        "NR-ER": "class",
        "NR-ER-LBD": "class",
        "NR-PPAR-gamma": "class",
        "SR-ARE": "class",
        "SR-ATAD5": "class",
        "SR-HSE": "class",
        "SR-MMP": "class",
        "SR-p53": "class",
        "PAMPA":"pama",
        "Bioavailability":"bio"
}
regdic = {
    "LogS": 0,
    "LogP": 1,
    "LogD": 2,
    "Caco-2": 3,
    "CL": 4,
    "Drug Half-Life": 5,
    "ROA": 6,
    "PPBR": 7,
    "VDss": 8
}
classdic={
    "CYP1A2-inh": 0,
    "CYP2C9-inh": 1,
    "CYP2C9-sub": 2,
    "CYP2C19-inh": 3,
    "CYP2D6-inh": 4,
    "CYP2D6-sub": 5,
    "CYP3A4-inh": 6,
    "CYP3A4-sub": 7,
    "hERG": 8,
    "Pgp-inh": 9,
    "Ames": 10,
    "BBB": 11,
    "DILI": 12,
    "HIA": 13,
    "SkinSen": 14,
    "NR-AR-LBD": 15,
    "NR-AR": 16,
    "NR-AhR": 17,
    "NR-Aromatase": 18,
    "NR-ER": 19,
    "NR-ER-LBD": 20,
    "NR-PPAR-gamma": 21,
    "SR-ARE": 22,
    "SR-ATAD5": 23,
    "SR-HSE": 24,
    "SR-MMP": 25,
    "SR-p53": 26
}

def predict(
        smiles_list: Union[str, List[str]],
        model_path: str,
        model_args: dict,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> np.ndarray:
    """
    使用训练好的模型预测SMILES的属性

    参数:
        smiles_list: 单个SMILES字符串或SMILES列表
        model_path: 训练好的模型路径
        model_args: 模型参数
        device: 运行设备

    返回:
        predictions: 预测结果数组
    """
    # 确保输入是列表形式
    if isinstance(smiles_list, str):
        smiles_list = [smiles_list]

    # 构建模型并加载权重
    model = build_model(model_args, encoder_name="HModel").to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # 设置为评估模式

    # 进行预测
    with torch.no_grad():
        predictions, _ = model(smiles_list)

    # 转换为numpy数组并返回
    return predictions.cpu().numpy()
def probability_to_admet_symbol(prob: float) -> str:
    """
    根据ADMETlab 2.0的分类阈值将概率值转换为符号。

    参数:
        prob: 预测概率值（范围0-1）

    返回:
        symbol: 对应的分类符号
    """
    if prob < 0.1:
        return '---'
    elif prob < 0.3:
        return '--'
    elif prob < 0.5:
        return '-'
    elif prob < 0.7:
        return '+'
    elif prob < 0.9:
        return '++'
    else:  # 0.9 <= prob <= 1.0
        return '+++'

def predict_class(
        smiles_list: Union[str, List[str]],
        model_path: str,
        model_args: dict,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用训练好的模型预测SMILES的属性，并返回ADMETlab 2.0符号分类结果。

    参数:
        smiles_list: 单个SMILES字符串或SMILES列表
        model_path: 训练好的模型路径
        model_args: 模型参数
        device: 运行设备

    返回:
        raw_predictions: 原始预测概率数组（形状: n_samples, n_tasks）
        admet_symbols: 对应的ADMET符号数组（形状: n_samples, n_tasks）
    """
    # 确保输入是列表形式
    if isinstance(smiles_list, str):
        smiles_list = [smiles_list]

    # 构建模型并加载权重
    model = build_model(model_args, encoder_name="HModel").to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # 设置为评估模式

    # 进行预测
    with torch.no_grad():
        predictions, _ = model(smiles_list)

    # 转换为numpy数组
    raw_predictions = predictions.cpu().numpy()

    # 将概率值转换为ADMET符号
    # 使用vectorize函数处理多维数组
    vectorized_map = np.vectorize(probability_to_admet_symbol)
    admet_symbols = vectorized_map(raw_predictions)

    return raw_predictions, admet_symbols
def judge(smile:str):
    return dic[smile]

def categorize_tasks(task_types):
    # 创建分类字典
    categorized = {
        "reg": [],
        "class": [],
        "hfe": [],
        "pama": [],
        "bio": []
    }

    # 对每个任务类型进行分类
    for task in task_types:
        if task in dic:
            category = dic[task]
            categorized[category].append(task)

    # 移除空类别
    categorized = {k: v for k, v in categorized.items() if v}
    return categorized
def task_index(task,task_type):
    if task_type == "class":
        return classdic[task]
    elif task_type == "reg":
        return regdic[task]
def predict_result(smiles,task_type):
    config_path = f"configs/{task_type}.json"
    config = json.load(open(config_path, 'r'))
    data_args = config['data']
    train_args = config['train']
    train_args['data_name'] = config_path.split('/')[-1].strip('.json')
    train_args["runs"] = 1
    data_args["features_generator"] = None
    model_args = config['model']
    model_args["pre_train"] = train_args["pre_train"]
    model_args["device"] = train_args["device"]
    seed = config['seed']
    if not isinstance(seed, list):
        seed = [seed]
    num_task = model_args["num_task"]
    if task_type == "class" or task_type == "bio" or task_type == "pama":
        predictions, symbols = predict_class(
            smiles_list=smiles,
            model_path=os.path.join(train_args["save_path"], "model1_class_latest.pt"),
            model_args=model_args,
            device=train_args['device']
        )
        return predictions, symbols
    else:
        predictions = predict(
            smiles_list=smiles,
            model_path=os.path.join(train_args["save_path"], "model1_reg_latest.pt"),
            model_args=model_args,
            device=train_args['device']
        )
        return predictions
def predict_smiles(smiles,task_types):
    categories = categorize_tasks(task_types)
    result = {}
    for smile_str in smiles:
        result[smile_str] = {}
    for k,v in categories.items():
        if k == "class":
            predictions,symbols = predict_result(smiles,"class")

            for task_name in v:
                # 获取该任务在预测结果中的索引
                idx = task_index(task_name, "class")
                for i, smile_str in enumerate(smiles):
                    # 确保每个smile的字典已初始化
                    if smile_str not in result:
                        result[smile_str] = {}

                    # 添加该任务的预测结果
                    result[smile_str][task_name] = {
                        "prediction": predictions[i][idx],
                        "symbol": symbols[i][idx]
                    }
        elif k == "bio":
            predictions, symbols = predict_result(smiles, "class")
            for i, smile_str in enumerate(smiles):
                # 确保每个smile的字典已初始化
                if smile_str not in result:
                    result[smile_str] = {}

                # 添加该任务的预测结果
                result[smile_str]["Bioavailability"] = {
                    "prediction": predictions[i][0],
                    "symbol": symbols[i][0]
                }
        elif k == "pama":
            predictions, symbols = predict_result(smiles, "class")
            for i, smile_str in enumerate(smiles):
                # 确保每个smile的字典已初始化
                if smile_str not in result:
                    result[smile_str] = {}

                # 添加该任务的预测结果
                result[smile_str]["PAMPA"] = {
                    "prediction": predictions[i][0],
                    "symbol": symbols[i][0]
                }

        else:
            predictions = predict_result(smiles, "reg")
            for task_name in v:
                # 获取该任务在预测结果中的索引
                idx = task_index(task_name, "reg")
                for i, smile_str in enumerate(smiles):
                    # 确保每个smile的字典已初始化
                    if smile_str not in result:
                        result[smile_str] = {}

                    # 添加该任务的预测结果
                    result[smile_str][task_name] = {
                        "prediction": predictions[i][idx]
                    }

    return result



if __name__ == '__main__':
    import sys


    # 示例SMILES
    test_smiles = [
        "CCO",  # 乙醇
        "CC(=O)O",  # 乙酸
        "c1ccccc1",  # 苯
    ]

    # 进行预测
    result = predict_smiles(test_smiles,[ "Bioavailability","LogS","LogP","Pgp-inh","Ames"])
    print("预测结果:")
    print(result)