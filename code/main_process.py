# main_process.py
# 主程序入口，进行数据加载、预处理、模型训练与测试，并评估分类准确率

import os
import pandas as pd
import numpy as np
import random
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch import nn
from data_loader import *
from sklearn.metrics import *
from sklearn.preprocessing import *
from collections import Counter
from autoencoder import *
from evt import *
import argparse

# 设置随机种子，保证实验可复现
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# 解析命令行参数，支持自定义数据集名称
default_data_name = 'demo'
parser = argparse.ArgumentParser()
parser.add_argument('--data_name', type = str, default = default_data_name, help = 'data name')
args = parser.parse_args()
data_name = args.data_name

# 加载数据集，并进行标准化
# data_load: 特征数据, label_load: 标签
data_load, label_load = read_dataset(data_name)
data_load = StandardScaler().fit_transform(data_load)

# 统计每个类别的样本数，确定训练/测试划分
count_number = Counter(label_load)
min_num = np.array(list(count_number.values())).min()  # 最小类别样本数
test_per_class = 260  # 每类测试样本数
num_per_class = min_num - test_per_class  # 每类训练样本数
dim = data_load.shape[1]  # 特征维度
b_size = test_per_class   # 批量大小
loss_func = nn.MSELoss()  # 损失函数

sum_num = len(set(list((label_load))))  # 类别总数
train_num = 1  # 训练类别数（只用一个类别做已知，其余为新类）
newclass_num = sum_num - train_num  # 新类别数

# 打乱数据顺序
shun = list(range(data_load.shape[0]))
random.shuffle(shun)
data_load = data_load[shun]
label_load = label_load[shun]

# 随机排列类别索引
allIndex = np.random.permutation(train_num + newclass_num)

# 构建训练集（只包含已知类别）
data = np.zeros((num_per_class * (train_num), dim))
label = np.zeros(num_per_class * (train_num))
for pos in range(train_num):
    i = allIndex[pos]
    data[pos * num_per_class:(pos + 1) * num_per_class,:] = data_load[label_load==i][0:num_per_class, :]
    label[pos * num_per_class:(pos + 1) * num_per_class] = i


# 构建流式测试集（包含所有类别，已知类标记为原标签，未知类标记为999）
streamdata = np.zeros((test_per_class * (train_num + newclass_num), dim))
streamlabel = np.zeros(test_per_class * (train_num + newclass_num))
gtlabel = np.zeros(test_per_class * (train_num + newclass_num))
for pos in range(train_num + newclass_num):
    i = allIndex[pos]
    streamdata[pos * test_per_class:(pos + 1) * test_per_class,:] = data_load[label_load==i][-test_per_class:, :]
    gtlabel[pos * test_per_class:(pos + 1) * test_per_class] = i
    if pos < train_num:
        streamlabel[pos * test_per_class:(pos+1) * test_per_class] = i
    else:
        streamlabel[pos * test_per_class:(pos + 1) * test_per_class] = 999


# 根据标签统计，筛选样本数大于50的类别
# 返回当前存在的类别列表
def make_lab(label):
    xianyou = pd.DataFrame(label).value_counts()
    curr_lab = []
    for j1 in xianyou.keys():
        if xianyou[j1] > 50:
            curr_lab.append(j1[0])
    return curr_lab

# 针对每个类别训练自编码器模型，并用SPOT方法确定阈值
# 返回模型列表、阈值列表、类别列表
def train(data, label, curr_lab):
    mod_ls = []
    thred_ls = []
    class_ls = []
    batch = 10
    epoch = 10
    y_in, y1, y2, y3, y4 = data_load.shape[1], 256, 128, 64, 32
    for i in curr_lab:
        class_ls.append(i)
        model = Autoencoder(y_in, y1, y2, y3, y4)
        optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay = 5e-4)
        # 训练自编码器
        for i2 in range(epoch):
            shun = list(range(data[label==i].shape[0]))
            random.shuffle(shun)
            for i3 in range(int(data[label==i].shape[0] / batch)):
                data_input = torch.from_numpy(data[label==i][shun][i3 * batch : (i3+1) * batch]).float()
                pred = model(data_input)
                loss = loss_func(pred, data_input)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        mod_eva = model.eval()
        mod_ls.append(model)
        # 计算重构误差，用于阈值确定
        mse_ls = []
        for i4 in range(int(data[label==i].shape[0] / batch)):
            data_input = torch.from_numpy(data[label==i][i4 * batch : (i4+1) * batch]).float()
            pred = model(data_input)
            for i5 in range(pred.shape[0]):
                loss = loss_func(pred[i5], data_input[i5])
                mse_ls.append(float(loss.detach().numpy()))
        data_input = torch.from_numpy(data[label==i][(i4 + 1) * batch:]).float()
        pred = model(data_input)
        for i5 in range(pred.shape[0]):
            loss = loss_func(pred[i5], data_input[i5])
            mse_ls.append(float(loss.detach().numpy()))
        loss_list_use = np.array(mse_ls)
        q = 5e-2 # 风险参数，可调
        s = SPOT(q)
        s.fit(loss_list_use, loss_list_use)
        s.initialize()
        results = s.run_simp()
        # 阈值选取
        if results['thresholds'][0] > 0:
            thred_ls.append(results['thresholds'][0])
        else:
            thred_ls.append(np.sort(s.init_data)[int(0.85 * s.init_data.size)])
    return mod_ls, thred_ls, class_ls

# 主流程入口
if __name__ == '__main__':
    print('=== Initializing ===')
    # 获取当前类别
    curr_lab = make_lab(label)
    # 训练初始模型
    mod_ls, thred_ls, class_ls = train(data, label, curr_lab)
    
    res_ls = []  # 预测结果列表
    # 流式数据逐步推理与模型更新
    for i5 in range(streamdata.shape[0]):
        # 每处理完一个新类，更新模型
        if i5 % b_size == 0 and int(i5 / b_size) > train_num:
            updatedata = np.concatenate([data, streamdata[:i5]], axis=0)
            updatelabel = np.concatenate([label, gtlabel[:i5]], axis=0)
            curr_lab = make_lab(updatelabel)
            mod_ls, thred_ls, class_ls = train(updatedata, updatelabel, curr_lab) 
            print('*** Update model ***')
        # 对当前样本用所有模型计算重构误差
        data_input = torch.from_numpy(streamdata[i5]).float()
        mse_test = []
        for model in mod_ls:
            mod_eva = model.eval()
            pred = model(data_input)
            loss = loss_func(pred, data_input)
            mse_test.append(float(loss.detach().numpy()))
        # 判断是否为新类
        cand_res = np.array(mse_test)[np.array(mse_test) < np.array(thred_ls)]
        if len(cand_res) == 0:
            res_ls.append(999)
        else:
            min_loss_res = cand_res.min()
            res_ls.append(class_ls[mse_test.index(min_loss_res)])
    
    # Output complete res_ls results
    print("=== Prediction Results ===")
    print("res_ls contents:", res_ls)

    # Show data distribution in res_ls
    res_distribution = Counter(res_ls)
    print("=== Data Distribution ===")
    for class_label, count in res_distribution.items():
        print(f"Class {class_label}: {count} samples")

    # 对新类样本，将999替换为真实标签
    for ii in range(train_num + newclass_num):
        if ii >= train_num:
            rep_npy = np.array(res_ls[test_per_class * ii : test_per_class * (ii + 1)])
            rep_npy2 = rep_npy.copy()
            rep_npy[rep_npy2==999] = allIndex[ii]
            res_ls[test_per_class * ii:test_per_class * (ii + 1)] = list(rep_npy)
    
    # 计算准确率
    y_pred = np.array(res_ls)
    y_true = gtlabel[:len(res_ls)].copy()
    acc = accuracy_score(y_true, y_pred)
    print('Dataset:', data_name)
    print('Accuracy:', acc)

