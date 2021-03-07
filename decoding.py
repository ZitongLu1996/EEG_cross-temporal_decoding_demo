# -*- coding: utf-8

"""
@File       :   decoding.py
@Author     :   Zitong Lu
@Contact    :   zitonglu1996@gmail.com
@License    :   MIT License
"""

import os
import numpy as np
import scipy.io as sio
import h5py
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from pyctrsa.util.progressbar import show_progressbar

# 数据进行decoding前的预处理

data_dir = 'data/'
classification_results_dir = 'classification_results/'

# 被试id
subs = ["201", "202", "203", "204", "205"]

exist = os.path.exists(data_dir + 'data_for_classification/ERP/')
if exist == False:
    os.makedirs(data_dir + 'data_for_classification/ERP/')

for sub in subs:
    data = sio.loadmat(data_dir + "data/ERP" + sub + ".mat")["filtData"][:, :, 250:]
    print(data.shape)
    # 数据shape: n_trials, n_channels, n_times
    pos_label = np.loadtxt(data_dir + "labels/pos_" + sub + ".txt")[:, 1]
    # 脑电数据
    pos_subdata500 = np.zeros([16, 40, 27, 500], dtype=np.float)
    # 标签
    pos_labelindex = np.zeros([16], dtype=np.int)
    for i in range(640):
        label = int(pos_label[i])
        pos_subdata500[label, pos_labelindex[label]] = data[i]
        pos_labelindex[label] = pos_labelindex[label] + 1
    # 对连续五个时间点的数据进行平均-即数据降采样
    pos_subdata = np.zeros([16, 40, 27, 100], dtype=np.float)
    for t in range(100):
        pos_subdata[:, :, :, t] = np.average(pos_subdata500[:, :, :, t * 5:t * 5 + 5], axis=3)
    # 存储预处理后的数据
    f = h5py.File(data_dir + "data_for_classification/ERP/" + sub + ".h5", "w")
    f.create_dataset("pos", data=pos_subdata)
    f.close()

# 经过预处理后的数据，其shape是[16, 40, 27, 100]
# 即16个不同的位置、每种条件40个试次、27个导联、100个时间点（从-500ms到1500ms，采样率为50Hz）

# 跨时域解码

print("\nPosition Decoding!")
subindex = 0
f = h5py.File(classification_results_dir+"ERP_pos.h5", "w")
total = len(subs)*10*3*100
# 对每个被试进行计算
for sub in subs:
    fdata = h5py.File(data_dir+"data_for_classification/ERP/"+sub+".h5", "r")
    data = np.array(fdata["pos"])
    fdata.close()
    acc = np.zeros([10, 100, 100, 3], dtype=np.float)
    # 重复试次decoding计算
    for k in range(10):
        # 对每一种朝向的40个试次，随机抽取39个试次，再每13个试次进行一次平均
        index_trials = np.array(range(40))
        shuffle = np.random.permutation(index_trials)
        newdata = data[:, shuffle[:39]]
        block_data = np.zeros([3, 16, 27, 100], dtype=np.float)
        for i in range(3):
            block_data[i] = np.average(newdata[:, i*13:i*13+13], axis=1)
        # 即每种条件下有三个平均后的脑电数据，共16*3个样本
        # 使用三折的方法，把48个样本分为3折（每折16个样本）
        y_train = np.zeros([2*16], dtype=np.int)
        for i in range(2):
            for j in range(16):
                y_train[i*16+j] = j
        y_test = np.zeros([16], dtype=np.int)
        for i in range(16):
            y_test[i] = i
        # 三折中每一折都会作为一次测试集（剩下两折作为训练集）
        for i in range(3):
            x_test = block_data[i]
            x_train = np.zeros([2, 16, 27, 100], dtype=np.float)
            index = 0
            for j in range(3):
                if j != i:
                    x_train[index] = block_data[j]
                    index = index + 1
            x_train = np.reshape(x_train, [2*16, 27, 100])
            for t in range(100):
                x_train_t = x_train[:, :, t]
                x_test_t = x_test[:, :, t]
                # 利用时间t的数据进行训练
                svm = SVC(kernel='linear', decision_function_shape='ovr')
                svm.fit(x_train_t, y_train)
                y_pred = svm.predict(x_test_t)
                # 对同一时间t的数据进行测试
                acc[k, t, t, i] = accuracy_score(y_test, y_pred)
                for tt in range(99):
                    # 对其他时间tt的数据进行测试
                    if tt < t:
                        x_test_tt = x_test[:, :, tt]
                        y_pred = svm.predict(x_test_tt)
                        acc[k, t, tt, i] = accuracy_score(y_test, y_pred)
                    if tt >= t:
                        x_test_tt = x_test[:, :, tt+1]
                        y_pred = svm.predict(x_test_tt)
                        acc[k, t, tt+1, i] = accuracy_score(y_test, y_pred)
                # 显示计算进度条
                percent = (subindex*10*3*100+k*3*100+i*100+t)/total*100
                show_progressbar("Calculating", percent)
    subindex = subindex + 1
    # 存储cross-temporal decoding结果
    f.create_dataset(sub, data=np.average(acc, axis=(0, 3)))
f.close()