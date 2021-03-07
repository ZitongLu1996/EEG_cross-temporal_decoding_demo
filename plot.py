# -*- coding: utf-8

"""
@File       :   plot.py
@Author     :   Zitong Lu
@Contact    :   zitonglu1996@gmail.com
@License    :   MIT License
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt

# cross-temporal decoding结果绘图函数
def plot_ctresults(decoding_results_dir, subs):
    nsubs = len(subs)
    rlts = np.zeros([nsubs, 100, 100], dtype=np.float)
    subindex = 0
    f = h5py.File(decoding_results_dir, "r")
    for sub in subs:
        rlts[subindex] = np.array(f[sub])
        subindex = subindex + 1
    f.close()

    avg = np.average(rlts, axis=0)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    plt.imshow(avg, extent=(-0.5, 1.5, -0.5, 1.5), origin='low', cmap="bwr", clim=(0.035, 0.09))
    cb = plt.colorbar(ticks=[0.035, 0.09])
    cb.ax.tick_params(labelsize=12)
    font = {'size': 15, }
    cb.set_label('Classification Accuracy', fontdict=font)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.xlabel("Training Time-point (s)", fontsize=16)
    plt.ylabel("Test Time-point (s)", fontsize=16)
    plt.show()

# 绘图
subs = ["201", "202", "203", "204", "205"]
classification_results_dir = 'classification_results/'
print("Position Classification-based Decoding Results!")
plot_ctresults(classification_results_dir+"ERP_pos.h5", subs)
