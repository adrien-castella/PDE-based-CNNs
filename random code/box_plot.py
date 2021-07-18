import os, json, random
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

folders = os.path.join("output", "correct-rotated-")
folders = [folders + str(i) for i in {0,2,3,4}]

DICE = {
    'conventional': [0.8017, 0.8021, 0.8033, 0.8047, 0.8057],
    'PDE-based': [0.7952, 0.7953, 0.7981, 0.7992, 0.8021]
}
TIMES_TRAIN = {
    'conventional': [1.3849, 1.4497, 1.5279, 1.6298, 1.8006],
    'PDE-based': [9.7426, 11.2137, 12.6799, 12.7736, 14.3227]
}
TIMES_TEST = {
    'conventional': [1.9335, 1.9263, 1.9456, 1.8972, 2.015],
    'PDE-based': [5.3705, 5.9184, 6.7018, 7.0891, 7.8623]
}
CHANNELS = [12, 14, 16, 18, 20]
PARAMS = {
    'conventional': [12900, 17142, 21984, 27426, 33468],
    'PDE-based': [1962, 2482, 3058, 3690, 4378]
}

def plot(dice, measure, keys, name, axis1, axis2, t):
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    plt.title(t)
    ax1.set_ylabel(axis1)
    plt1 = ax1.plot(keys, dice, color=color, label=axis1)

    ax2 = ax1.twinx()

    color='tab:blue'
    ax2.set_ylabel(axis2)
    plt2 = ax2.plot(keys, measure, color=color, label=axis2)

    ax = plt1 + plt2
    labels = [l.get_label() for l in ax]
    ax2.legend(ax, labels)

    plt.savefig(name + '.png')
    plt.close()

plot(DICE['conventional'], PARAMS['conventional'], CHANNELS, 'conv_d-v-p', 'DICE', 'Parameters', 'DICE v Parameters (Conventional CNN)')
plot(DICE['conventional'], TIMES_TRAIN['conventional'], CHANNELS, 'conv_d-v-train', 'DICE', 'Train time', 'DICE v Train time (Conventional CNN)')
plot(DICE['conventional'], TIMES_TEST['conventional'], CHANNELS, 'conv_d-v-test', 'DICE', 'Test time', 'DICE v Test time (Conventional CNN)')
plot(DICE['conventional'], DICE['PDE-based'], CHANNELS, 'conv-v-pde_DICE', 'Conventional', 'PDE-based', 'DICE score comparison')
plot(PARAMS['conventional'], PARAMS['PDE-based'], CHANNELS, 'conv-v-pde_param', 'Conventional', 'PDE-based', 'Parameter comparison')
plot(TIMES_TRAIN['conventional'], TIMES_TRAIN['PDE-based'], CHANNELS, 'conv-v-pde_train', 'Conventional', 'PDE-based', 'Training time comparison')
plot(TIMES_TEST['conventional'], TIMES_TEST['PDE-based'], CHANNELS, 'conv-v-pde_test', 'Conventional', 'PDE-based', 'Testing time comparison')
plot(DICE['PDE-based'], PARAMS['PDE-based'], CHANNELS, 'pde_d-v-p', 'DICE', 'Parameters', 'DICE v Parameters (PDE-based CNN)')
plot(DICE['PDE-based'], TIMES_TRAIN['PDE-based'], CHANNELS, 'pde_d-v-train', 'DICE', 'Train time', 'DICE v Train time (PDE-based CNN)')
plot(DICE['PDE-based'], TIMES_TEST['PDE-based'], CHANNELS, 'pde_d-v-test', 'DICE', 'Test time', 'DICE v Test time (PDE-based CNN)')