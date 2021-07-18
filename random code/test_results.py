import json, os, glob
import numpy as np
import torch

PATH = os.path.join('output', 'alpha tests')
NAME = 'alpha test '
n = 14

def get_scores(filepaths):
    scores_065 = []
    scores_100 = []

    for path in filepaths:
        if '065' in path:
            scores_065.append(open_scores(path))
        else:
            scores_100.append(open_scores(path))
    
    scores_065 = np.array(scores_065).transpose().tolist()
    scores_100 = np.array(scores_100).transpose().tolist()

    return [scores_065, scores_100]

def open_scores(path):
    with open(path, 'r') as file:
        content = file.read()
        content = content.split('\n')

        dice = float(content[2].split(' ')[2][:-1])
        acc = float(content[4].split(' ')[2][:-1])
        auc = float(content[6].split(' ')[2][:-1])
    
    return [dice, acc, auc]

def get_stats(scores):
    index = ['dice', 'acc', 'auc']

    mean = []
    best = []
    for i in range(3):
        mean.append([float(scores[i, 0].mean()), float(scores[i, 1].mean())])
    
    scores = scores.transpose(1,2)

    numwon = []
    for i in range(3):
        count = 0
        for i in scores[i]:
            if i[1] == max(i):
                count = count + 1
        numwon.append(count)

    output = ''
    for i in range(3):
        output = output + index[i] + '\n'
        output = output + f'Number of wins: {numwon[i]} / {n}\n'
        output = output + f'Mean score 0.65: {mean[i][0]}\n'
        output = output + f'Mean score 1.00: {mean[i][1]}\n'
    
    return output

def list_prod(elements):
    count = 1
    for i in elements:
        count = count * int(i)
    return count

def get_index(tensor, index):
    shape = list(tensor.shape)
    index = int(index)
    ind = []

    for i in range(len(shape)):
        modulo = list_prod(shape[i+1:])
        ind.append(int((index - (index % modulo)) / modulo))
        index = index % modulo
    return ind

filenames = []

for i in range(n):
    newpath = os.path.join(os.path.join(PATH, NAME+str(i)), 'JSON files')
    filenames = filenames + sorted([filename for filename in glob.glob(os.path.join(newpath, '*'))])

names_12 = []
names_16 = []
names_20 = []
names_24 = []

for name in filenames:
    if '12' in name[-10:]:
        names_12.append(name)
    elif '16' in name[-10:]:
        names_16.append(name)
    elif '20' in name[-10:]:
        names_20.append(name)
    else:
        names_24.append(name)

scores_12 = get_scores(names_12)
scores_16 = get_scores(names_16)
scores_20 = get_scores(names_20)
scores_24 = get_scores(names_24)

with open(os.path.join(PATH, 'raw_values_12'), 'w') as file:
    json.dump(scores_12, file)
with open(os.path.join(PATH, 'raw_values_16'), 'w') as file:
    json.dump(scores_16, file)
with open(os.path.join(PATH, 'raw_values_20'), 'w') as file:
    json.dump(scores_20, file)
with open(os.path.join(PATH, 'raw_values_24'), 'w') as file:
    json.dump(scores_24, file)

with open(os.path.join(PATH, 'stats'), 'w') as file:
    scores = torch.Tensor(scores_12).transpose(0,1)
    file.write(get_stats(scores))
    file.write('\n\n')
    scores = torch.Tensor(scores_16).transpose(0,1)
    file.write(get_stats(scores))
    file.write('\n\n')
    scores = torch.Tensor(scores_20).transpose(0,1)
    file.write(get_stats(scores))
    file.write('\n\n')
    scores = torch.Tensor(scores_24).transpose(0,1)
    file.write(get_stats(scores))
    file.write('\n\n')

    scores = torch.cat((torch.Tensor(scores_12).unsqueeze(0),
                        torch.Tensor(scores_16).unsqueeze(0),
                        torch.Tensor(scores_20).unsqueeze(0),
                        torch.Tensor(scores_24).unsqueeze(0)), 0)
    
    scores = scores.transpose(0,2)
    
    dice = float(torch.max(scores[0]))
    acc = float(torch.max(scores[1]))
    auc = float(torch.max(scores[2]))
    
    file.write(f'dice: {dice}; acc: {acc}; auc: {auc}\n')

    a = get_index(scores[0], torch.argmax(scores[0]))
    b = get_index(scores[1], torch.argmax(scores[1]))
    c = get_index(scores[2], torch.argmax(scores[2]))
    file.write(f'indices: {a}, {b}, {c}')
