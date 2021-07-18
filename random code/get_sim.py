import os, json
import numpy as np

folder = "conv-dil-dil"

folder = [folder + " " + str(i) for i in range(9)]
path = [os.path.join("output", i) for i in folder]

configuration = input("Give configuration name: ")

stuff = []
for i in path:
    with open(os.path.join(i, 'listed_details'), 'r') as file:
        details = json.load(file)
    config = []
    with open(os.path.join(i, 'conf_code'), 'r') as file:
        config = json.load(file)
    
    for j in range(len(config)):
        if configuration in config[j]["name"]:
            configuration = config[j]["name"]
            stuff.append(details[j])

with open(os.path.join(path[1], 'JSON files', configuration+'_details'), 'r') as file:
    params = int(file.read().split('\n')[7].split(': ')[-1])
    print(params)

stuff = np.array(stuff)
print(stuff)
print(np.mean(stuff, 0))