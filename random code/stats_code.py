import os, json
import numpy as np

def stats(configs, details):
    details = np.array(details)
    average = np.mean(details, axis=0)
    return [configs, details, average]

def other(one, two, x):
    return [np.sum(one[1] > two[1], axis=0).tolist() + [len(one[1])], [one[2].tolist(), two[2].tolist()], x]

def get_binary(comp):
    return ''.join([str(int(i)) for i in comp])

FOLDER = os.path.join("output", input("Give the folder name: "))

configs = []
with open(os.path.join(FOLDER, "conf_code"), "r") as file:
    configs = json.loads(file.read())

details = []
with open(os.path.join(FOLDER, "listed_details"), "r") as file:
    details = json.loads(file.read())

for i in configs:
    i["components"] = get_binary(i["components"])

structures = {configs[i]["components"] for i in range(len(configs))} - {'1000'}

result = []
for x in structures:
    if x[0] == '1':
        configurations = []
        listed_details = []
        for i in range(len(configs)):
            if configs[i]["components"] == x:
                configurations.append(configs[i])
                listed_details.append(details[i])
        
        result.append([stats(configurations, listed_details), x])

n = len(result)
for i in range(n):
    x = result[i]
    configurations = []
    listed_details = []
    for i in range(len(configs)):
        if configs[i]["components"] == '0' + x[1][1:]:
            configurations.append(configs[i])
            listed_details.append(details[i])
    
    result.append([stats(configurations, listed_details), '0' + x[1][1:]])

output = []

n = int(len(result) / 2)
for i in range(n):
    output.append(other(result[i][0], result[i+n][0], result[i][1][1:]))

with open(os.path.join(FOLDER, 'stats'), 'w') as file:
    json.dump(output, file, indent = 2)