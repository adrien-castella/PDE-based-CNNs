import os, json
import numpy as np

IND = int(input("Give metric: "))

folder = "conv-dil-dil"

folder = [folder + " " + str(i) for i in range(4)]
path = [os.path.join("output", i) for i in folder]

bound = float(input("Give model bound: "))

best = []
for i in path:
    details = []
    with open(os.path.join(i, 'listed_details'), 'r') as file:
        details = json.load(file)
    config = []
    with open(os.path.join(i, 'conf_code'), 'r') as file:
        config = json.load(file)
    
    temp = set([])
    for j in range(len(details)):
        if details[j][IND] > bound:# and "000" in config[j]["name"]
            temp.add(config[j]["name"][4:])
    best.append(temp)
    del temp

details = []
for i in path[:-1]:
    with open(os.path.join(i, 'listed_details'), 'r') as file:
        details.append(json.load(file))

other_d = []
with open(os.path.join(path[-1], 'listed_details'), 'r') as file:
    other_d = np.array(json.load(file))
other_c = []
with open(os.path.join(path[-1], 'conf_code'), 'r') as file:
    other_c = json.load(file)

details = np.array(details)
details = np.mean(details, 0)

config = []
with open(os.path.join(path[0], 'conf_code'), 'r') as file:
    config = json.load(file)

j = 0
for i in range(len(config)):
    if config[i]['name'] == other_c[j]['name']:
        details[i] = details[i]*(3/4) + (1/4)*other_d[j]
        j = j + 1

average = set()
for i in range(len(config)):
    if details[i][IND] > bound:
        average.add(config[i]['name'][4:])

print(average)
print(len(average))
print()
output = best[0].union(best[1].union(best[2].union(best[3])))
print(output)
print(len(output))

print(best[0])
print(best[1])
print(best[2])
print(best[3])
print()
# print(best[0].intersection(best[1]))
# print(best[0].intersection(best[2]))
# print(best[1].intersection(best[2]))
# print(best[1].intersection(best[3]))
# print(best[2].intersection(best[3]))
# print(best[0].intersection(best[3]))
print()
output = best[3]
for i in range(3):
    output = output.intersection(best[i])
print(output)
print(len(output))
# print(best[0].intersection(best[1].intersection(best[2])))