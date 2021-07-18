import os, glob, json
import numpy as np
from configurations_ed import config

def read_json(PATH):
    newpath = os.path.join(PATH, 'JSON files')
    detail_filenames = [filename for filename in glob.glob(os.path.join(newpath, '*'))]

    print(detail_filenames)

    data = []
    for name in detail_filenames:
        with open(name, 'r') as file:
            current = file.read()
        
        current = current.split('\n')
        current = [current[2], current[4], current[6]]

        stuff = []
        for i in current:
            c = i.split(' ')
            stuff.append(float(c[2][:-1]))
        
        data.append(stuff)
    
    return data


NAME = input('Give the folder name: ')
PATH = os.path.join('output', NAME)

choice = input('Use configuration (0) or file (1)? ')
if '1' in choice:
    with open(os.path.join(PATH, 'conf_code'), 'r') as file:
        config = json.load(file)

choice = input('Use listed_details (0) or JSON files (1)? ')
if '1' in choice:
    data = read_json(PATH)
    with open(os.path.join(PATH, 'listed_details'), 'w') as file:
        json.dump(data, file)
else:
    with open(os.path.join(PATH, 'listed_details'), 'r') as file:
        data = file.read()[:-1] + ']'
        data = json.loads(data)

data = np.array(data)
data = data.transpose()
dice = np.argmax(data[0])
acc = np.argmax(data[1])
auc = np.argmax(data[2])

with open(os.path.join(PATH, 'max_index'), 'w') as file:
    file.write('dice\n')
    json.dump(config[dice], file, indent=2)
    file.write(f'\nscore: {data[0][dice]}\n\n')
    file.write('acc\n')
    json.dump(config[acc], file, indent=2)
    file.write(f'\nscore: {data[1][acc]}\n\n')
    file.write('auc\n')
    json.dump(config[auc], file, indent=2)
    file.write(f'\nscore: {data[2][auc]}')