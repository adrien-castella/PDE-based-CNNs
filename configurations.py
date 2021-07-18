import os, json, math, copy, argparse

ap = argparse.ArgumentParser()
req = ap.add_argument_group('required arguments')
req.add_argument(
    '-n', '--name', type=str, required=True,
    help='Name of the configuration.'
)
req.add_argument(
    '-t', '--type', type=str, required=True,
    help='General (GEN), dil-ero (DE), or convectional (CON) CNN.'
)
ap.add_argument(
    '-c', '--runs', type=int, default=1,
    help='The number of times to run these configurations.'
)
ap.add_argument(
    '-e', '--epochs', type=int, default=75,
    help='The number of epochs used in each configuration.'
)
ap.add_argument(
    '-f', '--first', type=str, default='d',
    help='Module in the first layer (d or e).'
)
ap.add_argument(
    '-l', '--last', type=str, default='e',
    help='Module in the last layer (d or e).'
)
ap.add_argument(
    '-g', '--gamma', type=float, default=0.96,
    help='Multiplier of the learning rate.'
)
ap.add_argument(
    '-r', '--rate', type=float, default=0.01,
    help='Learning rate for the network.'
)
args = vars(ap.parse_args())

dilero = [args['first'], args['last']]
epochs = args['epochs']
rate = args['rate']
gamma = args['gamma']

configurations = {
    "name": [],
    "epochs": [],
    "rate": [],
    "gamma": [],
    "layers": [],
    "components": [],
    "title": [],
    "channels": [],
    "alpha": [],
    "convection": [],
    "size": [],
    "dil-ero": [],
    "model": []
}

def toSet(string: str, t: type):
    string = string.replace(' ', '')
    if not (string[0]=='{' and string[-1]=='}'):
        print('Syntax Exception: The input is not a set.')
        raise
    
    string = string.strip('{')
    string = string.strip('}')
    string = string.split(',')

    output = set()
    for i in string:
        try:
            output.add(t(i))
        except:
            print('Type error: The items are not the correct type.')
            raise
    
    return output

def toList(string: str, t: type):
    string = string.replace(' ', '')
    if not (string[0]=='[' and string[-1]==']'):
        print('Syntax Exception: The input is not a list.')
        raise
    
    string = string.strip('[')
    string = string.strip(']')
    string = string.split(',')

    output = []
    for i in string:
        try:
            output.append(t(i))
        except:
            print('Type error: The items are not the correct type.')
            raise
    
    return output

def get(items, toType, string, t):
    print(items, " ", t)
    choice = input(string)
    if len(choice) == 0:
        return items
    else:
        return toType(choice, t)

def fill(string: str, i: int):
    return '0'*(i - len(string)) + string

def get_comps(i: int):
    nums = set()
    for j in range(int(math.pow(2,i))):
        nums.add(fill(bin(j)[2:], i))
    return nums

def de(configurations):
    comp = get_comps(6)

    layers = [5]; alpha = [0.65]; convection = [True]; channels = [12]; loss = 0; unwanted = set()
    layers = get(layers, toList, 'Give a layer list: ', int)
    alpha = get(alpha, toList, 'Give an alpha list: ', float)
    channels = get(channels, toList, 'Give a channel list: ', int)
    convection = get(convection, toList, 'Give convection list: ', bool)
    if (bool(int(input('Use complement (0) of unwanted or not (1)? ')))):
        unwanted = get(unwanted, toSet, 'Give unwanted set: ', str)
    else:
        unwanted = get(unwanted, toSet, 'Give wanted set: ', str)
        unwanted = comp - unwanted
    convection = list(set(convection))

    comp = comp - unwanted
    for i in layers:
        for j in comp:
            for k in channels:
                for l in alpha:
                    for m in convection:
                        # if not ('1' in j[:2] and '1' in j[2:4] and '1' in j[4:]):
                        #     continue
                        
                        configurations['layers'].append(i)
                        configurations['channels'].append(k)
                        configurations['alpha'].append(l)
                        configurations['size'].append(5)
                        configurations['rate'].append(rate)
                        configurations['gamma'].append(gamma)
                        configurations['components'].append([bool(int(n)) for n in j])
                        configurations['convection'].append(m)
                        configurations['dil-ero'].append(dilero)
                        configurations['epochs'].append(epochs)
                        configurations['model'].append('DE')

                        name = []
                        if len(layers) > 1:
                            name.append(''.join(str(i).split('.')))
                        if len(alpha) > 1:
                            name.append(''.join(str(l).split('.')))
                        if len(convection) > 1:
                            name.append(''.join(str(m).split('.')))
                        if len(channels) > 1:
                            name.append(''.join(str(k).split('.')))
                        
                        if len(name) > 0:
                            name = '_'.join([j, '_'.join(name)])
                        else:
                            name = j
                        configurations['name'].append(name)
                        configurations['title'].append(' '.join(name.split('_')))
    
    return configurations

def gen(configurations):
    comp = get_comps(4)
    configurations['loss'] = []; configurations['mult'] = []

    channels = [12]; alpha = [0.65]; size = [5]; loss = [0]; mult = [0.1]; unwanted = set()
    alpha = get(alpha, toList, 'Give an alpha list: ', float)
    channels = get(channels, toList, 'Give a channel list: ', int)
    size = get(size, toList, 'Give size list: ', int)
    loss = get(loss, toList, 'Give list of diffusion bounds to use: ', int)
    mult = get(mult, toList, 'Multipliers for the bound loss factor: ', float)
    if (bool(int(input('Use complement (0) of unwanted or not (1)? ')))):
        unwanted = get(unwanted, toSet, 'Give unwanted set: ', str)
    else:
        unwanted = get(unwanted, toSet, 'Give wanted set: ', str)
        unwanted = comp - unwanted
    
    comp = comp - unwanted
    for i in channels:
        for j in alpha:
            for k in size:
                for l in comp:
                    for m in loss:
                        for n in mult:
                            configurations['layers'].append(5)
                            configurations['channels'].append(i)
                            configurations['alpha'].append(j)
                            configurations['size'].append(k)
                            configurations['rate'].append(rate)
                            configurations['gamma'].append(gamma)
                            configurations['components'].append([bool(int(n)) for n in l])
                            configurations['convection'].append(True)
                            configurations['dil-ero'].append(dilero)
                            configurations['model'].append('GEN')
                            configurations['loss'].append(m)
                            configurations['mult'].append(n)
                            configurations['epochs'].append(epochs)

                            name = []
                            if len(channels) > 1:
                                name.append(''.join(str(i).split('.')))
                            if len(alpha) > 1:
                                name.append(''.join(str(j).split('.')))
                            if len(size) > 1:
                                name.append(''.join(str(k).split('.')))
                            if len(loss) > 1:
                                name.append('loss_'+str(m))
                            if len(mult) > 1:
                                name.append(''.join(str(n).split('.')))
                            
                            if len(name) > 0:
                                name = '_'.join([l, '_'.join(name)])
                            else:
                                name = l
                            configurations['name'].append(name)
                            configurations['title'].append(' '.join(name.split('_')))
    
    return configurations

def con(configurations):
    layers = [5]; channels = [12]
    channels = get(channels, toList, 'Give a channel list: ', int)
    layers = get(layers, toList, 'Give a layer list: ', int)

    for i in channels:
        for j in layers:
            configurations['epochs'].append(epochs)
            configurations['rate'].append(rate)
            configurations['gamma'].append(gamma)
            configurations['dil-ero'].append(dilero)
            configurations['alpha'].append(1)
            configurations['size'].append(5)
            configurations['convection'].append(False)
            configurations['components'].append([])
            configurations['channels'].append(i)
            configurations['layers'].append(j)
            configurations['model'].append('CON')

            name = []
            if len(channels) > 1:
                name.append(str(i))
            if len(layers) > 1:
                name.append(str(j))
            
            if len(name) > 0:
                name = '_'.join(['CNN', '_'.join(name)])
            else:
                name = 'CNN'
            
            configurations['name'].append(name)
            configurations['title'].append(' '.join(name.split('_')))

    return configurations

if args['type'] == 'GEN':
    configurations = gen(configurations)
elif args['type'] == 'DE':
    configurations = de(configurations)
else:
    configurations = con(configurations)

config = [dict(zip(configurations,t)) for t in zip(*configurations.values())]

new = copy.deepcopy(config)
for i in range(args['runs'] - 1):
    config = config + copy.deepcopy(new)

if args['runs'] > 1:
    j = 0
    for i in range(len(config)):
        config[i]["name"] = config[i]["name"] + "_" + str(j)
        j = j + 1

with open(os.path.join('input', 'configurations', args['name']), 'w') as file:
    json.dump(config, file)