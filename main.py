import argparse, torch, json, os, math
import numpy as np
from skimage.util import random_noise
import matplotlib.pyplot as plt
from run import run_model

# Listing parameters
ap = argparse.ArgumentParser()
req = ap.add_argument_group('required arguments')
req.add_argument(
    '-c', '--configuration', type=str, required=True,
    help='Choose a configuration file. You can create one using the configuration code.'
)
req.add_argument(
    '-f', '--folder', type=str, required=True,
    help='folder to store results.'
)
ap.add_argument(
    '-d', '--device', type=str,
    default='GPU', help='device to run code with (GPU or CPU).'
)
ap.add_argument(
    '-m', '--model', type=str,
    default='GEN', help='Use dil-ero (DE), pde (GEN), or conventional (CON) CNN model.'
)
ap.add_argument(
    '-r', '--rotated', type=int,
    default=0, help='Add n rotated images.'
)
ap.add_argument(
    '-n', '--noise', type=int, default=0,
    help='Add (1) or dont add (0) noise to the training and testing images.'
)
ap.add_argument(
    '-t', '--train', type=int, default=math.inf,
    help='Reduce the training data by providing an integer smaller than the training set.'
)
args = vars(ap.parse_args())


# Checking for availability of device
if torch.cuda.is_available() and args['device'] == 'GPU':
    device = torch.device('cuda')
    print('Using CUDA')
elif not torch.cuda.is_available() and args['device'] == 'GPU':
    device = torch.device('cpu')
    print('GPU unavailable, using CPU.')
    print('Either no GPU is available or packages are missing from:')
    print('https://varhowto.com/install-pytorch-cuda-10-2/')
elif agrs['device'] == 'CPU':
    device = torch.device('cpu')
    print('Using CPU')
else:
    raise f'{args["device"]} is not an option for the device choice! Try instead GPU or CPU.'

# Loading data
import load_data
train_set, test_set = load_data.finish_loading(args['rotated'], args['train'], device)

# old implementation network import
if args['model'] == 'GEN':
    from CDEPdeCNNge import CDEPdeCNN as CNNmodel
elif args['model'] == 'DE':
    from CDEPdeCNNed import CDEPdeCNN as CNNmodel
elif args['model'] == 'CON':
    from ConvCNN import ConventionalCNN as CNNmodel
else:
    raise f'{args["model"]} is not an option for the model choice! Try instead GEN, DE, or CON.'

# new implementation network import
from CDEPdeCNNge import CDEPdeCNN as GEN
from CDEPdeCNNed import CDEPdeCNN as DE
from ConvCNN import ConventionalCNN as CON

PATH = 'output'
NAME = args['folder']
BATCH_SIZE = 12

PATH = os.path.join(PATH, NAME)
if not os.path.exists(PATH):
    os.makedirs(PATH)

newpath = os.path.join(PATH, 'images')
if not os.path.exists(newpath):
    os.makedirs(newpath)
newpath = os.path.join(PATH, 'JSON files')
if not os.path.exists(newpath):
    os.makedirs(newpath)
newpath = os.path.join(PATH, 'times')
if not os.path.exists(newpath):
    os.makedirs(newpath)

# Creating data loaders
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=BATCH_SIZE, shuffle=True
)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)

# Adding noise to images
if bool(args['noise']):
    for data in train_loader:
        img, _ = data[0].cpu(), data[1]
        img = torch.Tensor(random_noise(img, mode='speckle', mean=0, var=0.01, clip=True))
        img = torch.Tensor(random_noise(img, mode='gaussian', mean=0, var=0.01, clip=True))
        data[0] = img.to(device)

    for data in test_loader:
        img, _ = data[0].cpu(), data[1]
        img = torch.Tensor(random_noise(img, mode='speckle', mean=0, var=0.01, clip=True))
        img = torch.Tensor(random_noise(img, mode='gaussian', mean=0, var=0.01, clip=True))
        data[0] = img.to(device)


L2_LOSS_MULTIPLIER = 0.005

# Loading specified configuration +
# Creating detail files in the designated folder
with open(os.path.join(PATH, 'arguments'), 'w') as file:
    json.dump(args, file, indent=2)

try:
    with open(os.path.join('input', 'configurations', args['configuration']), 'r') as file:
        config = json.load(file)
except:
    raise f'Configuration {args["configuration"]} does not exist. Use show_config.py to see available configurations or create a new one with configurations.py.'

with open(os.path.join(PATH, 'conf_code'), 'w') as file:
    json.dump(config, file)

with open(os.path.join(PATH, 'listed_details'), 'w') as file:
    file.write("[")


# Running each of the simulations in order
j = 0
for i in config:
    print("Starting configuration " + str(j+1) + " / " + str(len(config)))
    if not 'model' in i:
        run_model(device, CNNmodel, train_loader, test_loader, PATH, i, j, len(config), L2_LOSS_MULTIPLIER, len(train_set))
    elif i['model'] == 'DE':
        run_model(device, DE, train_loader, test_loader, PATH, i, j, len(config), L2_LOSS_MULTIPLIER, len(train_set))
    elif i['model'] == 'GEN':
        run_model(device, GEN, train_loader, test_loader, PATH, i, j, len(config), L2_LOSS_MULTIPLIER, len(train_set))
    elif i['model'] == 'CON':
        run_model(device, CON, train_loader, test_loader, PATH, i, j, len(config), L2_LOSS_MULTIPLIER, len(train_set))
    else:
        raise f"No model of type {i['model']} exists. Try instead 'CON', 'GEN', or 'DE'."
    j = j + 1


# Completing the detail files and adding the max index file
with open(os.path.join(PATH, 'listed_details'), 'r') as file:
    data = file.read()[:-1] + ']'

with open(os.path.join(PATH, 'listed_details'), 'w') as file:
    file.write(data)

data = np.array(json.loads(data))

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