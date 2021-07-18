import torch
import torch.nn as nn
import os, glob
import random
import matplotlib.pyplot as plt
from torchvision.transforms import RandomRotation as Rotate

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def get_data(n: int):
    train_dir = 'input/drive-full/DRIVE/training'
    test_dir = 'input/drive-full/DRIVE/test'

    print(f'Loading the data... {0}/{6}', end='\r')

    train_img_filenames = sorted([filename for filename in glob.glob(os.path.join(train_dir, 'images', '*.tif'))])
    train_img = torch.Tensor([plt.imread(filename) for filename in train_img_filenames]).transpose(1,3).transpose(-1,-2).to(device) / 255.0

    print(f'Loading the data... {1}/{6}', end='\r')

    train_lbl_filenames = sorted([filename for filename in glob.glob(os.path.join(train_dir, '1st_manual', '*.gif'))])
    train_lbl = torch.Tensor([plt.imread(filename) for filename in train_lbl_filenames]).to(device) / 255.0

    print(f'Loading the data... {2}/{6}', end='\r')

    test_img_filenames = sorted([filename for filename in glob.glob(os.path.join(test_dir, 'images', '*.tif'))])
    test_img = torch.Tensor([plt.imread(filename) for filename in test_img_filenames]).transpose(1,3).transpose(-1,-2).to(device) / 255.0

    print(f'Loading the data... {3}/{6}', end='\r')

    test_lbl_filenames = sorted([filename for filename in glob.glob(os.path.join(test_dir, '1st_manual', '*.gif'))])
    test_lbl = torch.Tensor([plt.imread(filename) for filename in test_lbl_filenames]).to(device) / 255.0

    print(f'Loading the data... {4}/{6}', end='\r')

    test_mask_filenames = sorted([filename for filename in glob.glob(os.path.join(test_dir, 'mask', '*.gif'))])
    test_mask = torch.Tensor([plt.imread(filename) for filename in test_mask_filenames]).to(device) / 255.0

    print(f'Loading the data... {5}/{6}', end='\r')

    train_mask_filenames = sorted([filename for filename in glob.glob(os.path.join(train_dir, 'mask', '*.gif'))])
    train_mask = torch.Tensor([plt.imread(filename) for filename in train_mask_filenames]).to(device) / 255.0

    print(f'Data loaded. {6}/{6}\t\t\t')

    if (n < len(train_img)):
        choices = list(range(len(train_img)))
        random.shuffle(choices)
        new_train_img = torch.Tensor().to(device)
        new_train_lbl = torch.Tensor().to(device)
        new_train_mask = torch.Tensor().to(device)
        for i in range(n):
            new_train_img = torch.cat((new_train_img, train_img[choices[i]].unsqueeze(0)), axis=0)
            new_train_lbl = torch.cat((new_train_lbl, train_lbl[choices[i]].unsqueeze(0)), axis=0)
            new_train_mask = torch.cat((new_train_mask, train_mask[choices[i]].unsqueeze(0)), axis=0)
        train_img = new_train_img
        train_lbl = new_train_lbl
        train_mask = new_train_mask

    return train_img, train_lbl, test_img, test_lbl, test_mask, train_mask

def remove_mask(images, masks):
    images = images.transpose(0,1)
    masks = masks.unsqueeze(0)
    masks = torch.cat((masks, masks, masks), dim=0)

    images[masks == 0] = 0
    images = images.transpose(0,1)

def cut_into_patches(x:torch.Tensor, size=64, stride=48) -> torch.Tensor:
        
    C, x = (x.size(1), x) if len(x.shape) == 4 else (0, x.unsqueeze(1))
    
    x = nn.functional.unfold(x, kernel_size=size, stride=stride)
    x = x.transpose(-1,1)
    
    if C is 0:
        x = x.reshape(x.size(0)*x.size(1), size, size)
    else:
        x = x.reshape(x.size(0)*x.size(1), C, size, size)
    
    return x.contiguous()

def finish_loading(m: int, n: int, d):
    device = d
    train_img, train_lbl, test_img, test_lbl, test_mask, train_mask = get_data(n)

    print('Removing mask...', end='\r')

    remove_mask(train_img, train_mask)
    remove_mask(test_img, test_mask)

    print('Removed masks.\t\t\t')

    n = len(train_img)
    total = m*n

    print(f'Adding {total} rotated images...', end='\r')

    for i in range(n):
        image = train_img[i]
        label = train_lbl[i]
        for j in range(m):
            degree = random.random()*180
            rotation = Rotate((degree, degree))
            train_img = torch.cat((train_img, rotation(image).unsqueeze(0)), 0)
            train_lbl = torch.cat((train_lbl, rotation(label.unsqueeze(0))), 0)
            print(f'Adding {total} rotated images... {i*m + j}/{total}', end='\r')

    print(f'Added {total} rotated images. {total}/{total}\t\t\t')
    print('Cutting images into patches...', end='\r')

    train_img2 = cut_into_patches(train_img).to(device)
    train_lbl2 = cut_into_patches(train_lbl).to(device)

    print('Cut images into patches.\t\t\t')

    # create datasets
    train_set = torch.utils.data.TensorDataset(train_img2, train_lbl2)
    test_set = torch.utils.data.TensorDataset(test_img, test_lbl)
    
    return train_set, test_set