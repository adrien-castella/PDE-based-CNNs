from modules import DiffusionR2, inf_convolution_2d, convolution_2d
import os, torch, json, glob
import seaborn as sns
import matplotlib.pyplot as plt
from torch import nn
from skimage.util import random_noise
import numpy as np

def heatmap(kernel, name, path, type_n, add = ''):
    t = ''.join(str(time).split('.'))
    channels = kernel.shape[0]
    kernel = kernel.tolist()

    values = []
    for i in range(channels):
        fig = sns.heatmap(kernel[i], vmin = 0, cbar = False, square=True, xticklabels=False, yticklabels=False).get_figure()
        maximum = max(max(kernel[i]))
        minimum = min(min(kernel[i]))
        values.append([i, maximum, minimum, abs(maximum - minimum)])

        fig.savefig(os.path.join(path, name+' '+t+'.png'))
    
    with open(os.path.join(path, 'bounds '+t), 'w') as file:
        json.dump(values, file, indent = 2)

def plot_image(data, name, path, use = True):
    if use:
        t = '_'+''.join(str(time).split('.'))
    else:
        t = ''
    new = data.clone().detach().cpu()
    new = new.transpose(0,1).transpose(1,2)
    # shape = [int(i/10) for i in list(data.shape)]

    fig = plt.figure()
    # fig.set_size_inches((shape[2], shape[1]))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.imshow(new, vmin=0, vmax=1, aspect='equal')
    plt.savefig(os.path.join(path, name+t))#, dpi=60)
    plt.close()

def plot_kernels(size, alpha, path, img):
    module = DiffusionR2(3, size, alpha).to(device)
    # iden = torch.eye(2).to(device).unsqueeze(0)
    iden = torch.Tensor([[1,0], [0,1]]).to(device).unsqueeze(0)
    iden = torch.cat((iden, iden, iden), 0)
    kernel = module.get_kernel(iden, time)
    heatmap(kernel, "Struct_func", path, "diff")

    plot_image(img[0], "original", path, False)
    # output = -inf_convolution_2d(-img, kernel)
    # output = inf_convolution_2d(img, kernel)
    # output = torch.Tensor(np.fft.fft(img.cpu())).to(device)
    output = convolution_2d(img, kernel)
    # print(torch.max(output))
    # print(torch.min(output))
    plot_image(output[0], "output", path)

    print(img.shape)
    print(output.shape)

def remove_mask(images, masks):
    images = images.transpose(0,1)
    masks = masks.unsqueeze(0)
    masks = torch.cat((masks, masks, masks), dim=0)

    images[masks == 0] = 0
    images = images.transpose(0,1)

def load_data(i):
    test_dir = 'input/drive-full/DRIVE/test'

    test_img = torch.Tensor([plt.imread(os.path.join(test_dir, 'diff_image.tif'))]) / 255.0
    print(test_img.shape)
    test_img = test_img.transpose(-2,-1).transpose(-3,-2)
    print(test_img.shape)
    # noise = torch.rand(test_img.shape)
    # noise[True] = 0
    # noise = torch.Tensor(random_noise(noise, mode='s&p', amount=1, clip=True))
    # noise = torch.Tensor(1) - torch.Tensor(random_noise(noise, mode='gaussian', mean=0, var=0.001, clip=True))
    # test_img = test_img + noise
    # test_img[test_img > 1] = 1
    # test_img[test_img < 0] = 0
    
    test_img = test_img.to(device)
    print(test_img.shape)

    del test_dir
    return test_img

def cut_into_patches(x:torch.Tensor, size=64, stride=48) -> torch.Tensor:
        
    C, x = (x.size(1), x) if len(x.shape) == 4 else (0, x.unsqueeze(1))
    
    x = nn.functional.unfold(x, kernel_size=size, stride=stride)
    x = x.transpose(-1,1)
    
    if C is 0:
        x = x.reshape(x.size(0)*x.size(1), size, size)
    else:
        x = x.reshape(x.size(0)*x.size(1), C, size, size)
    
    return x.contiguous()

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Using CUDA')
else:
    device = torch.device('cpu')
    print('Using CPU')

folder = input("Give folder name: ")
size = int(input("Give the size: "))
alpha = float(input("Give alpha: "))

time = float(input("Give time: "))

img = load_data(1)
# img = cut_into_patches(img)[64].unsqueeze(0).to(device)

PATH = os.path.join("test", folder)
if not os.path.exists(PATH):
    os.makedirs(PATH)
plot_kernels(size, alpha, PATH, img)