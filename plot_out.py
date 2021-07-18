from modules import ConvectionR2, DilationR2, DiffusionR2, ErosionR2, LinearR2
import torch, os, glob, json, random, argparse
import matplotlib.pyplot as plt
import seaborn as sns

def plot_image(data, name, path):
    new = data.clone().detach().cpu()
    new = new.transpose(0,1).transpose(1,2)
    shape = [int(i/20) for i in list(data.shape)]

    fig = plt.figure()
    fig.set_size_inches((shape[2], shape[1]))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(new, vmin=0, vmax=1, aspect='equal')
    plt.savefig(os.path.join(path, NAME+"_"+name), dpi=120)
    plt.close()

def plot_label(data, name, path):
    new = data.clone().detach().cpu()
    shape = [int(i/20) for i in list(data.shape)]

    fig = plt.figure()
    fig.set_size_inches((shape[1], shape[0]))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(new, vmin=0, vmax=1, aspect='equal')
    plt.savefig(os.path.join(path, NAME+"_"+name), dpi=120)
    plt.close()

def value_map(v, t):
    temp = torch.Tensor(t.shape)
    temp[t == v] = 1
    temp[t != v] = 0
    return temp

def difference(out, cat, v):
    temp = torch.Tensor(out.shape).to(device)
    temp[out >= v] = 0.5
    temp[out < v] = 0
    temp = cat - temp

    white = value_map(0, temp).unsqueeze(0)

    Gtemp = value_map(0.5, temp).unsqueeze(0) + white
    Btemp = value_map(1, temp).unsqueeze(0) + white
    Rtemp = value_map(-0.5, temp).unsqueeze(0) + white

    return torch.cat((Rtemp, Gtemp, Btemp), dim=0).transpose(0,1)

def heatmap(kernel, name, path, type_n, add = ''):
    channels = kernel.shape[0]
    kernel = kernel.tolist()
    newpath = os.path.join(path, type_n+" layer_"+name[5])
    if not os.path.exists(newpath):
            os.makedirs(newpath)

    values = []
    for i in range(channels):
        fig = sns.heatmap(kernel[i], vmin = 0, cbar = False, square=True, xticklabels=False, yticklabels=False).get_figure()
        maximum = max(max(kernel[i]))
        minimum = min(min(kernel[i]))
        values.append([i, maximum, minimum, abs(maximum - minimum)])

        fig.savefig(os.path.join(newpath, name+f'_{i}{add}.png'))
    
    with open(os.path.join(newpath, 'bounds'), 'w') as file:
        json.dump(values, file, indent = 2)

    del values
    del newpath
    del channels
    del kernel

def plot_kernels(model, size, alpha, path):
    for name, param in model.named_parameters():
        if "dil_metric" in name:
            dilation = DilationR2(param.shape[0], size, alpha)
            dilation.dil_metric = param
            heatmap(dilation.get_kernel(param), name, path, "dil")
            del dilation
        elif "diff_metric" in name:
            diffusion = DiffusionR2(param.shape[0], size, alpha)
            diffusion.diff_metric = param
            heatmap(diffusion.get_kernel(param), name, path, "diff")
            del diffusion
        elif "ero_metric" in name:
            erosion = ErosionR2(param.shape[0], size, alpha)
            erosion.ero_metric = param
            heatmap(erosion.get_kernel(param), name, path, "ero")
            del erosion
        elif "metric" in name:
            erosion = ErosionR2(param.shape[0], size, alpha)
            dilation = DilationR2(param.shape[0], size, alpha)
            erosion.ero_metric = param
            dilation.dil_metric = param
            heatmap(erosion.get_kernel(param), name, path, "metric", "_ero")
            heatmap(dilation.get_kernel(param), name, path, "metric", "_dil")
            del erosion
            del dilation

def remove_mask(images, masks):
    images = images.transpose(0,1)
    masks = masks.unsqueeze(0)
    masks = torch.cat((masks, masks, masks), dim=0)

    images[masks == 0] = 0
    images = images.transpose(0,1)

def load_data(i):
    test_dir = 'input/drive-full/DRIVE/test'

    test_img_filenames = sorted([filename for filename in glob.glob(os.path.join(test_dir, 'images', '*.tif'))])
    test_img = torch.Tensor([plt.imread(test_img_filenames[i])]).transpose(1,3).transpose(-1,-2).to(device) / 255.0
    del test_img_filenames

    test_lbl_filenames = sorted([filename for filename in glob.glob(os.path.join(test_dir, '1st_manual', '*.gif'))])
    test_lbl = torch.Tensor([plt.imread(test_lbl_filenames[i])]).to(device) / 255.0
    del test_lbl_filenames

    test_mask_filenames = sorted([filename for filename in glob.glob(os.path.join(test_dir, 'mask', '*.gif'))])
    test_mask = torch.Tensor([plt.imread(test_mask_filenames[i])]).to(device) / 255.0
    del test_mask_filenames

    remove_mask(test_img, test_mask)

    del test_mask
    del test_dir

    return test_img.to(device), test_lbl.to(device)

ap = argparse.ArgumentParser()
req = ap.add_argument_group('required arguments')
req.add_argument(
    '-n', '--name', type=str, required=True,
    help='Give the filename of the model you want to plot.'
)
req.add_argument(
    '-f', '--folder', type=str, required=True,
    help='Folder to store results.'
)
req.add_argument(
    '-m', '--model', type=str, required=True,
    help='Use dil-ero (DE), pde (GEN), or conventional (CON) CNN model.'
)
ap.add_argument(
    '-o', '--output', type=int, default=1,
    help='Whether to plot (1) the output or not (0).'
)
ap.add_argument(
    '-k', '--kernels', type=int, default=1,
    help='Whether to plot (1) the kernels or not (0).'
)
ap.add_argument(
    '-s', '--size', type=int, default=5,
    help='Size of the kernel you would like to plot.'
)
ap.add_argument(
    '-a', '--alpha', type=float, default=0.65,
    help='Alpha used for the kernels.'
)
ap.add_argument(
    '-i', '--index', type=int, default=1,
    help='Index of the image you want to test on.'
)
ap.add_argument(
    '-t', '--threshold', type=float, default=0.5,
    help='Threshold used for the color coded image.'
)
ap.add_argument(
    '-d', '--device', type=str, default='GPU',
    help='device to run code with (GPU or CPU).'
)
args = vars(ap.parse_args())

if args['model'] == 'GEN':
    from CDEPdeCNNge import CDEPdeCNN as CNNmodel
elif args['model'] == 'DE':
    from CDEPdeCNNed import CDEPdeCNN as CNNmodel
else:
    from ConvCNN import ConventionalCNN as CNNmodel

if torch.cuda.is_available() and args['device'] == 'GPU':
    device = torch.device('cuda')
    print('Using CUDA')
elif not torch.cuda.is_available() and args['device'] == 'GPU':
    device = torch.device('cpu')
    print('GPU unavailable, using CPU.')
    print('Either no GPU is available or packages are missing. Make sure everything is installed at')
    print('https://varhowto.com/install-pytorch-cuda-10-2/')
else:
    device = torch.device('cpu')
    print('Using CPU')

PATH = 'input\\models'
NAME = args['name']
path = args['folder']

if bool(args['kernels']) or bool(args['output']):
    model = torch.load(os.path.join(PATH, NAME+'.pt'))

    path = os.path.join("test", path)
    if not os.path.exists(path):
        os.makedirs(path)

if bool(args['kernels']):
    plot_kernels(model, args['size'], args['alpha'], path)

if bool(args['output']):
    i = args['index']
    test, t_ca = load_data(i)

    output = model(test)

    graph = difference(output.squeeze(1), t_ca[0], args['threshold'])

    plot_image(test[0], "input_"+str(i), path)
    # plot_image(output[0], "out_"+str(i), path)
    plot_image(graph[0], "diff_"+str(i), path)
    # plot_label(t_ca[0], "label_"+str(i), path)