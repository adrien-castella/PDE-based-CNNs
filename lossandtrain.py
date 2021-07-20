import sklearn.metrics
import time, math
import torch
import numpy as np

"""
    Prints a command line progress bar.
    
    @parameters
        batch (int)     - how many batches have been processed
        epoch (int)     - how many epochs have been trained
        batches (int)   - the total number of batches that will be used
        epochs (int)    - the total number of epochs that will be trained
        name (str)      - name of the "batch" (or "test") counter
        fill (str)      - character that fills the progress bar
        l_1 (int)       - length of the "batch" (or "test") progress bar
        l_2 (int)       - length of the epoch progress barx

    WARNING: avoid printing other things in the command line during use. Errors will definitely occur.
"""
def progressBar(batch, epoch, batches=202, epochs=75, name='Batch', fill='█', l_1=20, l_2=10):
    output = 'Progress: '
    b_1 = '|' + int(epoch * l_1 / epochs) * fill + '-'*(l_1 - int(epoch * l_1 / epochs)) + '| '
    b_2 = '|' + int(batch * l_2 / batches) * fill + '-'*(l_2 - int(batch * l_2 / batches)) + '| '
    p = ("{0:.1f}").format(100 * epoch / float(epochs)) + '% '

    output = output + b_1 + p + f'Epoch: {epoch}/{epochs}     '
    output = output + b_2 + f'{name}: {batch}/{batches}       '
    print(output, end='\r')

def dice_loss(input, target, smooth=1):
    """
    Continuous DICE coefficient suitable for use as a loss function for binary segmentation, calculated as:
    $$
        1 - \\frac{2*(input*target).sum()+smooth}{input.sum()+target.sum()+smooth}
    $$
    
    Parameters
    ------------
    input: torch.Tensor
    Tensor of any shape with scalar elements in the range [0,1].

    target: torch.Tensor
    Label tensor with the same shape as the input with elements in the set {0,1}.

    smooth: float
    Smoothing factor that is added to both the numerator and denominator to avoid divide-by-zero.
    
    """
    AinterB = (input.view(-1) * target.view(-1)).sum()
    A = input.view(-1).sum()
    B = target.view(-1).sum()
    dice = (2 * AinterB + smooth) / (A + B + smooth)
    return 1 - dice

def loss(model, device, output, y, total_params: int, L2: float, config: dict):
    """
        Calculate loss including regularization loss.
    """
    data_loss = dice_loss(output, y)
    
    l2_loss = torch.tensor(0.0, device=device)
    for p in model.parameters():
        l2_loss += p.pow(2).sum()

    l2_loss = l2_loss / total_params

    # computes the Diffusion coefficient if a norm is used
    if not ((not 'loss' in config) or config['loss'] == 0):
        for n, p in model.named_parameters():
            if "diff_metric" in n:
                if config['loss'] == 1:
                    norms = torch.max(torch.abs(torch.inverse(p.transpose(1,2) @ p)))
                elif config['loss'] == 2:
                    norms = torch.pow(torch.linalg.norm((p.transpose(1,2) @ p) - torch.eye(2).to(device)), 6)
                elif config['loss'] == 3:
                    norms = torch.pow(torch.abs((p.transpose(1,2) @ p) - torch.eye(2).to(device)), 6)
                    norms = torch.max(torch.pow(torch.sum(torch.sum(norms, axis=1), axis=1), 1/6))
                elif config['loss'] == 4:
                    norms = torch.linalg.norm(p.transpose(1,2) @ p - torch.eye(2).to(device)) / torch.max(torch.abs(p.transpose(1,2) @ p))
                    norms = torch.pow(norms, 6)
                else:
                    raise 'Non-existing bound.'

                l2_loss += config['mult'] * norms
    
    return data_loss + L2 * l2_loss

def train(model, device, train_loader, optimizer, total_params: int, L2: float, epoch: int, config: dict):
    """
        Train one epoch
    """
    batch_list = []
    
    model.train()

    total = len(train_loader)
    time_list = []

    norms = []
    for batch_idx, (x, y) in enumerate(train_loader):
        progressBar(batch_idx, epoch, total, config['epochs'])
        start = time.perf_counter()
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        batch_loss = loss(model, device, output, y, total_params, L2, config)
        batch_list.append(batch_loss.item())
        batch_loss.backward()
        optimizer.step()
        stop = time.perf_counter()
        time_list.append(stop - start)

    return np.mean(batch_list), sum(time_list)

def test(model, device, test_loader, total_params: int, L2: float, epoch: int, config: dict):
    """
        Evaluate the model
    """
    model.eval()
    test_loss = []
    acc_score = []
    auc_score = []
    dice_score = []

    total = len(test_loader)

    start = time.perf_counter()

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            progressBar(batch_idx, epoch, total, config['epochs'], name='Test')
            x, y = x.to(device), y.to(device)

            output = model(x)

            test_loss.append(loss(model, device, output, y, total_params, L2, config)[0].item())
            y = y.cpu().view(-1)
            prediction = output.round().cpu().view(-1)
            auc_score.append(sklearn.metrics.roc_auc_score(y, prediction))
            acc_score.append(sklearn.metrics.accuracy_score(y, prediction))
            dice_score.append(
                float((2 * torch.sum(prediction * y)).item())
                / float((prediction.sum() + y.sum()).item())
            )

    stop = time.perf_counter()
    process_time = float(stop - start)

    test_loss = np.mean(test_loss)
    auc_score = np.mean(auc_score)
    acc_score = np.mean(acc_score)
    dice_score = np.mean(dice_score)
    
    return dice_score, acc_score, auc_score, test_loss, process_time