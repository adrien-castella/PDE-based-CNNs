from lossandtrain import train, test
import torch, json, os, time
import matplotlib.pyplot as plt
import numpy as np

"""
    prints the progress bar at 100% to indicate that the simulation is finished
    
    @parameters
        acc (float)     -
        auc (float)     -
        dice (float)    -
        fill (str)      -
        length (int)    -
"""
def finalBar(acc, auc, dice, fill='█', length=20):
    output = 'Progress: |' + length * fill + '| 100% Complete.\t\t\t\t\t\t\t\n'
    output = output + f'acc = {acc}   auc = {auc}   dice = {dice}'
    print(output)

"""
    runs the configuration, plots the epochs, and outputs the JSON files
    
    @parameters
        device (torch.device)       - the torch device being used (GPU or CPU)
        MODEL (class)               - the model type (CDEPdeCNNed, CDEPdeCNNge, or ConvCNN)
        train_loader (DataLoader)   - loader containing the training data
        test_loader (DataLoader)    - loader containing the test data
        PATH (str)                  - the location for storing the output
        conf (dict)                 - specifying the configurations
        j (int)                     - configuration idnex
        tot (int)
        L2 (float)                  - multiplier for the parameters in the loss
        im (int)                    - number of images for training
"""
def run_model(device, MODEL, train_loader, test_loader, PATH, conf: dict, j: int, tot: int, L2: float, im: int):
    print(conf)
    EPOCHS = conf["epochs"]
    
    LR = conf["rate"]
    LR_GAMMA = conf["gamma"]
    NAME = conf["name"]
    
    TEST_DICE_LIST = []
    TEST_ACC_LIST = []
    TEST_AUC_LIST = []
    TRAIN_LOSS_LIST = []
    TEST_LOSS_LIST = []
    CONV_VECTORS = []
    TEST_TIME = []
    TRAIN_TIME = []

    modelpath = os.path.join(PATH, conf["name"])
    if not os.path.exists(modelpath):
        os.makedirs(modelpath)
    
    # instanciate model
    
    model = MODEL(conf).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, LR_GAMMA)
    
    total_params = sum(p.numel() for p in model.parameters(recurse=True))
    
    print(
        f"Model: {MODEL.__module__}."
        + f"{MODEL.__name__}"
        + f" with {total_params} parameters"
    )
    
    print(f"Conf: {j+1}/{tot}")
    for epoch in range(EPOCHS):
        loss_v, train_t = train(model, device, train_loader, optimizer, 
                                total_params, L2, epoch, conf)
        TRAIN_LOSS_LIST.append(loss_v)
        TRAIN_TIME.append(train_t)
        output = test(model, device, test_loader, total_params, L2,
                      epoch, conf)
        TEST_DICE_LIST.append(output[0])
        TEST_ACC_LIST.append(output[1])
        TEST_AUC_LIST.append(output[2])
        TEST_LOSS_LIST.append(output[3])
        TEST_TIME.append(output[4])
        num = '0'+str(epoch+1) if epoch < 9 else str(epoch+1)
        torch.save(model, os.path.join(modelpath, num+'_'+NAME+'.pt'))
        scheduler.step()

    val_dice = max(TEST_DICE_LIST)
    ind_dice = TEST_DICE_LIST.index(val_dice)
    val_acc = max(TEST_ACC_LIST)
    ind_acc = TEST_ACC_LIST.index(val_acc)
    val_auc = max(TEST_AUC_LIST)
    ind_auc = TEST_AUC_LIST.index(val_auc)

    # remove in case of one configuration
    finalBar(val_acc, val_auc, val_dice)

    newpath = os.path.join(PATH, 'JSON files')
    with open(os.path.join(newpath, conf["name"]+"_details"), 'w') as file:
        file.write(f"Number of images: {im}\n")
        file.write("Dice\n")
        file.write(f"Max value: {val_dice}; Max index: {ind_dice+1}\n")
        file.write("Acc\n")
        file.write(f"Max value: {val_acc}; Max index: {ind_acc+1}\n")
        file.write("Auc\n")
        file.write(f"Max value: {val_auc}; Max index: {ind_auc+1}\n")
        file.write(f"Total parameters: {total_params}\n")
        file.write(f"Average test time: {np.mean(TEST_TIME)}\n")
        file.write(f"Average train time: {np.mean(TRAIN_TIME)}\n")
    
    with open(os.path.join(PATH, 'listed_details'), 'r+') as file:
        file.read()
        file.write(f"[{val_dice}, {val_acc}, {val_auc}],")
    
    newpath = os.path.join(PATH, 'times')
    with open(os.path.join(newpath, conf["name"]+'_test'), 'w') as file:
        json.dump(TEST_TIME, file)
    with open(os.path.join(newpath, conf["name"]+'_train'), 'w') as file:
        json.dump(TRAIN_TIME, file)

    plot_loss(TRAIN_LOSS_LIST, TEST_LOSS_LIST, EPOCHS, conf, PATH)
    plot_test(TEST_DICE_LIST, TEST_ACC_LIST, TEST_AUC_LIST, EPOCHS, conf, PATH)

"""
    plots the loss as a function of the epochs
    
    @parameters
        train (list)    - 
        loss (list)     - 
        epoch (int)     - 
        conf (dict)     - 
        PATH (str)      - 
"""
def plot_loss(train: list, loss: list, epoch: int, conf: dict, PATH):
    fig, ax = plt.subplots()
    ax.plot(np.arange(1, epoch+1), train, label="training")
    ax.plot(np.arange(1, epoch+1), loss, label="testing")
    plt.legend()
    ax.set(xlabel='Epoch', ylabel='loss', title='LOSS on '+conf['title'])
    newpath = os.path.join(PATH, 'images')
    plt.savefig(os.path.join(newpath, conf['name']+'_loss'))
    
    plt.close(fig)

"""
    plots the test score against the epochs for each metric
    
    @parameters
        t_1 (list)      - 
        t_2 (list)      - 
        t_3 (list)      - 
        epoch (int)     - 
        conf (dict)     - 
        PATH (str)      - 
"""
def plot_test(t_1: list, t_2: list, t_3: list, epoch: int, conf: dict, PATH):
    fig, ax = plt.subplots()
    ax.plot(np.arange(1, epoch+1), t_1, label="DICE")
    ax.plot(np.arange(1, epoch+1), t_2, label="acc")
    ax.plot(np.arange(1, epoch+1), t_3, label="auc")
    plt.legend()
    ax.set(xlabel='Epoch', ylabel='Rating', title='Rating on '+conf['title'])
    newpath = os.path.join(PATH, 'images')
    plt.savefig(os.path.join(newpath, conf['name']+'_rate'))

    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(np.arange(1, epoch+1), t_1, label="DICE")
    plt.legend()
    ax.set(xlabel='Epoch', ylabel='DICE', title='Dice on '+conf['title'])
    plt.savefig(os.path.join(newpath, conf['name']+'_dice'))

    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(np.arange(1, epoch+1), t_2, label="acc")
    plt.legend()
    ax.set(xlabel='Epoch', ylabel='Rating', title='Accuracy on '+conf['title'])
    plt.savefig(os.path.join(newpath, conf['name']+'_acc'))

    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(np.arange(1, epoch+1), t_3, label="auc")
    plt.legend()
    ax.set(xlabel='Epoch', ylabel='Rating', title='AUC on '+conf['title'])
    plt.savefig(os.path.join(newpath, conf['name']+'_auc'))
    
    plt.close(fig)