import argparse
import os
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader

# import torchvision
# from torchvision import datasets, models, transforms
from torch.optim import lr_scheduler
import torch.optim as optim
import cv2
from datasets import DatasetLSTM
from model import LSTM
import time
import copy
def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='nam')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('-j', '--workers', type=int, default=2)
    parser.add_argument('--replicate', type=int, default=30)
    parser.add_argument('--shuffle', action='store_true')
    
    #
    parser.add_argument('--dataset_dir', type=str, default='dataset')
    parser.add_argument('--train_dir', type=str, default='data/train')
    parser.add_argument('--save_dir', type=str, default='results/')
    
    #checkpoints, train
    parser.add_argument('--name_model', type= str, default= 'resnet50')
    parser.add_argument('--checkpoint_dir', type=str, default='XaRac/checkpoints/')
    parser.add_argument("--gpu", type=str, default='1', help="choose gpu device.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=int, default=0.005)

    opt = parser.parse_args()
    return opt
# -------------------------------------------------
def train(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device :", device)
    model = LSTM(input_size= opt.replicate * 50, hidden_size= 128, num_layers= 4, num_classes= 2, device= device)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(1, opt.epochs+1):
        print(f'Epoch {epoch}/{opt.epochs}')
        print('-' * 15)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0
            for inputs in tqdm(dataLoader[phase]):
                input = inputs['input'].float().to(device)
                # print(input.shape)
                labels = inputs['label'].to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(input)
                    # print(outputs)
                    _, preds = torch.max(outputs, 1)
                    # print(preds)
                    loss = criterion(outputs, labels)
                    # print(loss)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * input.size(0)
                
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                optimizer.step()
            # print(running_loss)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                best_epoch = epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, os.path.join(opt.checkpoint_dir, "LSTM", (str(epoch)+".pth")))
        print()
    print(f'Best val Acc: {best_acc:4f}, epoch: {best_epoch}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_epoch
    
    
#-------------------------------------------

if __name__ == '__main__':
    opt = get_opt()
    # crop_train(opt = opt)
    
    if not os.path.exists(os.path.join(opt.checkpoint_dir, "LSTM")):
        os.mkdir(os.path.join(opt.checkpoint_dir, "LSTM"))

    #----------------------dataloader---------------------
    Dataset = DatasetLSTM(opt)
    train_size = int(0.8 * len(Dataset))
    val_size = len(Dataset) - train_size
    trainDataset, valDataset = torch.utils.data.random_split(Dataset, [train_size, val_size])
    trainLoader = DataLoader(trainDataset, batch_size=opt.batch_size, shuffle= True, num_workers= opt.workers)
    valLoader = DataLoader(valDataset, batch_size=opt.batch_size, shuffle= True, num_workers= opt.workers)
    dataset_sizes = {
        'train' : len(trainDataset),
        'val' : len(valDataset),
    }
    dataLoader = {
        'train' : trainLoader,
        'val': valLoader
    }
    print(trainDataset.__getitem__(0)['input'].shape)

    model, best_epoch = train(opt)
    torch.save({
        'epoch': best_epoch,
        'model_state_dict': model.state_dict(),
    }, os.path.join(opt.checkpoint_dir, "LSTM", ("best_epoch"+".pth")))

    print("done")