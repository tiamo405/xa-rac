import argparse
import os
import numpy as np
import torch
import torch.optim as optim
import cv2
import time
import copy
import datetime

from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from src import write_txt
from torch.optim import lr_scheduler
from preprocessing.datasets import DatasetLSTM
from model import LSTM
def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='nam')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('-j', '--workers', type=int, default=2)
    parser.add_argument('--replicate', type=int, default=30)
    parser.add_argument('--shuffle', action='store_true')
    
    #path dir
    parser.add_argument('--dataset_dir', type=str, default='dataset')
    parser.add_argument('--train_dir', type=str, default='data/train')
    parser.add_argument('--save_dir', type=str, default='results/')
    
    #checkpoints, train
    parser.add_argument('--name_model', type= str, default= 'LSTM')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/')
    parser.add_argument("--gpu", type=str, default='1', help="choose gpu device.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=int, default=0.005)
    parser.add_argument('--num_save_ckp', type= int, default= 5)
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
                print(input.shape)
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

            if phase == 'val' and epoch_acc >= best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                best_epoch = epoch
        if epoch % opt.num_save_ckp == 0 or epoch == opt.epochs :
            path_save_ckp = os.path.join(opt.checkpoint_dir, opt.name_model, \
                          str(len(os.listdir(os.path.join(opt.checkpoint_dir, opt.name_model))) -1),\
                            str(epoch)+'.pth')
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            },path_save_ckp)
        print()
    print(f'Best val Acc: {best_acc:4f}, epoch: {best_epoch}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_epoch
    
    
#-------------------------------------------

if __name__ == '__main__':
    opt = get_opt()
    # crop_train(opt = opt)
    print(opt)

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

    if not os.path.exists(os.path.join(opt.checkpoint_dir, opt.name_model)) :
        os.mkdir(os.path.join(opt.checkpoint_dir, opt.name_model))

    os.mkdir(os.path.join(opt.checkpoint_dir, opt.name_model, \
                          str(len(os.listdir(os.path.join(opt.checkpoint_dir, opt.name_model))))))

    write_txt(opt, os.path.join(opt.checkpoint_dir, opt.name_model,\
            str(len(os.listdir(os.path.join(opt.checkpoint_dir, opt.name_model))) -1), 'opt.txt'))
    start_time = time.time()
    model, best_epoch = train(opt)
    torch.save({
        'epoch': best_epoch,
        'model_state_dict': model.state_dict(),
    }, os.path.join(opt.checkpoint_dir, opt.name_model, \
                          str(len(os.listdir(os.path.join(opt.checkpoint_dir, opt.name_model))) -1),\
                            'best_epoch.pth'))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))