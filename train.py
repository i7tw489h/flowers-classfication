#!/usr/bin/env python
# coding: utf-8

# # 基于经典网络架构训练图像分类模型

# ### 数据预处理部分：
# 
# - 数据增强：torchvision中transforms模块自带功能，比较实用
# - 数据预处理：torchvision中transforms也帮我们实现好了，直接调用即可
# - DataLoader模块直接读取batch数据
# 
# ### 网络模块设置：
# 
# - 加载预训练模型，torchvision中有很多经典网络架构，调用起来十分方便，并且可以用人家训练好的权重参数来继续训练，也就是所谓的迁移学习(也叫做微调)
# - 需要注意的是别人训练好的任务跟咱们的可不是完全一样，需要把最后的head层改一改，一般也就是最后的全连接层，改成咱们自己的任务,比如分类数，resnet是1000分类，我们的任务是102分类，最后需要重新设置一个fc层
# - 训练时可以全部重头训练，也可以只训练最后咱们任务的层，因为前几层都是做特征提取的，本质任务目标是一致的
# - 本次训练，先冻住特征提取层，只训练输出层，待fc层训练完成之后，再解冻特征提取层，进行参数的全部更新
# 
# ### 网络模型保存与测试
# - 模型保存的时候可以带有选择性，例如在验证集中如果当前效果好则保存
# - 读取模型进行实际测试
# 
# ![title](1.png)



import os
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torchvision
#pip install torchvision
from torchvision import transforms, models, datasets
#https://pytorch.org/docs/stable/torchvision/index.html
import time
import warnings
import random
import sys
import copy
import json
from PIL import Image


# ### 数据读取与预处理操作



data_dir = './flower_data/'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'


# ### 制作好数据源（数据增强操作）：
# - data_transforms中指定了所有图像预处理操作
# - ImageFolder假设所有的文件按文件夹保存好，每个文件夹下面存贮同一类别的图片，文件夹的名字为分类的名字



data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(45),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.RandomGrayscale(p=0.025),
        # transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # 注释掉：防止过拟合
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


# 构建dataloder（文件夹的方式，文件夹的名字对应标签）,让模型加载数据
# 



batch_size = 8

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) for x in ['train', 'valid']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
class_names = image_datasets['train'].classes





# ### 读取标签对应的实际名字



with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f) #利用json文件将标签转化为对应的名字，便于展示，增强可读性




# ### 加载models中提供的模型，并且直接用训练的好权重当做初始化参数（直接拿训练好的模型来用，除了输出层外全部冻结不训练）
# - 第一次执行需要下载，可能会比较慢，我会提供给大家一份下载好的，可以直接放到相应路径



model_name = 'resnet'  #可选的比较多 ['resnet', 'alexnet', 'vgg', 'squeezenet', 'densenet', 'inception']
#是否用人家训练好的特征来做
feature_extract = True #表示用别人的参数，这里全部冻结了，不计算梯度了，接下来需要进行参数的冻结准备




# 是否用GPU训练
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
            #冻结操作，就是不计算梯度了，这样就不会更新这些参数了，节省计算资源和时间了。




model_ft = models.resnet152()
#model_ft #打印模型看看结构
#卷积之后一般要更一个Bn层，稳定数据分布是正态分布，有助于模型的训练和收敛。这是因为卷积之后数据的分布肯定变了
#可以看到最后直接是一层池化+fc，并没有高维特征图打平成向量，这是因为最大池化本身就将特征图打平了，所以不需要再打平了。


# ### 参考pytorch官网例子



def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    model_ft = None
    input_size = 0

    if model_name == "resnet50":
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_ftrs, num_classes),
            nn.LogSoftmax(dim=1)
        )
        input_size = 224

    elif model_name == "resnet101":
        model_ft = models.resnet101(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_ftrs, num_classes),
            nn.LogSoftmax(dim=1)
        )
        input_size = 224

    elif model_name == "resnet152":
        model_ft = models.resnet152(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_ftrs, num_classes),
            nn.LogSoftmax(dim=1)
        )
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg16(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


# ### 训练模块



def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25, is_inception=False, filename='checkpoint.pth', patience=5):
    since = time.time()
    best_acc = 0
    best_model_wts = copy.deepcopy(model.state_dict())

    model.to(device)

    val_acc_history = []
    train_acc_history = []
    train_losses = []
    valid_losses = []
    LRs = [optimizer.param_groups[0]['lr']]

    epochs_no_improve = 0
    early_stop = False

    for epoch in range(num_epochs):
        if early_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    if is_inception and phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            time_elapsed = time.time() - since
            print('Time elapsed {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'valid':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    # epochs_no_improve = 0  # 注释掉：早停机制
                    state = {
                        'state_dict': model.state_dict(),
                        'best_acc': best_acc,
                        'optimizer': optimizer.state_dict(),
                        'train_acc_history': train_acc_history,
                        'val_acc_history': val_acc_history,
                        'train_losses': train_losses,
                        'valid_losses': valid_losses,
                        'LRs': LRs,
                    }
                    torch.save(state, filename)
                    print(f"  -> New best model saved! Best val Acc: {best_acc:.4f}")
                else:
                    pass
                    # epochs_no_improve += 1  # 注释掉：早停机制

                val_acc_history.append(float(epoch_acc.cpu().numpy()))
                valid_losses.append(float(epoch_loss))
                scheduler.step()

            if phase == 'train':
                train_acc_history.append(float(epoch_acc.cpu().numpy()))
                train_losses.append(float(epoch_loss))

        print('Optimizer learning rate : {:.7f}'.format(optimizer.param_groups[0]['lr']))
        LRs.append(optimizer.param_groups[0]['lr'])
        print()

        # if epochs_no_improve >= patience:  # 注释掉：早停机制
        #     early_stop = True

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)

    final_state = {
        'state_dict': model.state_dict(),
        'best_acc': best_acc,
        'optimizer': optimizer.state_dict(),
        'train_acc_history': train_acc_history,
        'val_acc_history': val_acc_history,
        'train_losses': train_losses,
        'valid_losses': valid_losses,
        'LRs': LRs,
    }
    torch.save(final_state, filename)

    return model, val_acc_history, train_acc_history, valid_losses, train_losses, LRs 


# 关于模型训练的函数的思路流程总结：
# 分为2大块，第一是准备部分，第二是核心训练部分。
# 对于核心训练部分：1.整体的思路是：共训练x轮，每一轮的训练分为训练阶段和验证阶段，而每个阶段都要对一轮里面的所有批次的数据来进行相应的操作。所以有三层循环，对于最外层，也就是epoch层，来看，这次的操作是：打印当前的轮次/总轮次，分割线，开启训练和验证，打印学习率和换行 eg.Epoch 0/19
# ----------  xxxxx  Optimizer learning rate : 0.0100000   ；对于第二层循环，也就是训练和验证循环，来看操作是：如果是训练，开启训练模式，初始化loss和acc，开始批次取数据，计算这一轮次所对应的训练阶段的loss和acc,打印这一轮次对应的训练的对应的时间和结果，保存历史的loss和acc eg.Time elapsed 1m 32s
# train Loss: 5.5265 Acc: 0.4208  .如果是验证，开始验证模式，初始化loss和acc，开始批次取数据，计算这一轮次所对应的验证阶段的loss和acc，打印这一轮次对应的验证阶段的时间和结果，保存历史的loss和acc，以及更新acc和最好的模型，还有调整学习率 eg.Time elapsed 1m 40s
# valid Loss: 5.2129 Acc: 0.3704    ；对于第三层循环，也就按批次取数据的循环，来看操作是：一批一批地取对应数据集的数据，把取到的数据和对应的标签放到对应的设备上，开始前向传播，计算loss，如果是训练阶段，还需要梯度下降法更新参数[清0梯度，计算梯度，梯度下降更新参数]，最后计算这一批次的的总的loss和acc,acc的计算是看预测对了多少个，所以先判断模型的预测结果 _, preds = torch.max(outputs, 1);当所有的轮次都训练完了，那就记录一下训练阶段总共花了多少时间，最好的acc，最终的模型里面的参数不是最后一次训练的参数情况，而是最好的那一次的参数情况，所以加载最好的那一次模型参数，将其作为本次模型的最终结果，返回model，历史的val_acc,val_loss,train_acc,train_loss，学习率历史。
# 对于准备部分：依据核心部分来说，准备部分其实只要初始化最大acc，将模型放到对应的设备上，返回5个结果的初始化，以及保存最好的模型参数
# 







# ### 开始训练！

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ResNet model on flower dataset')
    parser.add_argument('--model', type=str, default='resnet152', 
                        choices=['resnet50', 'resnet101', 'resnet152'],
                        help='Model type: resnet50, resnet101, resnet152')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    args = parser.parse_args()

    model_name = args.model
    num_epochs = args.epochs
    learning_rate = args.lr
    patience = args.patience

    print(f"Training {model_name} with {num_epochs} epochs, lr={learning_rate}, patience={patience}")

    num_classes = 102
    feature_extract = True

    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

    model_ft = model_ft.to(device)

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    filename = f'checkpoint_{model_name}_{timestamp}.pth'

    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t",name)

    optimizer_ft = optim.Adam(params_to_update, lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    criterion = nn.NLLLoss()

    model_ft, val_acc_history, train_acc_history, valid_losses, train_losses, LRs = train_model(
        model_ft, dataloaders, criterion, optimizer_ft, scheduler,
        num_epochs=num_epochs, is_inception=False, filename=filename, patience=patience
    )

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(train_losses) + 1)

    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, valid_losses, 'r-', label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{model_name} - Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, train_acc_history, 'b-', label='Training Accuracy')
    ax2.plot(epochs, val_acc_history, 'r-', label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'{model_name} - Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(f'training_curves_{model_name}_{timestamp}.png', dpi=150)
    print(f"Training curves saved to training_curves_{model_name}_{timestamp}.png")

    plt.show()

    print("训练完成！")