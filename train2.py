#!/usr/bin/env python
# coding: utf-8

import os
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torchvision
from torchvision import transforms, models, datasets
import time
import warnings
import random
import sys
import copy
import json
from PIL import Image


# ==================== CBAM Module ====================
class ChannelAttention(nn.Module):
    """通道注意力模块 - 关注'什么'特征重要"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        return self.sigmoid(avg_out + max_out).unsqueeze(2).unsqueeze(3)


class SpatialAttention(nn.Module):
    """空间注意力模块 - 关注'哪里'的位置重要"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(out))


class CBAM(nn.Module):
    """CBAM: Convolutional Block Attention Module
    依次应用通道注意力和空间注意力
    """
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(spatial_kernel)
    
    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


# ==================== ResNet with CBAM ====================
class ResNet152CBAM(nn.Module):
    """带CBAM的ResNet152"""
    def __init__(self, num_classes=102, pretrained=True):
        super().__init__()
        # 加载预训练ResNet152
        resnet = models.resnet152(pretrained=pretrained)
        
        # 复制原始结构
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        # Layer1-4: 每个layer后添加CBAM
        self.layer1 = resnet.layer1  # 256 channels
        self.cbam1 = CBAM(channels=256, reduction=16)
        
        self.layer2 = resnet.layer2  # 512 channels
        self.cbam2 = CBAM(channels=512, reduction=16)
        
        self.layer3 = resnet.layer3  # 1024 channels
        self.cbam3 = CBAM(channels=1024, reduction=16)
        
        self.layer4 = resnet.layer4  # 2048 channels
        self.cbam4 = CBAM(channels=2048, reduction=16)
        
        self.avgpool = resnet.avgpool
        self.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(2048, num_classes),
            nn.LogSoftmax(dim=1)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.cbam1(x)
        
        x = self.layer2(x)
        x = self.cbam2(x)
        
        x = self.layer3(x)
        x = self.cbam3(x)
        
        x = self.layer4(x)
        x = self.cbam4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


data_dir = './flower_data/'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(45),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.RandomGrayscale(p=0.025),
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


batch_size = 8

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) for x in ['train', 'valid']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
class_names = image_datasets['train'].classes


with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


model_name = 'resnet'
feature_extract = True


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

    elif model_name == "resnet152_cbam":
        # 使用自定义的带CBAM的ResNet152
        model_ft = ResNet152CBAM(num_classes=num_classes, pretrained=use_pretrained)
        if feature_extract:
            # 冻结骨干网络，只训练FC层和CBAM
            set_parameter_requires_grad(model_ft.conv1, feature_extract)
            set_parameter_requires_grad(model_ft.bn1, feature_extract)
            set_parameter_requires_grad(model_ft.layer1, feature_extract)
            set_parameter_requires_grad(model_ft.layer2, feature_extract)
            set_parameter_requires_grad(model_ft.layer3, feature_extract)
            set_parameter_requires_grad(model_ft.layer4, feature_extract)
            # CBAM和FC层默认是可训练的
        input_size = 224

    elif model_name == "alexnet":
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        model_ft = models.vgg16(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


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

                val_acc_history.append(float(epoch_acc.cpu().numpy()))
                valid_losses.append(float(epoch_loss))
                scheduler.step()

            if phase == 'train':
                train_acc_history.append(float(epoch_acc.cpu().numpy()))
                train_losses.append(float(epoch_loss))

        print('Optimizer learning rate : {:.7f}'.format(optimizer.param_groups[0]['lr']))
        LRs.append(optimizer.param_groups[0]['lr'])
        print()

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ResNet model on flower dataset')
    parser.add_argument('--model', type=str, default='resnet152', 
                        choices=['resnet50', 'resnet101', 'resnet152', 'resnet152_cbam'],
                        help='Model type: resnet50, resnet101, resnet152, resnet152_cbam')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs for stage 1')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for stage 1')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--unfreeze', action='store_true', help='Enable two-stage training with unfreezing')
    parser.add_argument('--epochs_stage2', type=int, default=5, help='Number of epochs for stage 2')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint file to resume from (skip stage 1)')
    args = parser.parse_args()

    model_name = args.model
    num_epochs = args.epochs
    learning_rate = args.lr
    patience = args.patience
    use_unfreeze = args.unfreeze
    epochs_stage2 = args.epochs_stage2
    resume_path = args.resume

    num_classes = 102

    val_acc_history = []
    train_acc_history = []
    valid_losses = []
    train_losses = []
    LRs = []

    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract=True, use_pretrained=True)
    model_ft = model_ft.to(device)

    timestamp = time.strftime('%Y%m%d_%H%M%S')

    if resume_path:
        print(f"Resuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model_ft.load_state_dict(checkpoint['state_dict'])
        
        val_acc_history = checkpoint.get('val_acc_history', [])
        train_acc_history = checkpoint.get('train_acc_history', [])
        valid_losses = checkpoint.get('valid_losses', [])
        train_losses = checkpoint.get('train_losses', [])
        LRs = checkpoint.get('LRs', [])
        
        print(f"Loaded previous training history: {len(train_acc_history)} epochs")
        print(f"Best val Acc in checkpoint: {checkpoint.get('best_acc', 0):.4f}")
        
        num_epochs = len(train_acc_history)
        use_unfreeze = True

        criterion = nn.NLLLoss()

    else:
        print(f"Training {model_name} with {num_epochs} epochs, lr={learning_rate}, patience={patience}")
        if use_unfreeze:
            print(f"Two-stage training enabled: {num_epochs} epochs stage 1, {epochs_stage2} epochs stage 2")

        filename = f'checkpoint_{model_name}_{timestamp}.pth'

        # 收集需要训练的参数（requires_grad=True的）
        params_to_update = [p for p in model_ft.parameters() if p.requires_grad]
        print("Params to learn (Stage 1 - Feature Extraction):")
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t", name)

        optimizer_ft = optim.Adam(params_to_update, lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
        criterion = nn.NLLLoss()

        model_ft, val_acc_history, train_acc_history, valid_losses, train_losses, LRs = train_model(
            model_ft, dataloaders, criterion, optimizer_ft, scheduler,
            num_epochs=num_epochs, is_inception=False, filename=filename, patience=patience
        )

    if use_unfreeze:
        print("\n" + "="*60)
        print("Stage 2: Fine-tuning with unfrozen layers")
        print("="*60)
        print(f"Stage 2 epochs: {epochs_stage2}")

        for param in model_ft.parameters():
            param.requires_grad = True

        # 基础参数组
        params_group = [
            {'params': model_ft.conv1.parameters(), 'lr': 1e-5},
            {'params': model_ft.bn1.parameters(), 'lr': 1e-5},
            {'params': model_ft.layer1.parameters(), 'lr': 1e-5},
            {'params': model_ft.layer2.parameters(), 'lr': 3e-5},
            {'params': model_ft.layer3.parameters(), 'lr': 5e-5},
            {'params': model_ft.layer4.parameters(), 'lr': 1e-4},
            {'params': model_ft.fc.parameters(), 'lr': 5e-4},
        ]

        # 如果是CBAM模型，添加CBAM模块的学习率
        if model_name == 'resnet152_cbam':
            params_group.extend([
                {'params': model_ft.cbam1.parameters(), 'lr': 1e-5},
                {'params': model_ft.cbam2.parameters(), 'lr': 3e-5},
                {'params': model_ft.cbam3.parameters(), 'lr': 5e-5},
                {'params': model_ft.cbam4.parameters(), 'lr': 1e-4},
            ])

        print("Params to learn (Stage 2 - Fine-tuning):")
        print("\tconv1, bn1, layer1: lr=1e-5")
        print("\tlayer2: lr=3e-5")
        print("\tlayer3: lr=5e-5")
        print("\tlayer4: lr=1e-4")
        print("\tfc: lr=5e-4")
        if model_name == 'resnet152_cbam':
            print("\tcbam1: lr=1e-5, cbam2: lr=3e-5, cbam3: lr=5e-5, cbam4: lr=1e-4")

        optimizer_stage2 = optim.Adam(params_group)
        scheduler_stage2 = optim.lr_scheduler.StepLR(optimizer_stage2, step_size=3, gamma=0.5)

        if resume_path:
            basename = os.path.basename(resume_path).replace('.pth', '')
            filename_stage2 = f'{basename}_finetuned_{timestamp}.pth'
        else:
            filename_stage2 = f'checkpoint_{model_name}_{timestamp}_finetuned.pth'

        model_ft, val_acc_history_st2, train_acc_history_st2, valid_losses_st2, train_losses_st2, LRs_st2 = train_model(
            model_ft, dataloaders, criterion, optimizer_stage2, scheduler_stage2,
            num_epochs=epochs_stage2, is_inception=False, filename=filename_stage2, patience=patience
        )

        val_acc_history.extend(val_acc_history_st2)
        train_acc_history.extend(train_acc_history_st2)
        valid_losses.extend(valid_losses_st2)
        train_losses.extend(train_losses_st2)
        LRs.extend(LRs_st2)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(train_losses) + 1)

    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, valid_losses, 'r-', label='Validation Loss')
    if use_unfreeze:
        ax1.axvline(num_epochs, color='g', linestyle='--', label='Stage 2 Start')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{model_name} - Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, train_acc_history, 'b-', label='Training Accuracy')
    ax2.plot(epochs, val_acc_history, 'r-', label='Validation Accuracy')
    if use_unfreeze:
        ax2.axvline(num_epochs, color='g', linestyle='--', label='Stage 2 Start')
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