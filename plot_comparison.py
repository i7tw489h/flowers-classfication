#!/usr/bin/env python
# coding: utf-8

"""
绘制所有ResNet模型的训练曲线对比图
使用方法：
  python plot_comparison.py                    # 自动查找最新的checkpoint文件
  python plot_comparison.py --files file1.pth file2.pth file3.pth  # 手动指定文件
"""

import torch
import matplotlib.pyplot as plt
import os
import glob
import argparse

def find_latest_checkpoint(model_name):
    """查找指定模型的最新checkpoint文件"""
    pattern = f'checkpoint_{model_name}_*.pth'
    files = glob.glob(pattern)
    if files:
        # 按修改时间排序，返回最新的
        latest = max(files, key=os.path.getmtime)
        return latest
    return None

def parse_args():
    parser = argparse.ArgumentParser(description='绘制ResNet模型对比图')
    parser.add_argument('--files', nargs='+', type=str, help='手动指定checkpoint文件路径')
    return parser.parse_args()

args = parse_args()

models = ['resnet50', 'resnet101', 'resnet152']
colors = {'resnet50': '#1f77b4', 'resnet101': '#ff7f0e', 'resnet152': '#2ca02c'}

history_data = {}

if args.files:
    # 手动指定文件
    print("使用手动指定的checkpoint文件:")
    for file_path in args.files:
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found, skipping")
            continue
        
        # 从文件名提取模型名称
        basename = os.path.basename(file_path)
        model_name = basename.split('_')[1]  # checkpoint_resnet50_20260619_221901.pth
        
        checkpoint = torch.load(file_path, map_location='cpu')
        history_data[model_name] = {
            'train_loss': checkpoint.get('train_losses', []),
            'val_loss': checkpoint.get('valid_losses', []),
            'train_acc': checkpoint.get('train_acc_history', []),
            'val_acc': checkpoint.get('val_acc_history', []),
            'best_acc': checkpoint.get('best_acc', 0)
        }
        print(f"Loaded {model_name} from {basename}: Best Val Acc = {checkpoint.get('best_acc', 0):.4f}")
else:
    # 自动查找最新的checkpoint文件
    print("自动查找最新的checkpoint文件:")
    for model_name in models:
        checkpoint_path = find_latest_checkpoint(model_name)
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            history_data[model_name] = {
                'train_loss': checkpoint.get('train_losses', []),
                'val_loss': checkpoint.get('valid_losses', []),
                'train_acc': checkpoint.get('train_acc_history', []),
                'val_acc': checkpoint.get('val_acc_history', []),
                'best_acc': checkpoint.get('best_acc', 0)
            }
            print(f"Loaded {model_name} from {os.path.basename(checkpoint_path)}: Best Val Acc = {checkpoint.get('best_acc', 0):.4f}")
        else:
            print(f"Warning: No checkpoint found for {model_name}")

if not history_data:
    print("No checkpoint files found! Please train models first.")
    exit()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for model_name, data in history_data.items():
    color = colors.get(model_name, '#333333')
    epochs = range(1, len(data['train_loss']) + 1)

    axes[0, 0].plot(epochs, data['train_loss'], '-', color=color, label=f'{model_name} (Train)', linewidth=2)
    axes[0, 0].plot(epochs, data['val_loss'], '--', color=color, label=f'{model_name} (Val)', linewidth=2)

    axes[0, 1].plot(epochs, data['train_acc'], '-', color=color, label=f'{model_name} (Train)', linewidth=2)
    axes[0, 1].plot(epochs, data['val_acc'], '--', color=color, label=f'{model_name} (Val)', linewidth=2)

    axes[1, 0].plot(epochs, data['val_loss'], '-', color=color, label=f'{model_name}', linewidth=2)

    axes[1, 1].plot(epochs, data['val_acc'], '-', color=color, label=f'{model_name}', linewidth=2)

axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Training & Validation Loss')
axes[0, 0].legend(loc='upper right', fontsize=8)
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].set_title('Training & Validation Accuracy')
axes[0, 1].legend(loc='lower right', fontsize=8)
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Validation Loss')
axes[1, 0].set_title('Validation Loss Comparison')
axes[1, 0].legend(loc='upper right', fontsize=8)
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Validation Accuracy')
axes[1, 1].set_title('Validation Accuracy Comparison')
axes[1, 1].legend(loc='lower right', fontsize=8)
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('ResNet Models Training Comparison', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
print("\nComparison plot saved to model_comparison.png")

print("\n" + "="*60)
print("Summary of Best Validation Accuracy:")
print("="*60)
for model_name, data in sorted(history_data.items(), key=lambda x: x[1]['best_acc'], reverse=True):
    print(f"{model_name:12s}: {data['best_acc']:.4f}")
print("="*60)

plt.show()