#!/usr/bin/env python
# coding: utf-8

import os
import torch
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_checkpoint_history(path):
    if not os.path.exists(path):
        print(f"Checkpoint not found: {path}")
        return None
    checkpoint = torch.load(path, map_location=device)
    
    best_acc = checkpoint.get('best_acc', 0)
    if isinstance(best_acc, torch.Tensor):
        best_acc = best_acc.cpu().item()
    
    return {
        'val_acc': checkpoint.get('val_acc_history', []),
        'train_acc': checkpoint.get('train_acc_history', []),
        'val_loss': checkpoint.get('valid_losses', []),
        'train_loss': checkpoint.get('train_losses', []),
        'best_acc': best_acc,
        'model_name': path.split('/')[-1].replace('.pth', '')
    }


def merge_checkpoints(stage1_path, stage2_path):
    stage1 = load_checkpoint_history(stage1_path)
    stage2 = load_checkpoint_history(stage2_path)
    
    if stage1 is None or stage2 is None:
        return None
    
    merged = {
        'val_acc': stage1['val_acc'] + stage2['val_acc'],
        'train_acc': stage1['train_acc'] + stage2['train_acc'],
        'val_loss': stage1['val_loss'] + stage2['val_loss'],
        'train_loss': stage1['train_loss'] + stage2['train_loss'],
        'best_acc': max(stage1['best_acc'], stage2['best_acc']),
        'model_name': 'ResNet152 + CBAM (40 epochs)'
    }
    
    return merged


def find_checkpoints():
    base_dir = './'
    
    stage1_cbam = 'checkpoint_resnet152_cbam_20260620_112858.pth'
    stage2_cbam = 'checkpoint_resnet152_cbam_20260620_112858_finetuned_20260620_122723.pth'
    resnet152 = 'checkpoint_resnet152_20260619_214211.pth'
    
    checkpoints = {
        'resnet50': None,
        'resnet101': None,
        'resnet152': None,
        'resnet152_cbam': None,
    }
    
    for f in os.listdir(base_dir):
        if f == stage1_cbam:
            stage1_path = os.path.join(base_dir, f)
        elif f == stage2_cbam:
            stage2_path = os.path.join(base_dir, f)
        elif f == resnet152:
            checkpoints['resnet152'] = os.path.join(base_dir, f)
        elif 'resnet101' in f and 'cbam' not in f:
            checkpoints['resnet101'] = os.path.join(base_dir, f)
        elif 'resnet50' in f and 'cbam' not in f:
            checkpoints['resnet50'] = os.path.join(base_dir, f)
    
    if 'stage1_path' in locals() and 'stage2_path' in locals():
        checkpoints['resnet152_cbam'] = (stage1_path, stage2_path)
    
    print("\nCheckpoint详细信息:")
    for name, path in checkpoints.items():
        if path is None:
            print(f"  {name}: Not found")
        elif isinstance(path, tuple):
            print(f"  {name}: Merged ({len(path)} checkpoints), stage1={os.path.basename(path[0])}, stage2={os.path.basename(path[1])}")
        else:
            data = load_checkpoint_history(path)
            if data:
                print(f"  {name}: best_acc={data['best_acc']:.4f}, epochs={len(data['val_acc'])}")
    
    return checkpoints


def smooth_curve(data, window=3):
    if len(data) < window:
        return data
    
    data = np.array(data)
    smoothed = np.convolve(data, np.ones(window)/window, mode='valid')
    
    pad_left = (window - 1) // 2
    pad_right = window - 1 - pad_left
    
    result = list(data[:pad_left]) + list(smoothed) + list(data[-pad_right:])
    return result


def setup_cvpr_style(ax):
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_linewidth(1.2)
    ax.grid(axis='y', alpha=0.2, linestyle='-', color='#cccccc')


def plot_comparison_curves(checkpoints, save_path='figure1_comparison.png'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    
    colors = {
        'resnet50': '#3498db',
        'resnet101': '#e67e22',
        'resnet152': '#2ecc71',
        'resnet152_cbam': '#e74c3c',
    }
    
    labels = {
        'resnet50': 'ResNet50',
        'resnet101': 'ResNet101',
        'resnet152': 'ResNet152',
        'resnet152_cbam': 'ResNet152 + CBAM + Fine-tuning',
    }
    
    stage2_epoch = 20
    
    for name, path in checkpoints.items():
        if path is None:
            continue
        
        if isinstance(path, tuple):
            data = merge_checkpoints(path[0], path[1])
        else:
            data = load_checkpoint_history(path)
        
        if data is None:
            continue
            
        epochs = range(1, len(data['val_acc']) + 1)
        color = colors.get(name, '#000000')
        label = labels.get(name, name.upper())
        
        val_loss_smooth = smooth_curve(data['val_loss'])
        val_acc_smooth = smooth_curve(data['val_acc'])
        
        axes[0].plot(epochs, val_loss_smooth, label=label, linewidth=2, color=color)
        axes[1].plot(epochs, val_acc_smooth, label=label, linewidth=2, color=color)
    
    for name, path in checkpoints.items():
        if path is None:
            continue
        
        if isinstance(path, tuple):
            data = merge_checkpoints(path[0], path[1])
        else:
            data = load_checkpoint_history(path)
        
        if data is None:
            continue
        
        if len(data['val_acc']) >= stage2_epoch:
            axes[0].axvline(x=stage2_epoch, color='#95a5a6', linestyle='--', linewidth=1.5, alpha=0.7, label='Stage 2 Start')
            axes[1].axvline(x=stage2_epoch, color='#95a5a6', linestyle='--', linewidth=1.5, alpha=0.7, label='Stage 2 Start')
            break
    
    for ax in axes:
        setup_cvpr_style(ax)
    
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Validation Loss', fontsize=12)
    axes[0].set_title('(a) Validation Loss', fontsize=14)
    axes[0].legend(fontsize=10, loc='upper right')
    
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Validation Accuracy', fontsize=12)
    axes[1].set_title('(b) Validation Accuracy', fontsize=14)
    axes[1].legend(fontsize=10, loc='lower right')
    axes[1].set_ylim(0.4, 1.0)
    
    plt.tight_layout(pad=2)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Figure 1 saved: {save_path}")
    plt.close()


def plot_cbam_full_training(stage1_path, stage2_path, save_path='figure1b_cbam_full.png'):
    data = merge_checkpoints(stage1_path, stage2_path)
    
    if data is None:
        print("Failed to merge checkpoints")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    
    epochs = range(1, len(data['val_acc']) + 1)
    stage2_epoch = 20
    
    axes[0].set_facecolor('white')
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[0].spines['bottom'].set_linewidth(1.2)
    axes[0].spines['left'].set_linewidth(1.2)
    axes[0].grid(axis='y', alpha=0.2, linestyle='-', color='#cccccc')
    axes[0].plot(epochs, data['train_loss'], label='Training Loss', linewidth=2, color='#3498db')
    axes[0].plot(epochs, data['val_loss'], label='Validation Loss', linewidth=2, color='#e74c3c')
    axes[0].axvline(x=stage2_epoch, color='#95a5a6', linestyle='--', linewidth=1.5, alpha=0.7, label='Stage 2 Start')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('(a) Loss', fontsize=14)
    axes[0].legend(fontsize=10)
    
    axes[1].set_facecolor('white')
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].spines['bottom'].set_linewidth(1.2)
    axes[1].spines['left'].set_linewidth(1.2)
    axes[1].grid(axis='y', alpha=0.2, linestyle='-', color='#cccccc')
    axes[1].plot(epochs, data['train_acc'], label='Training Accuracy', linewidth=2, color='#3498db')
    axes[1].plot(epochs, data['val_acc'], label='Validation Accuracy', linewidth=2, color='#e74c3c')
    axes[1].axvline(x=stage2_epoch, color='#95a5a6', linestyle='--', linewidth=1.5, alpha=0.7, label='Stage 2 Start')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('(b) Accuracy', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].set_ylim(0.3, 1.05)
    
    plt.tight_layout(pad=2)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Figure 1b saved: {save_path}")
    plt.close()


def plot_ablation_study():
    data_dir = './'
    
    stage1_cbam = 'checkpoint_resnet152_cbam_20260620_112858.pth'
    stage2_cbam = 'checkpoint_resnet152_cbam_20260620_112858_finetuned_20260620_122723.pth'
    resnet152 = 'checkpoint_resnet152_20260619_214211.pth'
    
    results = {}
    
    if resnet152 in os.listdir(data_dir):
        data = load_checkpoint_history(os.path.join(data_dir, resnet152))
        if data:
            results['ResNet152'] = data['best_acc']
    
    if stage1_cbam in os.listdir(data_dir):
        data = load_checkpoint_history(os.path.join(data_dir, stage1_cbam))
        if data:
            results['+ CBAM'] = data['best_acc']
    
    if stage1_cbam in os.listdir(data_dir) and stage2_cbam in os.listdir(data_dir):
        data = merge_checkpoints(
            os.path.join(data_dir, stage1_cbam),
            os.path.join(data_dir, stage2_cbam)
        )
        if data:
            results['+ Fine-tuning'] = data['best_acc']
    
    if not results:
        print("No checkpoint data found for ablation study")
        return
    
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_linewidth(1.2)
    ax.grid(axis='y', alpha=0.2, linestyle='-', color='#cccccc')
    
    models = list(results.keys())
    accs = list(results.values())
    
    colors = ['#3498db', '#e67e22', '#2ecc71']
    
    bars = ax.bar(models, accs, color=colors, edgecolor='black', linewidth=1.2, width=0.4)
    
    for bar, acc in zip(bars, accs):
        height = bar.get_height()
        ax.annotate(f'{acc:.2%}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11)
    
    baseline_acc = results['ResNet152']
    for bar, acc in zip(bars[1:], accs[1:]):
        improvement = (acc - baseline_acc) * 100
        height = bar.get_height()
        ax.annotate(f'+{improvement:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 18),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, color='#e74c3c')
    
    ax.set_ylabel('Validation Accuracy', fontsize=12)
    ax.set_title('Ablation Study', fontsize=14)
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig('figure2_ablation.png', dpi=150, bbox_inches='tight')
    print(f"Figure 2 saved: figure2_ablation.png")
    plt.close()
    
    return results


if __name__ == '__main__':
    print("="*60)
    print("论文实验可视化脚本")
    print("="*60)
    
    checkpoints = find_checkpoints()
    
    print("\n[1/3] 生成综合对比图...")
    try:
        plot_comparison_curves(checkpoints, 'figure1_comparison.png')
    except Exception as e:
        print(f"Error generating figure 1: {e}")
    
    print("\n[2/3] 生成CBAM完整训练曲线...")
    stage1_path = os.path.join('./', 'checkpoint_resnet152_cbam_20260620_112858.pth')
    stage2_path = os.path.join('./', 'checkpoint_resnet152_cbam_20260620_112858_finetuned_20260620_122723.pth')
    try:
        plot_cbam_full_training(stage1_path, stage2_path, 'figure1b_cbam_full.png')
    except Exception as e:
        print(f"Error generating figure 1b: {e}")
    
    print("\n[3/3] 生成消融实验图...")
    try:
        plot_ablation_study()
    except Exception as e:
        print(f"Error generating figure 2: {e}")
    
    print("\n可视化完成！")