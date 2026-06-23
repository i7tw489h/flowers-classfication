#!/usr/bin/env python
"""
绘制训练曲线，分析过拟合问题
支持绘制单个模型的训练历史
"""
import torch
import matplotlib.pyplot as plt
import argparse

def plot_checkpoint(checkpoint_path, model_name="ResNet152"):
    """从checkpoint加载训练历史并绘制曲线"""
    
    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 获取训练历史
    train_acc_history = checkpoint.get('train_acc_history', [])
    val_acc_history = checkpoint.get('val_acc_history', [])
    train_loss_history = checkpoint.get('train_loss_history', [])
    val_loss_history = checkpoint.get('val_loss_history', [])
    val_acc_list = checkpoint.get('val_acc_list', [])
    train_acc_list = checkpoint.get('train_acc_list', [])
    train_loss_list = checkpoint.get('train_loss_list', [])
    val_loss_list = checkpoint.get('val_loss_list', [])
    LRs = checkpoint.get('LRs', [])
    
    # 合并数据（有些key名字不同）
    if train_acc_list and not train_acc_history:
        train_acc_history = train_acc_list
    if val_acc_list and not val_acc_history:
        val_acc_history = val_acc_list
    if train_loss_list and not train_loss_history:
        train_loss_history = train_loss_list
    if val_loss_list and not val_loss_history:
        val_loss_history = val_loss_list
    
    print(f"训练历史数据点数: {len(train_acc_history)}")
    print(f"最佳验证准确率: {checkpoint.get('best_acc', 'N/A'):.4f}" if isinstance(checkpoint.get('best_acc'), float) else f"最佳验证准确率: {checkpoint.get('best_acc')}")
    
    # 创建图形
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    epochs = range(1, len(train_acc_history) + 1)
    
    # 图1: 准确率对比
    axes[0].plot(epochs, train_acc_history, 'b-', label='训练准确率', linewidth=2)
    axes[0].plot(epochs, val_acc_history, 'r-', label='验证准确率', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('准确率', fontsize=12)
    axes[0].set_title(f'{model_name} - 准确率曲线', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1.05])
    
    # 图2: Loss对比
    axes[1].plot(epochs, train_loss_history, 'b-', label='训练Loss', linewidth=2)
    axes[1].plot(epochs, val_loss_history, 'r-', label='验证Loss', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title(f'{model_name} - Loss曲线', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # 图3: 学习率变化
    if LRs:
        axes[2].plot(epochs, LRs, 'g-', linewidth=2)
        axes[2].set_xlabel('Epoch', fontsize=12)
        axes[2].set_ylabel('学习率', fontsize=12)
        axes[2].set_title(f'{model_name} - 学习率变化', fontsize=14)
        axes[2].set_yscale('log')
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图形
    output_path = f'training_curves_{model_name.lower()}_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"训练曲线已保存到: {output_path}")
    
    # 分析过拟合
    print("\n" + "="*50)
    print("过拟合分析")
    print("="*50)
    
    if train_acc_history and val_acc_history:
        final_train_acc = train_acc_history[-1]
        final_val_acc = val_acc_history[-1]
        gap = final_train_acc - final_val_acc
        
        print(f"最终训练准确率: {final_train_acc:.4f} ({final_train_acc*100:.2f}%)")
        print(f"最终验证准确率: {final_val_acc:.4f} ({final_val_acc*100:.2f}%)")
        print(f"过拟合差距: {gap:.4f} ({gap*100:.2f}%)")
        
        if gap > 0.15:
            print("\n⚠️ 严重过拟合！差距超过15%")
        elif gap > 0.10:
            print("\n⚠️ 存在过拟合，差距在10-15%之间")
        elif gap > 0.05:
            print("\n✓ 轻微过拟合，可接受")
        else:
            print("\n✓ 无明显过拟合")
        
        # 分析趋势
        if len(val_acc_history) >= 5:
            last_5_trend = val_acc_history[-1] - val_acc_history[-5]
            if last_5_trend < -0.05:
                print("\n⚠️ 验证准确率呈下降趋势，明显过拟合")
            elif last_5_trend > 0.02:
                print("\n✓ 验证准确率仍在上升，可以继续训练")
            else:
                print("\n✓ 验证准确率趋于稳定")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='绘制训练曲线分析过拟合')
    parser.add_argument('--checkpoint', '-c', type=str, 
                       default='checkpoint_resnet152.pth',
                       help='checkpoint文件路径')
    parser.add_argument('--name', '-n', type=str, 
                       default='ResNet152',
                       help='模型名称')
    
    args = parser.parse_args()
    
    plot_checkpoint(args.checkpoint, args.name)

if __name__ == '__main__':
    main()
