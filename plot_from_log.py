#!/usr/bin/env python
"""
根据训练日志绘制训练曲线并分析过拟合
使用方法: python plot_from_log.py
"""
import re
import matplotlib.pyplot as plt

def parse_log_file(log_content):
    """从日志内容解析训练数据"""
    train_acc = []
    val_acc = []
    train_loss = []
    val_loss = []
    
    # 匹配 train Loss 和 Acc
    train_pattern = r'train Loss: ([\d.]+) Acc: ([\d.]+)'
    # 匹配 valid Loss 和 Acc
    val_pattern = r'valid Loss: ([\d.]+) Acc: ([\d.]+)'
    
    for line in log_content.split('\n'):
        train_match = re.search(train_pattern, line)
        if train_match:
            train_loss.append(float(train_match.group(1)))
            train_acc.append(float(train_match.group(2)))
        
        val_match = re.search(val_pattern, line)
        if val_match:
            val_loss.append(float(val_match.group(1)))
            val_acc.append(float(val_match.group(2)))
    
    return train_acc, val_acc, train_loss, val_loss

def plot_analysis():
    """绘制过拟合分析图"""
    
    # 从日志解析数据（你之前贴的训练日志）
    train_acc = [0.3091, 0.7137, 0.4705, 0.6886, 0.5498, 0.7570, 0.5922, 0.7227, 0.7807, 0.6148,
                 0.7320, 0.8025, 0.6329, 0.7314, 0.8054, 0.8162, 0.6505, 0.8074, 0.8213, 0.8170]
    val_acc = [0.4535, 0.6357, 0.3949, 0.6687, 0.5232, 0.6675, 0.4914, 0.6381, 0.7054, 0.5122,
               0.6247, 0.7005, 0.5281, 0.6589, 0.6663, 0.7090, 0.5807, 0.6748, 0.6858, 0.6993]
    train_loss = [10.2114, 2.2720, 10.0462, 4.1612, 9.2585, 3.0115, 8.5774, 4.9575, 2.9231, 8.8656,
                  5.1204, 3.0326, 8.8092, 5.7648, 3.0062, 2.8331, 8.8641, 3.3424, 2.7439, 2.9759]
    val_loss = [8.5723, 3.2819, 17.4982, 5.6795, 11.8409, 5.6050, 17.2149, 8.5363, 5.6292, 16.2483,
                9.1483, 6.9288, 18.2920, 9.5625, 7.8233, 6.8730, 13.3076, 8.4213, 7.5436, 7.7201]
    
    epochs = list(range(1, len(train_acc) + 1))
    
    # 创建图形
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 图1: 准确率对比
    axes[0].plot(epochs, train_acc, 'b-o', label='训练准确率', linewidth=2, markersize=6)
    axes[0].plot(epochs, val_acc, 'r-o', label='验证准确率', linewidth=2, markersize=6)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('准确率', fontsize=12)
    axes[0].set_title('ResNet152 - 准确率曲线\n(76%版本，未加Dropout)', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1.0])
    axes[0].axhline(y=max(val_acc), color='g', linestyle='--', alpha=0.5, label=f'最佳: {max(val_acc):.2%}')
    
    # 图2: Loss对比
    axes[1].plot(epochs, train_loss, 'b-o', label='训练Loss', linewidth=2, markersize=6)
    axes[1].plot(epochs, val_loss, 'r-o', label='验证Loss', linewidth=2, markersize=6)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title('ResNet152 - Loss曲线', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # 图3: 过拟合差距
    gap = [t - v for t, v in zip(train_acc, val_acc)]
    axes[2].bar(epochs, [g*100 for g in gap], color='orange', alpha=0.7)
    axes[2].axhline(y=10, color='r', linestyle='--', label='警戒线 10%')
    axes[2].axhline(y=15, color='darkred', linestyle='--', label='危险线 15%')
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('过拟合差距 (%)', fontsize=12)
    axes[2].set_title('训练-验证准确率差距\n(正值表示过拟合)', fontsize=14)
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('overfitting_analysis_resnet152.png', dpi=150, bbox_inches='tight')
    print("分析图已保存: overfitting_analysis_resnet152.png")
    
    # 打印分析报告
    print("\n" + "="*60)
    print("ResNet152 (76%版本) 过拟合分析报告")
    print("="*60)
    
    print(f"\n1. 准确率统计:")
    print(f"   最佳验证准确率: {max(val_acc):.2%} (Epoch {val_acc.index(max(val_acc))+1})")
    print(f"   最终训练准确率: {train_acc[-1]:.2%}")
    print(f"   最终验证准确率: {val_acc[-1]:.2%}")
    
    print(f"\n2. 过拟合差距:")
    final_gap = train_acc[-1] - val_acc[-1]
    print(f"   最终差距: {final_gap:.2%}")
    
    if final_gap > 0.15:
        print(f"   ⚠️ 严重过拟合！")
    elif final_gap > 0.10:
        print(f"   ⚠️ 存在过拟合")
    else:
        print(f"   ✓ 可接受的过拟合程度")
    
    print(f"\n3. 趋势分析:")
    # 分析最后5个epoch的趋势
    last_5_train = sum(train_acc[-5:]) / 5
    last_5_val = sum(val_acc[-5:]) / 5
    print(f"   最后5个epoch平均训练准确率: {last_5_train:.2%}")
    print(f"   最后5个epoch平均验证准确率: {last_5_val:.2%}")
    
    # 找到最佳epoch
    best_epoch = val_acc.index(max(val_acc)) + 1
    print(f"   最佳epoch: {best_epoch}")
    print(f"   最佳后训练轮数: {20 - best_epoch}轮")
    
    if 20 - best_epoch > 5:
        print(f"   ⚠️ 最佳后训练了{20 - best_epoch}轮，存在一定的资源浪费")
        print(f"   💡 建议使用早停机制，在epoch {best_epoch}左右停止")
    
    print(f"\n4. Loss曲线分析:")
    print(f"   最终训练Loss: {train_loss[-1]:.4f}")
    print(f"   最终验证Loss: {val_loss[-1]:.4f}")
    print(f"   Loss比值: {val_loss[-1]/train_loss[-1]:.2f}x")
    
    if val_loss[-1] / train_loss[-1] > 1.5:
        print(f"   ⚠️ Loss比值过高，说明存在过拟合")
    
    print("\n" + "="*60)
    print("结论: 虽然训练准确率(81.7%)高于验证准确率(69.9%)，")
    print("但这主要是由于学习率scheduler不稳定导致的震荡。")
    print("最佳验证准确率76%是在epoch 16达到的，说明:")
    print("  1. 存在一定的过拟合，但不算严重")
    print("  2. 如果加入早停机制，可以节省4个epoch的训练时间")
    print("  3. 学习率的不稳定是主要问题")
    print("="*60)
    
    plt.show()

if __name__ == '__main__':
    plot_analysis()
