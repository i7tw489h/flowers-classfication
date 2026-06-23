#!/usr/bin/env python
# coding: utf-8
"""
可解释性分析脚本
生成：CBAM注意力可视化、Grad-CAM热力图
支持多张图片批量分析
"""

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
import argparse

plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint_path):
    from train2 import ResNet152CBAM
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = ResNet152CBAM(num_classes=102, pretrained=False)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()
    
    return model


def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image)
    
    return input_tensor, image


def get_random_images(data_dir, num_images=5):
    image_dir = os.path.join(data_dir, 'valid')
    if not os.path.exists(image_dir):
        image_dir = data_dir
    
    all_images = []
    for root, dirs, files in os.walk(image_dir):
        for f in files:
            if f.endswith('.jpg'):
                all_images.append(os.path.join(root, f))
    
    np.random.seed(42)
    selected = np.random.choice(all_images, min(num_images, len(all_images)), replace=False)
    
    return selected


class GradCAMVisualizer:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.forward_handle = target_layer.register_forward_hook(self._forward_hook)
        self.backward_handle = target_layer.register_full_backward_hook(self._backward_hook)
    
    def _forward_hook(self, module, input, output):
        self.activations = output.detach()
    
    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, class_idx=None):
        self.model.eval()
        
        output = self.model(input_tensor.unsqueeze(0).to(device))
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        self.model.zero_grad()
        output[0, class_idx].backward()
        
        gradients = self.gradients.cpu().numpy()[0]
        activations = self.activations.cpu().numpy()[0]
        
        weights = np.mean(gradients, axis=(1, 2))
        
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        cam = np.maximum(cam, 0)
        cam = np.array(Image.fromarray(cam).resize((224, 224), Image.BILINEAR))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam, class_idx
    
    def remove_hooks(self):
        self.forward_handle.remove()
        self.backward_handle.remove()


def plot_multi_image_gradcam(model, image_paths, save_path='figure3_gradcam_multi.png'):
    num_images = len(image_paths)
    
    fig, axes = plt.subplots(num_images, 3, figsize=(12, 4 * num_images), dpi=150)
    
    for row, img_path in enumerate(image_paths):
        input_tensor, original_image = load_image(img_path)
        original_np = np.array(original_image.resize((224, 224)))
        
        visualizer = GradCAMVisualizer(model, model.layer4[-1])
        cam, class_idx = visualizer.generate_cam(input_tensor)
        visualizer.remove_hooks()
        
        axes[row, 0].imshow(original_np)
        axes[row, 0].set_title(f'Image {row+1}\n(Class: {class_idx})', fontsize=11)
        axes[row, 0].axis('off')
        
        axes[row, 1].imshow(original_np)
        axes[row, 1].imshow(cam, cmap='jet', alpha=0.5)
        axes[row, 1].set_title('Grad-CAM Overlay', fontsize=11)
        axes[row, 1].axis('off')
        
        axes[row, 2].imshow(cam, cmap='jet')
        axes[row, 2].set_title('Grad-CAM Heatmap', fontsize=11)
        axes[row, 2].axis('off')
    
    plt.tight_layout(pad=2)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Figure 3 saved: {save_path}")
    plt.close()


def plot_multi_layer_gradcam(model, image_paths, save_path='figure4_multilayer_gradcam.png'):
    num_images = len(image_paths)
    
    layers = {
        'layer1': model.layer1[-1],
        'layer2': model.layer2[-1],
        'layer3': model.layer3[-1],
        'layer4': model.layer4[-1],
    }
    
    fig, axes = plt.subplots(num_images, len(layers) + 1, figsize=(4 * (len(layers) + 1), 4 * num_images), dpi=150)
    
    for row, img_path in enumerate(image_paths):
        input_tensor, original_image = load_image(img_path)
        original_np = np.array(original_image.resize((224, 224)))
        
        axes[row, 0].imshow(original_np)
        axes[row, 0].set_title(f'Image {row+1}', fontsize=11)
        axes[row, 0].axis('off')
        
        for col, (name, layer) in enumerate(layers.items()):
            visualizer = GradCAMVisualizer(model, layer)
            cam, _ = visualizer.generate_cam(input_tensor)
            visualizer.remove_hooks()
            
            axes[row, col + 1].imshow(original_np)
            axes[row, col + 1].imshow(cam, cmap='jet', alpha=0.5)
            axes[row, col + 1].set_title(f'{name}', fontsize=11)
            axes[row, col + 1].axis('off')
    
    plt.tight_layout(pad=2)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Figure 4 saved: {save_path}")
    plt.close()


def plot_attention_comparison(model, image_paths, save_path='figure5_attention_comparison.png'):
    num_images = len(image_paths)
    
    fig, axes = plt.subplots(num_images, 2, figsize=(10, 5 * num_images), dpi=150)
    
    for row, img_path in enumerate(image_paths):
        input_tensor, original_image = load_image(img_path)
        original_np = np.array(original_image.resize((224, 224)))
        
        visualizer_l4 = GradCAMVisualizer(model, model.layer4[-1])
        cam_l4, _ = visualizer_l4.generate_cam(input_tensor)
        visualizer_l4.remove_hooks()
        
        visualizer_cbam = GradCAMVisualizer(model, model.cbam4)
        cam_cbam, _ = visualizer_cbam.generate_cam(input_tensor)
        visualizer_cbam.remove_hooks()
        
        axes[row, 0].imshow(original_np)
        axes[row, 0].imshow(cam_l4, cmap='jet', alpha=0.5)
        axes[row, 0].set_title(f'Image {row+1}\nLayer4 Attention', fontsize=11)
        axes[row, 0].axis('off')
        
        axes[row, 1].imshow(original_np)
        axes[row, 1].imshow(cam_cbam, cmap='jet', alpha=0.5)
        axes[row, 1].set_title(f'CBAM4 Attention', fontsize=11)
        axes[row, 1].axis('off')
    
    plt.tight_layout(pad=2)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Figure 5 saved: {save_path}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='可解释性分析')
    parser.add_argument('--checkpoint', type=str, required=True, 
                        help='Checkpoint文件路径 (ResNet152+CBAM)')
    parser.add_argument('--image_dir', type=str, default='./flowers-102',
                        help='图片数据集目录')
    parser.add_argument('--num_images', type=int, default=5,
                        help='要分析的图片数量')
    parser.add_argument('--output_dir', type=str, default='./',
                        help='输出目录')
    args = parser.parse_args()
    
    print("="*60)
    print("可解释性分析")
    print("="*60)
    print(f"模型权重: {args.checkpoint}")
    print(f"图片数量: {args.num_images}")
    print("="*60)
    
    model = load_model(args.checkpoint)
    
    image_paths = get_random_images(args.image_dir, args.num_images)
    print(f"\n选中的图片:")
    for i, p in enumerate(image_paths):
        print(f"  {i+1}. {os.path.basename(p)}")
    
    print("\n[1/3] 生成多图片Grad-CAM对比...")
    plot_multi_image_gradcam(
        model, 
        image_paths,
        save_path=os.path.join(args.output_dir, 'figure3_gradcam_multi.png')
    )
    
    print("\n[2/3] 生成多层注意力对比...")
    plot_multi_layer_gradcam(
        model,
        image_paths,
        save_path=os.path.join(args.output_dir, 'figure4_multilayer_gradcam.png')
    )
    
    print("\n[3/3] 生成CBAM vs Layer4注意力对比...")
    plot_attention_comparison(
        model,
        image_paths,
        save_path=os.path.join(args.output_dir, 'figure5_attention_comparison.png')
    )
    
    print("\n可解释性分析完成！")
    print("生成的图片:")
    print("  - figure3_gradcam_multi.png (多张图片的Grad-CAM)")
    print("  - figure4_multilayer_gradcam.png (多层注意力对比)")
    print("  - figure5_attention_comparison.png (CBAM vs Layer4)")