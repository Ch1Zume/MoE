import matplotlib.pyplot as plt
import numpy as np

def plot_images(images, labels, class_names):
    """绘制 9 张图像"""
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].permute(1, 2, 0).cpu().numpy())  # 适用于 PyTorch 数据
        ax.set_title(class_names[labels[i]])
        ax.axis("off")
    plt.show()
