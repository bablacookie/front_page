import torch
import torch.backends.cudnn
from torchvision import transforms
import os
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from model import UNet
import model

# 标签映射
label_mapping = {
    0: 0,  # 背景 -> 类别0
    128: 1,  # 器官A -> 类别1
    255: 2  # 器官B -> 类别2
}
# 逆向映射（用于预测结果）
inverse_mapping = {v: k for k, v in label_mapping.items()}


def set_seed(seed):
    random.seed(seed)  # 设置Python的随机种子
    np.random.seed(seed)  # 设置NumPy的随机种子
    torch.manual_seed(seed)  # 设置PyTorch的CPU随机种子
    torch.cuda.manual_seed(seed)  # 设置当前GPU的随机种子（如果使用GPU）
    torch.cuda.manual_seed_all(seed)  # 设置所有GPU的随机种子（如果使用多个GPU）
    torch.backends.cudnn.deterministic = True  # 确保每次卷积操作结果一致
    torch.backends.cudnn.benchmark = False  # 禁用CUDNN的自动优化


# ... [保留之前的代码，包括 label_mapping, inverse_mapping, set_seed 函数] ...

def predict_single_image(image_path, model, device, save_dir='C:\\Users\\chao_w\\Desktop\\bs\\res'):
    try:
        # 创建保存目录
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        print(f"尝试读取图像: {image_path}")
        # 读取图像
        image = Image.open(image_path).convert("RGB")
        print("图像读取成功")

        # 图像预处理
        transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])

        input_tensor = transform(image).unsqueeze(0)  # 添加批次维度
        input_tensor = input_tensor.to(device)

        print("开始模型预测")
        # 模型预测
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.argmax(output, dim=1)[0]  # 移除批次维度
        print("模型预测完成")

        # 将预测结果映射回原始标签值
        pred_np = pred.cpu().numpy()
        pred_mapped = np.vectorize(inverse_mapping.get)(pred_np)

        # 将预测结果转换为PIL图像
        pred_img = Image.fromarray(pred_mapped.astype(np.uint8))

        # 原始图像用于对比
        orig_img = image.resize((256, 256))

        # 保存结果
        file_name = os.path.basename(image_path)
        img_save_path = os.path.join(save_dir, f"orig_{file_name}")
        pred_save_path = os.path.join(save_dir, f"pred_{file_name}")

        print("保存原始图像和预测结果")
        # 保存原始图像
        orig_img.save(img_save_path)

        # 保存预测结果
        pred_img.save(pred_save_path)

        print("生成比较图像")
        # 可视化并保存组合图像
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(orig_img)
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(pred_img, cmap='gray')
        plt.title('Prediction')
        plt.axis('off')

        plt.tight_layout()
        comparison_save_path = os.path.join(save_dir, f"comparison_{file_name.split('.')[0]}.png")
        plt.savefig(comparison_save_path)
        plt.close()

        print(f"预测完成！")
        print(f"原始图像已保存至: {img_save_path}")
        print(f"预测结果已保存至: {pred_save_path}")
        print(f"比较图像已保存至: {comparison_save_path}")

        return pred_mapped
    except Exception as e:
        print(f"发生错误: {str(e)}")
        return None


if __name__ == '__main__':
    # 设置设备（GPU/CPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用 {device} 进行预测...')

    # 设置随机种子
    seed = 3407
    set_seed(seed)
    print(f'随机种子设置为 {seed}')

    # 加载模型
    print(f'加载模型...')
    try:
        model = UNet(in_channels=3, num_classes=3)
        model = model.to(device)
        model.load_state_dict(torch.load('./models/best.pth'))
        print("模型加载成功")
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        exit(1)

    # 指定要预测的图像路径
    image_path = "C:/Users/chao_w/Desktop/bs/archive/LiTS/0/Image/45.png"
    print(f"预测图像路径: {image_path}")

    # 执行预测
    result = predict_single_image(image_path, model, device)

    if result is None:
        print("预测失败")
    else:
        print("预测成功完成")

