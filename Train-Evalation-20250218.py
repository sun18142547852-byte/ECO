# -*- coding: utf-8 -*-
"""
@Project : PY_Project
@Time    : 2025/2/18 16:03
@Author  : Yunhong Xie
@File    : Train-Evalation-20250218.py
@Software: PyCharm
@Adverbial : Chasing light forever
"""
import shutil
import os
from pathlib import Path
import cv2
import yaml
import pprint
import json
import numpy as np
import datetime
import time
import torch
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from tensorboardX import SummaryWriter
from models import maskrcnn
from utils.visualizer import draw_prediction2
from loader import SyntheticDataset, WISDOMDataset
from utils.engine import train_one_epoch, collate_fn, evaluate
from utils import visualizer
from utils.coco_utils import get_coco_api_from_dataset, coco_to_excel
import smtplib
from email.mime.text import MIMEText
from email.header import Header

# 记录开始时间
start_time = time.time()


# 配置文件和训练参数设置
def set_training_parameters():
    gpu = "0"
    # gpu = "1"
    """
    depth_only
    rgb_only
    rgb_depth_earlyfusion
    rgb_depth_latefusion
    rgb_depth_confidencefusion
    """
    cfg_name = 'rgb_depth_confidencefusion'  # 配置文件名 优先确保骨干网络结构
    resume_training = False
    resume_training_weight = None  # None 代表不恢复训练
    save_interval = 1  # 保存间隔
    return gpu, cfg_name, resume_training, resume_training_weight, save_interval


# 专门用于推断
def set_training_parameters2(cfg_name):
    gpu = "0"
    # gpu = "1"
    cfg_name = cfg_name  # 配置文件名 优先确保骨干网络结构
    resume_training = False
    resume_training_weight = None  # None 代表不恢复训练
    save_interval = 1  # 保存间隔
    return gpu, cfg_name, resume_training, resume_training_weight, save_interval


# 加载配置文件
def load_config(cfg_name, process="train"):
    with open(f'cfgs/{cfg_name}.yaml') as cfg_file:
        cfg = yaml.safe_load(cfg_file)
    """
    resnet50
    resnet101
    resnext50_32x4d
    resnext101_32x8d
    """
    cfg['backbone_name'] = 'resnext50_32x4d'  # 网络结构名称'resnext50_32x4d'
    # "confidence_map_estimator" "val_mask_as_confidence_map" "self_attention_as_confidence_map"
    cfg['fusion_method'] =  "confidence_map_estimator"
    cfg['batch_size'] = 8  # 设置批次大小
    cfg['best_epoch'] = 0
    # a_renzhensi_rgbh b_renzhensi_rgbe c_renzhensi_rgbn d_renzhensi_rgbneh e_renzhensi_rsi f_renzhensi_rgbdsm
    # depth_noisy depth_value rgb seg
    if process=="train":
        cfg['dataset_path'] = r"F:\2024\20240305RGBDpaper\MTH-Mask_R-CNN\examples\1_LiDAR_samples\train_aug_1012"
    if process=="test":
        # cfg['dataset_path'] = r"F:\2024\20240305RGBDpaper\MTH-Mask_R-CNN\examples\1_LiDAR_samples\test"  # 用于实例合并的测试集路径
        cfg['dataset_path'] = r"M:\2025\20250725AGB_model\xizang_data\database\rezhensi-c-val"
    cfg['height'] = 512
    cfg['width'] = 512
    cfg['lr'] = 0.0001 # 0.0001
    cfg['wd'] = 0.0001  # 权重衰减参数
    cfg['max_epoch'] = 25 + 1
    return cfg

# 初始化数据集
def load_train_dataset(cfg):
    dataset_path = cfg["dataset_path"]
    if cfg["dataset"] == 'synthetic':
        dataset = SyntheticDataset(dataset_path=dataset_path, mode="train", cfg=cfg)
    elif cfg["dataset"] == 'wisdom':
        dataset = WISDOMDataset(dataset_path=dataset_path, mode="train", cfg=cfg)
    else:
        raise ValueError(f"Unsupported dataset type {cfg['dataset']} in your config file")
    return torch.utils.data.DataLoader(dataset=dataset, batch_size=cfg["batch_size"],
                                       num_workers=4, shuffle=True, collate_fn=collate_fn)

def load_test_dataset(cfg, num_sample, target_samples):
    """
    加载测试数据集
    :param cfg: 配置文件
    :param num_sample: 索引数量
    :param target_samples: 索引位置 必须是列表 [a,b,c,d,e]
    :return: 返回测试数据集
    """
    dataset_path = cfg["dataset_path"]
    if cfg["dataset"] == 'synthetic':
        dataset = SyntheticDataset(dataset_path=dataset_path, mode="val", cfg=cfg, num_sample=num_sample, target_samples=target_samples)
    elif cfg["dataset"] == 'wisdom':
        dataset = WISDOMDataset(dataset_path=dataset_path, mode="val", cfg=cfg)
    else:
        raise ValueError(f"Unsupported dataset type {cfg['dataset']} in your config file")
    return torch.utils.data.DataLoader(dataset=dataset, batch_size=1,
                                       num_workers=4, shuffle=False, collate_fn=collate_fn)

# 设置日志记录
def setup_logging(cfg_name):
    current_time_stamp = time.time()
    formatted_time = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(current_time_stamp))
    logging_folder = os.path.join(os.getcwd(), 'logs_test_map', f"{cfg_name}_{formatted_time}")
    os.makedirs(logging_folder, exist_ok=True)
    summary = SummaryWriter(logdir=logging_folder)
    return logging_folder, summary

# 加载模型
def load_model(cfg, gpu):
    model = maskrcnn.get_model_instance_segmentation(cfg=cfg)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, device

def train_and_save_model(model, train_loader, device, cfg, logging_folder, save_interval, summary):
    params = [
        {'params': [p for n, p in model.named_parameters() if 'fc' not in n], 'lr': cfg['lr']},
        {'params': [p for n, p in model.named_parameters() if 'fc' in n], 'lr': cfg['lr'] * 10}
    ]
    optimizer = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["wd"])
    # 学习率衰减
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=8,
        T_mult=2,
        eta_min=0,
    )
    losses = []
    for epoch in range(cfg['max_epoch']):
        loss_value = train_one_epoch(model, optimizer, train_loader, device, epoch, 1, summary)
        losses.append(loss_value)
        # 直接使用 lr_scheduler 来更新学习率
        lr_scheduler.step()  # 更新学习率

        if epoch % save_interval == 0:
            torch.save(model.state_dict(), f'{logging_folder}/{epoch}.tar')
    return losses

# 评估模型
def evaluate_model(model, val_loader, device, summary, logging_folder):
    weights = [f for f in os.listdir(logging_folder) if f.endswith('.tar')]
    epoch_list = sorted([int(w[:-4]) for w in weights if w[-4:] == ".tar"])

    for epoch in epoch_list:
        print("Evaluating", epoch)
        model.load_state_dict(torch.load(logging_folder + "/" + str(epoch) + ".tar"))  ###########################
        coco = get_coco_api_from_dataset(val_loader.dataset)
        coco_evaluator = evaluate(coco, model, val_loader, device=device, summary=summary, epoch=epoch)
        coco_to_excel(coco_evaluator, epoch, logging_folder, "synthetic")
        print(logging_folder, "：已写入数据。")

    # coco_evaluator = evaluate(coco, model, val_loader, device=device, summary=summary, epoch=-1)
    # coco_to_excel(coco_evaluator, epoch=-1, logging_folder=logging_folder, eval_data=cfg['dataset'])
    # print("评估完成，结果已保存至Excel文件中。")

# 绘制训练损失图
def draw_train_loss_graph(losses, logging_folder, cfg_name, cfg_test):
    # 绘制训练损失图
    epochs = np.arange(1, len(losses) + 1)
    plt.plot(epochs, losses, color="red", linestyle="-", marker="o",
             markersize=4, markerfacecolor="white", label="Training Loss")
    plt.xlabel('Epoches')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    dataset_path = cfg_test['dataset_path']
    parent_folder = Path(dataset_path).parent.name
    output_file_path = os.path.join(logging_folder,
                                    cfg_name + "_" + parent_folder + "_" + "losses.png")  # 替换为想要保存的文件名和路径
    plt.savefig(output_file_path, format='png', dpi=300)  # dpi=300 以确保高分辨率
    # plt.show()
    plt.close()  # 关闭图像避免内存泄漏

# 绘制测试精度图
def draw_test_accuracy_graph(logging_folder, cfg_test, cfg_name):
    # 绘制AP散点图
    # 读取 Excel 文件的第一个 sheet
    file_path = os.path.join(logging_folder, "coco_result_synthetic.xlsx")  # 替换为实际路径
    df = pd.read_excel(file_path, sheet_name=0)

    # 提取数据
    y_values_map = df.iloc[12, 5:].values  # 第12行，第5列至最后一列
    y_values_apl = df.iloc[14, 5:].values  # 第13行，第5列至最后一列
    y_values_bbox_map = df.iloc[20, 5:].values  # 第0行，第5列至最后一列
    y_values_bbox_ap = df.iloc[22, 5:].values  # 第1行，第5列至最后一列

    # x 值为 0 到 N 的整数
    x_values = range(len(y_values_map))  # N 为数据的长度

    # 创建图形和子图
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # 函数用于添加最大 Y 值和对应 X 值的文本
    max_index_list = []

    def add_max_value_text(ax, y_values):
        max_value = y_values.max()
        max_index = y_values.argmax()  # 获取最大值的索引
        max_index_list.append(max_index)
        ax.text(0.5, 0.5, f'Max: {max_value}\nat X: {max_index}', fontsize=18,
                ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5),
                transform=ax.transAxes)

    # 绘制折线散点图
    axs[0, 0].plot(x_values, y_values_map, marker='o', label='mAP-all')
    axs[0, 0].set_title('mAP-all')
    axs[0, 0].set_xlabel('Index')
    axs[0, 0].set_ylabel('mAP-all Values')
    axs[0, 0].legend()
    add_max_value_text(axs[0, 0], y_values_map)

    axs[0, 1].plot(x_values, y_values_apl, marker='o', label='AP75-all', color='orange')
    axs[0, 1].set_title('AP75-all')
    axs[0, 1].set_xlabel('Index')
    axs[0, 1].set_ylabel('AP75-all Values')
    axs[0, 1].legend()
    add_max_value_text(axs[0, 1], y_values_apl)

    axs[1, 0].plot(x_values, y_values_bbox_map, marker='o', label='mAR-all', color='green')
    axs[1, 0].set_title('mAR-all')
    axs[1, 0].set_xlabel('Index')
    axs[1, 0].set_ylabel('mAR-all Values')
    axs[1, 0].legend()
    add_max_value_text(axs[1, 0], y_values_bbox_map)

    axs[1, 1].plot(x_values, y_values_bbox_ap, marker='o', label='mAR-medium', color='red')
    axs[1, 1].set_title('mAR-medium')
    axs[1, 1].set_xlabel('Index')
    axs[1, 1].set_ylabel('mAR-medium Values')
    axs[1, 1].legend()
    add_max_value_text(axs[1, 1], y_values_bbox_ap)

    # 调整布局
    plt.tight_layout()
    # 保存为 PNG 图像
    dataset_path = cfg_test['dataset_path']
    parent_folder = Path(dataset_path).parent.name
    output_file_path = os.path.join(logging_folder,
                                    cfg_name + "_" + parent_folder + "_" + "output.png")  # 替换为想要保存的文件名和路径
    plt.savefig(output_file_path, format='png', dpi=300)  # dpi=300 以确保高分辨率
    # 如果需要实时查看图像，可以取消注释下面这行
    # plt.show()
    fig = cv2.imread(output_file_path)
    Image.fromarray(fig).show()
    plt.close()  # 关闭图像避免内存泄漏

# 主流程
def train_eval_main():
    # 记录开始时间
    start_time = time.time()
    gpu, cfg_name, resume_training, resume_training_weight, save_interval = set_training_parameters()

    # 加载配置
    cfg = load_config(cfg_name, process="train")
    cfg_test = load_config(cfg_name, process="test")  # 加载测试配置
    print(f"Config loaded: {cfg}")

    # 数据加载
    train_loader = load_train_dataset(cfg)
    val_loader = load_test_dataset(cfg_test, target_samples=None, num_sample=None)  # 这里加载验证集

    # 设置日志文件夹
    logging_folder, summary = setup_logging(cfg_name)

    # 加载模型
    model, device = load_model(cfg, gpu)

    # 训练模型
    losses = train_and_save_model(model, train_loader, device, cfg, logging_folder, save_interval, summary)

    # 绘制训练损失图
    draw_train_loss_graph(losses, logging_folder, cfg_name, cfg_test)

    # 评估模型
    # logging_folder = "logs/rgb_depth_earlyfusion_2025-02-18_17-34-33"
    evaluate_model(model, val_loader, device, summary, logging_folder)

    # 绘制测试精度图
    draw_test_accuracy_graph(logging_folder, cfg_test, cfg_name)

    # 输出训练时间
    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    print(f"Total training time: {elapsed_time:.2f} minutes")

    # 记录结束时间
    end_time = time.time()
    # 计算经过的时间（秒）
    elapsed_time_in_seconds = end_time - start_time

    # 将经过的时间转换为分钟
    elapsed_time_in_minutes = elapsed_time_in_seconds / 60
    elapsed_time_in_minutes = np.round(elapsed_time_in_minutes, 2)

    # 输出经过的时间（分钟）
    print(f"Elapsed time: {elapsed_time_in_minutes} minutes")

    # # 邮件发送者和接收者
    # sender = 'zhangjunxiz521@163.com'
    # receiver = 'xyh1261233@163.com'
    # password = 'TLxcv38ahd4KSLqH'
    # # 邮件内容
    # message = MIMEText('主机2080Ti-训练完毕提醒', 'plain', 'utf-8')
    # message['From'] = Header("RL", 'utf-8')
    # message['To'] = Header("RLL", 'utf-8')
    # message['Subject'] = Header('本次测试代码共运行' + str(elapsed_time_in_minutes) + '分钟。', 'utf-8')
    # # 发送邮件
    # try:
    #     smtpObj = smtplib.SMTP_SSL('smtp.163.com', 465)  # 使用SSL
    #     smtpObj.login(sender, password)  # 登录验证
    #     smtpObj.sendmail(sender, [receiver], message.as_string())
    #     print("邮件发送成功")
    # except smtplib.SMTPException as e:
    #     print("Error: 无法发送邮件", e)
    # finally:
    #     smtpObj.quit()  # 断开连接

def evaluate_main(logging_folder):
    # 记录开始时间
    start_time = time.time()
    gpu, cfg_name, resume_training, resume_training_weight, save_interval = set_training_parameters()

    # 加载配置
    cfg = load_config(cfg_name, process="train")
    cfg_test = load_config(cfg_name, process="test")  # 加载测试配置
    print(f"Config loaded: {cfg}")

    # 数据加载
    train_loader = load_train_dataset(cfg)
    val_loader = load_test_dataset(cfg_test)  # 这里加载验证集

    # 设置日志文件夹
    summary = SummaryWriter(logdir=logging_folder)

    # 加载模型
    model, device = load_model(cfg, gpu)

    # 评估模型
    evaluate_model(model, val_loader, device, summary, logging_folder)

    # 绘制测试精度图
    draw_test_accuracy_graph(logging_folder, cfg_test, cfg_name)

def inference_main():
    # 模型路径
    weight_path = r"F:\2024\20241024SZ_INSEG_PAPER\RGB_NRE_SEG\logs_xizang\rgb_depth_confidencefusion_2025-12-12_19-02-50\4.tar"
    cpu_device = torch.device("cpu")

    # 根据模型路径判断模型架构
    if "depth_only" in weight_path:
        cfg_name = "depth_only"
    if "rgb_only" in weight_path:
        cfg_name = "rgb_only"
    if "rgb_depth_earlyfusion" in weight_path:
        cfg_name = "rgb_depth_earlyfusion"
    if "rgb_depth_latefusion" in weight_path:
        cfg_name = "rgb_depth_latefusion"
    if "rgb_depth_confidencefusion" in weight_path:
        cfg_name = "rgb_depth_confidencefusion"

    # 根据模型路径加载对应的数据集
    # 后续再写

    gpu, cfg_name, resume_training, resume_training_weight, save_interval = set_training_parameters2(cfg_name)

    # 加载配置
    cfg = load_config(cfg_name, process="train")
    cfg["batch_size"] = 1

    cfg_test = load_config(cfg_name, process="test")  # 加载测试配置
    print(f"Config loaded: {cfg}")

    # 数据加载
    train_loader = load_train_dataset(cfg)

    # val_loader = load_test_dataset(cfg_test, num_sample=None, target_samples=[0,1,2,3])  # 1: rezhensi_slice_10240_3584
    # val_loader = load_test_dataset(cfg_test, num_sample=None, target_samples=None)  # 全部数据推理
    val_loader = load_test_dataset(cfg_test, num_sample=None, target_samples=[0])

    # 设置日志文件夹
    logging_folder, summary = setup_logging(cfg_name)
    logging_folder = os.path.join(logging_folder, "Inference")

    # 加载模型
    model, device = load_model(cfg, gpu)
    model.load_state_dict(torch.load(weight_path))
    print(f"Model loaded: {cfg}")

    # 定义列名
    column_names = ['filename', 'file_size', 'file_attributes', 'region_count', 'region_id', 'region_shape_attributes',
                    'region_attributes']
    # 创建一个空的DataFrame，索引和列都是空的
    Inference_dataframe = pd.DataFrame(columns=column_names)
    # 设定掩膜面积处理阈值
    area_thresh = 400
    # 置信度
    thresh = 0.6
    vis_depth = False  # 是否可视化深度 如果是rgb_only需要为False参数

    save_dir = os.path.join(logging_folder, "vis_result_{}".format('synthetic'))
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    # 切换需要推理的数据集 以下只能选其一
    rgb_path = cfg_test["dataset_path"] + "/val/rgb"
    inference_loader = val_loader

    # rgb_path = dataset_path + "/train/rgb"
    # inference_loader = train_loader

    rgb_list = sorted([file for file in os.listdir(rgb_path) if file.endswith('.png')])
    # rgb_list = [file for file in os.listdir(rgb_path) if file.endswith('.png')]
    # print(f"rgb_list: {rgb_list}")
    for img_idx, (image, _) in enumerate(tqdm(inference_loader)):
        img_path = os.path.join(rgb_path, rgb_list[img_idx])
        # print("img_path:", img_path)
        img_name = rgb_list[img_idx]  # 修正 img_name 的定义
        # print("img_name:", img_name)
        image = list(img.to(device) for img in image)
        # print("image shape:", image[0].shape, len(image))
        outputs = model(image)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        # 生成标注数据框
        masks_np = outputs[0]["masks"].detach().numpy()
        scores_np = outputs[0]["scores"].detach().numpy()
        for mask_idx in range(masks_np.shape[0]):
            # 置信度筛选
            if scores_np[mask_idx] >= thresh:
                maski = masks_np[mask_idx, 0, :, :]  # 获取每个掩膜的单通道数据

                # 对掩膜进行二值化处理
                _, binary_mask = cv2.threshold(maski, 0.5, 1, cv2.THRESH_BINARY)

                # 使用cv2.findContours来找到轮廓
                contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # 如果找到轮廓
                if len(contours) > 0:
                    contour = contours[0]  # 取第一个轮廓

                    area = cv2.contourArea(contour)  # 计算该轮廓的面积
                    # 过滤掉面积小于阈值的掩膜
                    if area < area_thresh:
                        continue  # 跳过这个掩膜，不进行处理

                    contour = contour.reshape(-1, 2)  # 变成 (N, 2) 形状
                    x_indices = contour[:, 0].tolist()
                    y_indices = contour[:, 1].tolist()

                    # 创建字典
                    region_shape_attributes_dict = {
                        "name": "polygon",
                        "all_points_x": x_indices,
                        "all_points_y": y_indices
                    }

                else:
                    # print("No objects found in the mask.")
                    continue

                # 其他属性
                filename_dict = img_name
                file_size_dict = os.path.getsize(img_path)  # 获取图像文件的实际大小（字节）
                file_attributes_dict = {}
                region_count_dict = len(np.where(scores_np > thresh)[0])
                region_id_dict = mask_idx
                region_attributes_dict = {
                    "tree": "1"
                }

                # 将当前掩膜信息追加到数据框中
                new_row = {
                    'filename': filename_dict,
                    'file_size': file_size_dict,
                    'file_attributes': file_attributes_dict,
                    'region_count': region_count_dict,
                    'region_id': region_id_dict,
                    'region_shape_attributes': json.dumps(region_shape_attributes_dict),  # 转换为JSON字符串以便存入CSV
                    'region_attributes': json.dumps(region_attributes_dict)  # 转换为JSON字符串以便存入CSV
                }

                # 使用 pd.concat 而不是 append
                new_row_df = pd.DataFrame([new_row])  # 将字典转换为单行 DataFrame
                Inference_dataframe = pd.concat([Inference_dataframe, new_row_df], ignore_index=True)

        # 可以选择不显示置信度
        try:
            vis_image = draw_prediction2(image[0].to(cpu_device), outputs[0], thresh, vis_depth, show_confidence=False,
                                     transparency=0.5)
        except:
            print("No objects found in the image.")

        # 批量化执行前需要注释以下显示图像的代码
        # plt.figure(figsize=(10, 10))
        # plt.imshow(vis_image)
        # plt.show()

        save_name = "img{:03d}.png".format(img_idx)
        cv2.imwrite(os.path.join(save_dir, save_name), vis_image)
    print("Images are saved at", save_dir)
    Inference_dataframe.to_csv(os.path.join(save_dir, "InferenceDataframe.csv"), index=False)


if __name__ == "__main__":
    # 源文件路径
    # source_file = Path('./models/final_backbone/backbone_origin.py')
    # source_file = Path('./models/final_backbone/backbone_RGBH-CME-CHM-M2.py')
    # source_file = Path('./models/final_backbone/backbone_AFM-M4.py')
    source_file = Path('./models/final_backbone/backbone_AFM_RGBH-CME-CHM-M6.py')
    # 目标路径（可包含新文件名）
    destination = Path('./models/backbone.py')
    try:
        # 执行复制操作（会覆盖已存在的文件）
        shutil.copy(source_file, destination)
        print(f"{source_file}文件已成功复制并保存为：{destination}")
    except FileNotFoundError:
        print("错误：源文件不存在")
    except PermissionError:
        print("错误：没有足够的权限访问文件或目录")
    except Exception as e:
        print(f"发生未知错误：{e}")
    # 这里需要执行第二次才会生效

    # train_eval_main()

    # evaluate_main(r"F:\2024\20241024SZ_INSEG_PAPER\RGB_NRE_SEG\logs\e-M6-cme2-2_19_0-rgb_depth_confidencefusion_2025-05-13_15-09-09")

    inference_main()
