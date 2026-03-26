import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import numpy as np
import cv2
import imgviz
import matplotlib.pyplot as plt


def draw_sample_images(loader, save_path, cfg, prefix, n_images=10):
    for i in range(n_images):
        img, target = loader.__getitem__(i)
        images = dict.fromkeys(["rgb", "depth", "val_mask"])
        if cfg["input_modality"] == "rgb": 
            images["rgb"] = img[:3, :, :]
        elif cfg["input_modality"] == "inpainted_depth": 
            images["depth"] = img[:3, :, :]
        elif cfg["input_modality"] == "raw_depth":
            images["depth"] = img[:3, :, :]
            images["val_mask"] = img[3, :, :]
        elif cfg["input_modality"] == "rgb_inpainted_depth":
            images["rgb"] = img[:3, :, :]
            images["depth"] = img[3:6, :, :]
        elif cfg["input_modality"] == "rgb_raw_depth":
            images["rgb"] = img[:3, :, :]
            images["depth"] = img[3:6, :, :]
            images["val_mask"] = img[6, :, :]

        for k in images.keys():
            if images[k] is None: continue
            if k == "rgb":
                images[k] = F.normalize(images[k], 
                        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], 
                        std=[1/0.229, 1/0.224, 1/0.225])
            img = F.to_pil_image(images[k])
            img.save("{}/{}_{}_{}.png".format(save_path, prefix, k, i))

# def draw_prediction(image, pred, thresh, vis_depth=False):
#
#     inv_normalize = transforms.Compose([
#                         transforms.Normalize(
#                             mean = [ 0., 0., 0. ],
#                             std = [ 1/0.229, 1/0.224, 1/0.225 ]),
#                         transforms.Normalize(
#                             mean = [ -0.485, -0.456, -0.406 ],
#                             std = [ 1., 1., 1. ]),
#                         ])
#     rgb = image[:3]
#     rgb = inv_normalize(rgb)
#     rgb = rgb.transpose(0, 2).transpose(0, 1) * 255
#     rgb = np.uint8(rgb)
#
#     scores = pred["scores"].detach().numpy()
#     boxes = pred["boxes"].detach().numpy()
#     masks = pred["masks"].detach().numpy()
#     masks[masks >= 0.5] = 1
#     masks[masks < 0.5] = 0
#     cnd = scores[:] > thresh
#
#     masks = np.array(np.squeeze(masks), dtype=np.bool_)
#     instviz = imgviz.instances2rgb(image=rgb, masks=masks[cnd, :, :], labels=list(range(len(scores[cnd]))),
#                                     captions=[str(round(x, 2)) for x in scores[cnd]])
#     plt.figure(dpi=200)
#     plt.imshow(instviz)
#     plt.axis("off")
#     instviz = imgviz.io.pyplot_to_numpy()
#     instviz = cv2.cvtColor(cv2.resize(instviz, (rgb.shape[1], rgb.shape[0])), cv2.COLOR_BGR2RGB)
#     instviz = np.hstack((rgb, instviz))
#
#     if vis_depth:
#         # add depth image
#         depth = image[3:6]
#         depth = depth.transpose(0, 2).transpose(0, 1) * 255
#         depth = np.uint8(depth)
#         instviz = np.hstack((instviz, depth))
#
#     return instviz


def draw_prediction(image, pred, thresh, vis_depth=False, show_confidence=True, transparency=0.7):
    """
    可视化分割结果
    :param image: 输入图像
    :param pred: 分割预测结果
    :param thresh: 置信度阈值
    :param vis_depth: 是否可视化深度图像
    :param show_confidence: 是否显示置信度
    """

    inv_normalize = transforms.Compose([
        transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
    ])
    
    rgb = image[:3]
    rgb = inv_normalize(rgb)
    rgb = rgb.transpose(0, 2).transpose(0, 1) * 255
    rgb = np.uint8(rgb)

    # # 根据上述进行改编
    # image = image[:3]
    # rgb_array2 = image.numpy()  # 转换为 NumPy 数组
    # rgb_array2 = (rgb_array2 * 255).astype(np.uint8)  # 转换为 [0, 255] 范围并转换为 uint8
    # # 如果需要，转换回 PIL 图像
    # rgb_image2 = rgb_array2.transpose(1, 2, 0)  # 重新调整轴顺序为 (H, W, C)
    # # 显示图像
    # rgb = rgb_image2

    scores = pred["scores"].detach().numpy()
    boxes = pred["boxes"].detach().numpy()
    masks = pred["masks"].detach().numpy()

    masks[masks >= 0.5] = 1
    masks[masks < 0.5] = 0
    cnd = scores > thresh

    masks = np.array(np.squeeze(masks), dtype=np.bool_)

    # 创建可视化
    instviz = imgviz.instances2rgb(image=rgb, masks=masks[cnd, :, :], labels=list(range(len(scores[cnd]))),
                                   captions=[str(round(x, 2)) if show_confidence else '' for x in scores[cnd]], alpha=transparency)

    plt.figure(dpi=300)
    plt.imshow(instviz)
    plt.axis("off")

    instviz = imgviz.io.pyplot_to_numpy()
    instviz = cv2.cvtColor(cv2.resize(instviz, (rgb.shape[1], rgb.shape[0])), cv2.COLOR_BGR2RGB)
    instviz = np.hstack((rgb, instviz))

    if vis_depth:
        # 添加深度图像
        depth = image[3:6]
        depth = depth.transpose(0, 2).transpose(0, 1) * 255
        depth = np.uint8(depth)
        instviz = np.hstack((instviz, depth))

    return instviz


def draw_prediction2(image, pred, thresh, vis_depth=False, show_confidence=True, transparency=0.7):
    """
    可视化分割结果
    :param image: 输入图像
    :param pred: 分割预测结果
    :param thresh: 置信度阈值
    :param vis_depth: 是否可视化深度图像
    :param show_confidence: 是否显示置信度
    """

    inv_normalize = transforms.Compose([
        transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
    ])
    
    rgb = image[:3]
    rgb = inv_normalize(rgb)
    rgb = rgb.transpose(0, 2).transpose(0, 1) * 255
    rgb = np.uint8(rgb)

    # # 根据上述进行改编
    # image = image[:3]
    # rgb_array2 = image.numpy()  # 转换为 NumPy 数组
    # rgb_array2 = (rgb_array2 * 255).astype(np.uint8)  # 转换为 [0, 255] 范围并转换为 uint8
    # # 如果需要，转换回 PIL 图像
    # rgb_image2 = rgb_array2.transpose(1, 2, 0)  # 重新调整轴顺序为 (H, W, C)
    # # 显示图像
    # rgb = rgb_image2

    scores = pred["scores"].detach().numpy()
    boxes = pred["boxes"].detach().numpy()
    masks = pred["masks"].detach().numpy()

    masks[masks >= 0.5] = 1
    masks[masks < 0.5] = 0
    cnd = scores > thresh

    masks = np.array(np.squeeze(masks), dtype=np.bool_)

    # 创建可视化
    # instviz = imgviz.instances2rgb(image=rgb, masks=masks[cnd, :, :], labels=list(range(len(scores[cnd]))),
    #                                captions=[str(round(x, 2)) if show_confidence else '' for x in scores[cnd]], alpha=transparency,
    #                                bboxes=None, line_width=1)
    # 修改后的可视化代码
    instviz = imgviz.instances2rgb(
        image=rgb,
        masks=masks[cnd, :, :],
        labels=list(range(len(scores[cnd]))),
        captions=[str(round(x, 2)) if show_confidence else '' for x in scores[cnd]],
        alpha=transparency,  # 增加不透明度
        bboxes=None,
        boundary_width=1,
        line_width=1  # 增加边界线粗度
    )

    plt.figure(dpi=600)
    plt.imshow(instviz)
    plt.axis("off")

    instviz = imgviz.io.pyplot_to_numpy()
    instviz = cv2.cvtColor(cv2.resize(instviz, (rgb.shape[1], rgb.shape[0])), cv2.COLOR_BGR2RGB)
    instviz1 = np.hstack((rgb, instviz))
    instviz2 = instviz

    if vis_depth:
        # 添加深度图像
        depth = image[3:6]
        depth = depth.transpose(0, 2).transpose(0, 1) * 255
        depth = np.uint8(depth)
        instviz = np.hstack((instviz, depth))
    plt.close()

    return instviz2
