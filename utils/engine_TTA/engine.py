import math
import sys
import time
import torch

import torchvision.models.detection.mask_rcnn
from .coco_eval import CocoEvaluator
from .metric_logger import *

from torchvision import ops
from torchvision.transforms import functional as F


def collate_fn(batch):
    return tuple(zip(*batch))

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, summary):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None

    for idx, (images , targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)

        losses = loss_dict['loss_classifier'] + loss_dict['loss_box_reg'] \
            + loss_dict['loss_mask'] + loss_dict['loss_objectness'] + loss_dict['loss_rpn_box_reg']

        curr_itr = idx + epoch*len(data_loader) + 1
        summary.add_scalar('Loss/train/loss_classifier', loss_dict['loss_classifier'].item(), curr_itr)
        summary.add_scalar('Loss/train/loss_box_reg', loss_dict['loss_box_reg'].item(), curr_itr)
        summary.add_scalar('Loss/train/loss_mask', loss_dict['loss_mask'].item(), curr_itr)
        summary.add_scalar('Loss/train/loss_objectness', loss_dict['loss_objectness'].item(), curr_itr)
        summary.add_scalar('Loss/train/loss_rpn_box_reg', loss_dict['loss_rpn_box_reg'].item(), curr_itr)
        summary.add_scalar('Loss/train/loss_total', losses, curr_itr)


        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return loss_value


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


# @torch.no_grad()
# # @torch.no_grad() 表示在此函数中禁用梯度计算，减少内存占用并提高推理速度
# def evaluate(coco, model, data_loader, device, summary, epoch):
#     # 保存当前线程数量，并将其设置为1以避免GPU线程干扰
#     n_threads = torch.get_num_threads()
#     # FIXME: 应该移除此设置，并使 paste_masks_in_image 函数在GPU上运行
#     torch.set_num_threads(1)
    
#     # 定义将模型输出移动到CPU设备，防止GPU显存过多占用
#     cpu_device = torch.device("cpu")
    
#     # 切换模型为评估模式
#     model.eval()
    
#     # 初始化用于记录评估过程的日志对象
#     metric_logger = MetricLogger(delimiter="  ")
#     header = 'Test:'  # 日志输出的头信息

#     # 获取模型支持的IoU类型（例如bbox，segm等）
#     iou_types = _get_iou_types(model)
#     # 初始化COCO评估器
#     coco_evaluator = CocoEvaluator(coco, iou_types)

#     # 遍历数据加载器，评估每张图片
#     for image, targets in metric_logger.log_every(data_loader, 5, header):
#         # 将每张图片数据转移到指定的设备（例如GPU）
#         image = list(img.to(device) for img in image)
#         # 将目标数据（标注）转移到指定设备
#         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

#         # 同步CUDA操作，确保数据已传输完毕
#         torch.cuda.synchronize()
        
#         # 记录模型推理时间
#         model_time = time.time()
#         # 获取模型输出
#         outputs = model(image)

#         # 将模型输出转移到CPU设备以进行评估
#         outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
#         model_time = time.time() - model_time  # 计算模型推理时间

#         # 将预测结果与对应的图片ID匹配
#         res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        
#         # 记录评估器处理时间
#         evaluator_time = time.time()
#         # 更新评估器结果
#         coco_evaluator.update(res)
#         evaluator_time = time.time() - evaluator_time  # 计算评估器处理时间

#         # 更新日志，包括模型时间和评估时间
#         metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

#     # 同步所有进程之间的统计信息
#     metric_logger.synchronize_between_processes()
#     print("Averaged stats:", metric_logger)
    
#     # 同步评估器在所有进程中的状态
#     coco_evaluator.synchronize_between_processes()

#     # 累计所有图像的预测结果进行总结
#     coco_evaluator.accumulate()
#     coco_evaluator.summarize()

#     # 恢复线程数量到之前的设置
#     torch.set_num_threads(n_threads)

#     # 为每种IoU类型添加评估指标到summary对象（用于记录训练过程中的指标变化）
#     for iou_type, coco_eval in coco_evaluator.coco_eval.items():
        
#         p = coco_eval.stats  # 提取评估结果中的统计信息

#         # 添加不同AP和AR指标到summary中
#         summary.add_scalar('{}/AP'.format(iou_type), p[0], epoch)          # 平均精度（所有IoU阈值下的平均值）
#         summary.add_scalar('{}/AP_50'.format(iou_type), p[1], epoch)       # 在IoU阈值0.50下的平均精度
#         summary.add_scalar('{}/AP_75'.format(iou_type), p[2], epoch)       # 在IoU阈值0.75下的平均精度
#         summary.add_scalar('{}/AP_S'.format(iou_type), p[3], epoch)        # 小目标的平均精度
#         summary.add_scalar('{}/AP_M'.format(iou_type), p[4], epoch)        # 中等目标的平均精度
#         summary.add_scalar('{}/AP_L'.format(iou_type), p[5], epoch)        # 大目标的平均精度

#         summary.add_scalar('{}/AR_maxDets=1'.format(iou_type), p[6], epoch)      # 召回率（最大检测数为1）
#         summary.add_scalar('{}/AR_maxDets=10'.format(iou_type), p[7], epoch)     # 召回率（最大检测数为10）
#         summary.add_scalar('{}/AR_maxDets=100'.format(iou_type), p[8], epoch)    # 召回率（最大检测数为100）
#         summary.add_scalar('{}/AR_S_maxDets=100'.format(iou_type), p[9], epoch)  # 小目标的召回率
#         summary.add_scalar('{}/AR_M_maxDets=100'.format(iou_type), p[10], epoch) # 中等目标的召回率
#         summary.add_scalar('{}/AR_L_maxDets=100'.format(iou_type), p[11], epoch) # 大目标的召回率

#     # 返回评估结果
#     return coco_evaluator


def merge_augmentations(tta_outputs, iou_threshold=0.5):
    final_outputs = []

    for outputs in zip(*tta_outputs):  # 对不同增强结果的输出进行合并
        combined_output = {'boxes': [], 'masks': [], 'scores': [], 'labels': []}
        
        # 将不同增强结果中的 bbox、mask、scores 和 labels 合并
        boxes = [output['boxes'] for output in outputs]
        masks = [output['masks'] for output in outputs]
        scores = [output['scores'] for output in outputs]
        labels = [output['labels'] for output in outputs]

        # 将所有增强后的 boxes 进行 IoU 计算
        all_boxes = torch.cat(boxes, dim=0)
        all_masks = torch.cat(masks, dim=0)
        all_scores = torch.cat(scores, dim=0)
        all_labels = torch.cat(labels, dim=0)

        # 使用 NMS（非极大值抑制）来合并重叠的 bbox
        keep = ops.nms(all_boxes, all_scores, iou_threshold)

        # 将合并后的结果添加到 final_output
        combined_output['boxes'] = all_boxes[keep]
        combined_output['masks'] = all_masks[keep]
        combined_output['scores'] = all_scores[keep]
        combined_output['labels'] = all_labels[keep]

        final_outputs.append(combined_output)

    return final_outputs


def reverse_flip_boxes(boxes, image_size, flip_type='horizontal'):
    """
    还原水平或垂直翻转后的 bbox 坐标。
    boxes: 形如 [xmin, ymin, xmax, ymax]
    image_size: (width, height)
    flip_type: 'horizontal' 或 'vertical'
    """
    w, h = image_size
    if flip_type == 'horizontal':
        # 水平翻转时，x 轴需要变化
        boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
    elif flip_type == 'vertical':
        # 垂直翻转时，y 轴需要变化
        boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
    return boxes



@torch.no_grad()
def evaluate(coco, model, data_loader, device, summary, epoch):
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'
    
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for image, targets in metric_logger.log_every(data_loader, 5, header):
        image = list(img.to(device) for img in image)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # 初始化 TTA 结果列表
        tta_outputs = []
        
        # 正常推理
        outputs = model(image)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        tta_outputs.append(outputs)

        # 水平翻转推理
        flipped_image = [F.hflip(img) for img in image]
        outputs = model(flipped_image)

        # 还原水平翻转的 mask 和 bbox
        outputs = [{k: (F.hflip(v.to(cpu_device)) if k == 'masks' else reverse_flip_boxes(v.to(cpu_device), image[0].shape[-2:], 'horizontal') if k == 'boxes' else v.to(cpu_device)) for k, v in t.items()} for t in outputs]
        tta_outputs.append(outputs)

        # 垂直翻转推理
        flipped_image = [F.vflip(img) for img in image]
        outputs = model(flipped_image)

        # 还原垂直翻转的 mask 和 bbox
        outputs = [{k: (F.vflip(v.to(cpu_device)) if k == 'masks' else reverse_flip_boxes(v.to(cpu_device), image[0].shape[-2:], 'vertical') if k == 'boxes' else v.to(cpu_device)) for k, v in t.items()} for t in outputs]
        tta_outputs.append(outputs)

        # 合并增强后的输出结果
        final_outputs = merge_augmentations(tta_outputs)

        # 更新COCO评估器
        res = {target["image_id"].item(): output for target, output in zip(targets, final_outputs)}
        coco_evaluator.update(res)

    # 同步并总结结果
    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    torch.set_num_threads(n_threads)
    return coco_evaluator
