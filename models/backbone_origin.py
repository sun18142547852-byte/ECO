import os.path
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from typing import Dict
from collections import OrderedDict
from torchvision.ops import misc as misc_nn_ops
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool
from torch.nn import functional as F
from . import resnet
from typing import Union, List


def visualize_feature_map(tensor: torch.Tensor, channels: Union[int, List[int]] = 0, title_prefix: str = "Channel"):
    """
    可视化特征图的指定通道。

    参数:
    - tensor: torch.Tensor，形状应为 [B, C, H, W]
    - channels: int 或 List[int]，要可视化的通道索引
    - title_prefix: str，图像标题前缀

    用法:
    - visualize_feature_map(out[0], 0)  # 展示第0通道
    - visualize_feature_map(out[0], [0, 1, 2, 3])  # 展示前4个通道
    """

    # 安全检查
    if tensor.dim() != 4:
        raise ValueError("输入张量必须是4维 [B, C, H, W]")
    if isinstance(channels, int):
        channels = [channels]

    # 提取特征图
    feature_maps = tensor[0]  # 假设 batch size = 1
    num_channels = len(channels)

    # 自动设置图像布局
    cols = min(num_channels, 4)
    rows = int(np.ceil(num_channels / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))

    # 如果只有一个图，axs不是列表
    if num_channels == 1:
        axs = [axs]

    axs = np.array(axs).flatten()

    for i, ch in enumerate(channels):
        fm = feature_maps[ch].detach().cpu().numpy()
        axs[i].imshow(fm, cmap='viridis')
        axs[i].set_title(f"{title_prefix} {ch}")
        axs[i].axis('off')

    # 隐藏多余子图
    for j in range(i + 1, len(axs)):
        axs[j].axis('off')

    plt.tight_layout()
    # plt.savefig(os.path.join(r"F:\2024\20241024SZ_INSEG_PAPER\RGB_NRE_SEG\logs", f"{title_prefix}_visualization.png"))
    plt.close()


# 全局变量，记录保存次数
_save_counter = 0

def save_feature_maps(x):
    global _save_counter  # 声明使用全局变量

    save_dir = os.path.join("logs/val", str(_save_counter))
    os.makedirs(save_dir, exist_ok=True)

    for k, v in x.items():
        np_array = v.detach().cpu().numpy()
        if k == "pool":
            k = 4
        save_path = os.path.join(save_dir, f"{k}.npy")
        # np.save(save_path, np_array)
        print(f"Saved x[{k}] to {save_path}")

    _save_counter += 1  # 推理次数加1


class RGBDIntermediateLayerGetter(nn.ModuleDict):

    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, rgb_model: nn.Module, depth_model: nn.Module, 
                confidence_map, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset([name for name, _ in rgb_model.named_children()]):
            raise ValueError("return_layers are not present in model")
        
        rgb_return_layers = {k: v for k, v in return_layers.items()}
        depth_return_layers = {k: v for k, v in return_layers.items()}
        layers = OrderedDict()
        # rgb
        for name, module in rgb_model.named_children():
            layers['rgb_' + name] = module
            if name in rgb_return_layers:
                del rgb_return_layers[name]
            if not rgb_return_layers:
                break
        # depth
        for name, module in depth_model.named_children():
            layers['depth_' + name] = module
            if name in depth_return_layers:
                del depth_return_layers[name]
            if not depth_return_layers:
                break

        super(RGBDIntermediateLayerGetter, self).__init__(layers)
        self.return_layers = return_layers
        # fusion module 
        self.fusion_layers = nn.Sequential(
                                nn.Conv2d(512, 256, 3, 1, 2),
                                nn.Conv2d(1024, 512, 3, 1, 2),
                                nn.Conv2d(2048, 1024, 3, 1, 2),
                                nn.Conv2d(4096, 2048, 3, 1, 2)
                                        )
                                          

        # confidence map estimator with five 3x3 convolutional layer
        self.confidence_map = confidence_map
        if confidence_map == "estimator":
            # depth + val mask
            self.confidence_map_estimator = nn.Sequential(
                                            nn.Conv2d(2, 1, 3, 1, 2),
                                            nn.Conv2d(1, 1, 3, 1, 2),
                                            nn.Conv2d(1, 1, 1, 1, 0),
                                            nn.Conv2d(1, 1, 1, 1, 0),
                                            nn.Conv2d(1, 1, 1, 1, 0)
                                                )            
        elif confidence_map == "self_attention":
            # RGB + depth
            self.confidence_map_estimator = nn.Sequential(
                                            nn.Conv2d(4, 1, 3, 1, 2),
                                            nn.Conv2d(1, 1, 3, 1, 2),
                                            nn.Conv2d(1, 1, 1, 1, 0),
                                            nn.Conv2d(1, 1, 1, 1, 0),
                                            nn.Conv2d(1, 1, 1, 1, 0)
                                                )          

    def forward(self, x):
        out = OrderedDict()
        rgb_out = OrderedDict()
        depth_out = OrderedDict()
        rgb_x = x[:, :3, :, :]  # 对应了 RGB 图像的前3个通道
        depth_x = x[:, 3:6, :, :]  # 提取了第4到第6个通道
        # confidence map estimation
        if self.confidence_map == "estimator":
            confidence_map = self.confidence_map_estimator(x[:, -2:, :, :]) # depth + val_mask
        elif self.confidence_map == "val_mask":
            confidence_map = x[:, -1, :, :].unsqueeze(1) # val_mask
        elif self.confidence_map == "self_attention":
            confidence_map = self.confidence_map_estimator(x[:, :4, :, :]) # RGB + depth

        # resize confidence map for hierarchical fusion (conv2, conv3, conv4, conv5)
        # store them to list (confidence_maps)
        if self.confidence_map is not None:
            confidence_maps = {}
            _, _, H, W = x.shape
            for i in range(4):
                confidence_maps[str(i)] = F.interpolate(confidence_map, 
                                    size=(int(H/(2**(i+2))), int(W/(2**(i+2)))), 
                                    mode='bilinear', align_corners=True)
        # forward in rgb, depth branch
        for name, module in self.items():
            layer_name = name.split('_')[-1]
            if 'rgb' in name:
                rgb_x = module(rgb_x)
                if layer_name in self.return_layers:
                    out_name = self.return_layers[layer_name]
                    rgb_out[out_name] = rgb_x
            elif 'depth' in name:
                depth_x = module(depth_x)
                if layer_name in self.return_layers:
                    out_name = self.return_layers[layer_name]
                    if self.confidence_map is not None:
                        depth_out[out_name] = depth_x * confidence_maps[out_name]
                    else:
                        depth_out[out_name] = depth_x

        # concatenate rgb-d features in 4 hierarchical level and fuse them
        for i in range(4):
            out[str(i)] = torch.cat((rgb_out[str(i)], depth_out[str(i)]), dim=1)
        # hierarchical rgb-d fusion
        for i in range(4):
            out[str(i)] = self.fusion_layers[i](out[str(i)])
        return out



class IntermediateLayerGetter(nn.ModuleDict):

    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class BackboneWithFPN(nn.Module):

    def __init__(self, backbones, input_modality, fusion_method, 
                return_layers, in_channels_list, out_channels, extra_blocks=None):
        super(BackboneWithFPN, self).__init__()
        
        self.input_modality = input_modality
        self.fusion_method = fusion_method
        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()
        if input_modality == "rgb":
            self.body = IntermediateLayerGetter(backbones["rgb"], return_layers=return_layers)
        elif input_modality in ["raw_depth", "inpainted_depth"]:
            self.body = IntermediateLayerGetter(backbones["depth"], return_layers=return_layers)
        elif input_modality in ["rgb_raw_depth", "rgb_inpainted_depth"] and fusion_method == "early":
            self.body = IntermediateLayerGetter(backbones["rgbd"], return_layers=return_layers)
        elif input_modality in ["rgb_raw_depth", "rgb_inpainted_depth"] and fusion_method == "late":
            self.body = RGBDIntermediateLayerGetter(backbones["rgb"], backbones["depth"], 
                                                    confidence_map=None, return_layers=return_layers)
        elif input_modality in ["rgb_raw_depth", "rgb_inpainted_depth"] and fusion_method == "confidence_map_estimator":
            self.body = RGBDIntermediateLayerGetter(backbones["rgb"], backbones["depth"], 
                                                    confidence_map="estimator", return_layers=return_layers)
        elif input_modality in ["rgb_raw_depth", "rgb_inpainted_depth"] and fusion_method == "val_mask_as_confidence_map":
            self.body = RGBDIntermediateLayerGetter(backbones["rgb"], backbones["depth"], 
                                                    confidence_map="val_mask", return_layers=return_layers)
        elif input_modality in ["rgb_raw_depth", "rgb_inpainted_depth"] and fusion_method == "self_attention_as_confidence_map":
            self.body = RGBDIntermediateLayerGetter(backbones["rgb"], backbones["depth"], 
                                                    confidence_map="self_attention", return_layers=return_layers)
        else:
            print("Unsupported", input_modality, fusion_method)
            raise NotImplementedError

        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
        )
        self.out_channels = out_channels

    def forward(self, x):
        if self.input_modality in ["rgb", "raw_depth", "inpainted_depth"]:
            x = x[:, :3, :, :]
        if self.input_modality in ["rgb_raw_depth", "rgb_inpainted_depth"]: 
            if self.fusion_method == "early" :
                x = x[:, :4, :, :]
            if self.fusion_method in ["late", "confidence"] :
                x = x[:, :7, :, :]
        x = self.body(x)
        x = self.fpn(x)

        # 可视化特征图保存
        # save_feature_maps(x)
        return x


def get_backbone_with_fpn(input_modality, fusion_method, backbone_name, 
                        pretrained_backbone, trainable_layers, extra_blocks=None, returned_layers=None):

    backbones = dict.fromkeys(["rgb", "depth", "rgbd"])
    if input_modality == "rgb":
        backbones["rgb"] = resnet.__dict__[backbone_name](pretrained=True, norm_layer=misc_nn_ops.FrozenBatchNorm2d)
    elif input_modality in ["raw_depth", "inpainted_depth"]:
        backbones["depth"] = resnet.__dict__[backbone_name](pretrained=False, norm_layer=misc_nn_ops.FrozenBatchNorm2d)
    elif input_modality in ["rgb_raw_depth", "rgb_inpainted_depth"] and fusion_method == "early":
        backbones["rgbd"] = resnet.__dict__[backbone_name](pretrained=True, norm_layer=misc_nn_ops.FrozenBatchNorm2d)
        backbones["rgbd"].conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
    elif input_modality in ["rgb_raw_depth", "rgb_inpainted_depth"] \
            and fusion_method in ["late", "confidence_map_estimator", "val_mask_as_confidence_map", "self_attention_as_confidence_map"]:
        backbones["rgb"] = resnet.__dict__[backbone_name](pretrained=True, norm_layer=misc_nn_ops.FrozenBatchNorm2d)
        backbones["depth"] = resnet.__dict__[backbone_name](pretrained=False, norm_layer=misc_nn_ops.FrozenBatchNorm2d)
    else:
        print("Unsupported", input_modality, fusion_method)
        raise NotImplementedError

    assert trainable_layers <= 5 and trainable_layers >= 0
    layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:trainable_layers]
    # freeze layers single if pretrained backbone is used
    for k in backbones:
        if backbones[k] is None:
            continue
        for name, parameter in backbones[k].named_parameters():
            if all([not name.startswith(layer) for layer in layers_to_train]):
                parameter.requires_grad_(False)
        in_channels_stage2 = backbones[k].inplanes // 8
    if extra_blocks is None:
        extra_blocks = LastLevelMaxPool()
    if returned_layers is None:
        returned_layers = [1, 2, 3, 4]
    assert min(returned_layers) > 0 and max(returned_layers) < 5
    return_layers = {f'layer{k}': str(v) for v, k in enumerate(returned_layers)}
    in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
    out_channels = 256

    return BackboneWithFPN(backbones, input_modality, fusion_method, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks)   


