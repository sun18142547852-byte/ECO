import torch
from torch import nn
from typing import Dict
from collections import OrderedDict
from torchvision.ops import misc as misc_nn_ops
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool
from torch.nn import functional as F
from . import resnet


class AdaptiveFusionModule(nn.Module):
    def __init__(self, in_channels):
        super(AdaptiveFusionModule, self).__init__()
        # 全局平均池化
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        reduced_channels = max(1, in_channels // 16)

        # 定义固定通道数的卷积层
        self.fc1 = nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2_rgb = nn.Conv2d(reduced_channels, in_channels, kernel_size=1, bias=False)  # 用于RGB的权重
        self.fc2_depth = nn.Conv2d(reduced_channels, in_channels, kernel_size=1, bias=False)  # 用于Depth的权重
        self.sigmoid = nn.Sigmoid()

        # 融合后的卷积层
        self.fusion_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.fusion_bn = nn.BatchNorm2d(in_channels)
        self.fusion_relu = nn.ReLU(inplace=True)

    def forward(self, rgb_features, depth_features):
        # 合并RGB和深度特征
        # fused_features = rgb_features + depth_features

        # 全局平均池化得到通道重要性
        # squeeze_tensor = self.squeeze(fused_features)  # 形状：[b, c, 1, 1]
        squeeze_tensor_rgb = self.squeeze(rgb_features)
        squeeze_tensor_depth = self.squeeze(depth_features)

        # 使用卷积层得到每个通道的权重（RGB 和 Depth 分开）
        excitation_rgb = self.fc1(squeeze_tensor_rgb)
        excitation_rgb = self.relu(excitation_rgb)
        excitation_rgb = self.fc2_rgb(excitation_rgb)  # 生成RGB特征的权重
        excitation_rgb = self.sigmoid(excitation_rgb)

        excitation_depth = self.fc1(squeeze_tensor_depth)
        excitation_depth = self.relu(excitation_depth)
        excitation_depth = self.fc2_depth(excitation_depth)  # 生成Depth特征的权重
        excitation_depth = self.sigmoid(excitation_depth)

        # 为RGB和深度特征分配权重
        rgb_weighted = rgb_features * excitation_rgb
        depth_weighted = depth_features * excitation_depth

        # 融合加权后的特征
        fused_weighted = rgb_weighted + depth_weighted

        # 使用额外的卷积层进一步处理融合后的特征
        fused_weighted = self.fusion_conv(fused_weighted)
        fused_weighted = self.fusion_bn(fused_weighted)
        fused_weighted = self.fusion_relu(fused_weighted)

        return fused_weighted


class RGBDIntermediateLayerGetter(nn.ModuleDict):
    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, rgb_model: nn.Module, depth_model: nn.Module, 
                 confidence_map, return_layers: Dict[str, str]) -> None:
        # 初始化原有模型
        if not set(return_layers).issubset([name for name, _ in rgb_model.named_children()]):
            raise ValueError("return_layers are not present in model")
        
        rgb_return_layers = {k: v for k, v in return_layers.items()}
        depth_return_layers = {k: v for k, v in return_layers.items()}
        layers = OrderedDict()
        
        # 添加RGB和深度模型的特征提取层
        for name, module in rgb_model.named_children():
            layers['rgb_' + name] = module
            if name in rgb_return_layers:
                del rgb_return_layers[name]
            if not rgb_return_layers:
                break

        for name, module in depth_model.named_children():
            layers['depth_' + name] = module
            if name in depth_return_layers:
                del depth_return_layers[name]
            if not depth_return_layers:
                break

        super(RGBDIntermediateLayerGetter, self).__init__(layers)
        self.return_layers = return_layers
        
        # 确保初始化 self.confidence_map
        self.confidence_map = confidence_map if confidence_map else None

        # 自适应融合模块，明确每层输入通道数
        self.adaptive_fusion_layers = nn.ModuleList([
            AdaptiveFusionModule(256),   # 第一个融合模块，输入通道数256
            AdaptiveFusionModule(512),   # 第二个融合模块，输入通道数512
            AdaptiveFusionModule(1024),  # 第三个融合模块，输入通道数1024
            AdaptiveFusionModule(2048)   # 第四个融合模块，输入通道数2048
        ])

        # 自信度图估计器
        if confidence_map == "estimator":
            self.confidence_map_estimator = nn.Sequential(
                nn.Conv2d(2, 1, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(1, 1, 3, 1, 1)
            )
        elif confidence_map == "self_attention":
            self.confidence_map_estimator = nn.Sequential(
                nn.Conv2d(4, 1, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(1, 1, 3, 1, 1)
            )
        else:
            self.confidence_map_estimator = None

    def forward(self, x):
        out = OrderedDict()
        rgb_out = OrderedDict()
        depth_out = OrderedDict()
        rgb_x = x[:, :3, :, :]
        depth_x = x[:, 3:6, :, :]
        
        # 自信度图估计
        if self.confidence_map is not None:
            if self.confidence_map == "estimator":
                confidence_map = self.confidence_map_estimator(x[:, -2:, :, :])
            elif self.confidence_map == "val_mask":
                confidence_map = x[:, -1, :, :].unsqueeze(1)
            elif self.confidence_map == "self_attention":
                confidence_map = self.confidence_map_estimator(x[:, :4, :, :])

            confidence_maps = {}
            _, _, H, W = x.shape
            for i in range(4):
                confidence_maps[str(i)] = F.interpolate(confidence_map, 
                                    size=(int(H / (2 ** (i + 2))), int(W / (2 ** (i + 2)))), 
                                    mode='bilinear', align_corners=True)

        # 前向传播RGB和深度特征提取
        for name, module in self.items():
            layer_name = name.split('_')[-1]
            # if 'rgb' in name:
            #     rgb_x = module(rgb_x)
            #     if layer_name in self.return_layers:
            #         out_name = self.return_layers[layer_name]
            #         rgb_out[out_name] = rgb_x
            # if 'depth' in name:
            #     depth_x = module(depth_x)
            #     if layer_name in self.return_layers:
            #         out_name = self.return_layers[layer_name]
            #         depth_out[out_name] = depth_x
            if 'depth' in name:
                depth_x = module(depth_x)
                if layer_name in self.return_layers:
                    out_name = self.return_layers[layer_name]
                    if self.confidence_map is not None and confidence_map is not None:
                        depth_out[out_name] = depth_x * confidence_maps[out_name]
                    else:
                        depth_out[out_name] = depth_x
            elif 'rgb' in name:
                rgb_x = module(rgb_x)
                if layer_name in self.return_layers:
                    out_name = self.return_layers[layer_name]
                    if self.confidence_map is not None and confidence_map is not None:
                        rgb_out[out_name] = rgb_x * confidence_maps[out_name]
                    else:
                        rgb_out[out_name] = rgb_x

        # 自适应加权融合RGB-D特征
        for i, (adaptive_fusion, key) in enumerate(zip(self.adaptive_fusion_layers, rgb_out.keys())):
            out[key] = adaptive_fusion(rgb_out[key], depth_out[key])

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


