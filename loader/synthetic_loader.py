from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pandas as pd
import os
import numpy as np
from PIL import Image
import torch
from .transforms import AddGaussianNoise, RandomErasing, SaltPepperNoise
import cv2
from torchvision.transforms import functional as F


class SyntheticDataset(Dataset):

    def __init__(self, dataset_path, mode="train", cfg=None, num_sample=None, target_samples=None):

        self.mode = mode  # train mode or validation mode
        self.rgb_path = os.path.join(dataset_path, self.mode, "rgb")
        # depth_folder = "depth_sparse_perlin" if cfg["input_modality"] in "raw" else "depth_perlin"
        depth_folder = "depth_noisy" if "raw" in cfg["input_modality"] else "depth_value"
        self.depth_path = os.path.join(dataset_path, self.mode, depth_folder)
        self.seg_path = os.path.join(dataset_path, self.mode, "seg")

        if mode == "train":
            self.rgb_list = sorted([file for file in os.listdir(self.rgb_path) if file.endswith('.png')])##############
            self.depth_list = list(sorted(os.listdir(self.depth_path)))
            self.seg_list = list(sorted(os.listdir(self.seg_path)))
        elif mode == "val":
            self.rgb_list = sorted([file for file in os.listdir(self.rgb_path) if file.endswith('.png')])##############
            self.depth_list = list(sorted(os.listdir(self.depth_path)))
            self.seg_list = list(sorted(os.listdir(self.seg_path)))
            # print("rgb_list:", self.rgb_list)

        if num_sample is not None:
            self.rgb_list = self.rgb_list[:num_sample]
            self.depth_list = self.depth_list[:num_sample]
            self.seg_list = self.seg_list[:num_sample]

        # 根据 target_samples 只加载指定的样本
        if target_samples is not None:
            self.rgb_list = [self.rgb_list[i] for i in target_samples]
            self.depth_list = [self.depth_list[i] for i in target_samples]
            self.seg_list = [self.seg_list[i] for i in target_samples]

        print(mode, ":", len(self.rgb_list), "images")
        self.input_modality = cfg["input_modality"]
        self.width = cfg["width"]
        self.height = cfg["height"]
        # self.min_depth = 250 # mm
        # self.max_depth = 1250 # mm
        self.min_depth = cfg["depth_norm"]["min"] if "depth_norm" in cfg else 0
        self.max_depth = cfg["depth_norm"]["max"] if "depth_norm" in cfg else 1
        
        # 对RGB图像进行数据增强和标准化处理，使得输入数据更具多样性，能够有效提高模型的泛化能力和鲁棒性。
        if mode == "train":
            self.rgb_transform = transforms.Compose([
                                        transforms.ColorJitter(brightness=0.2,
                                            contrast=0.4,
                                            saturation=0.3,
                                            hue=0.25),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225]),
                                        AddGaussianNoise(mean=0., std=0.05)
                                        ])
            # self.rgb_transform = transforms.Compose([
            #     transforms.ToTensor(),
            #     transforms.Normalize(
            #         mean=[0.485, 0.456, 0.406],
            #         std=[0.229, 0.224, 0.225]),
            # ])
        else:
            self.rgb_transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
                            ])

    def __getitem__(self, idx):
        inputs = dict.fromkeys(["rgb", "depth", "val_mask"])
        if 'rgb' in self.input_modality:
            # print(os.path.join(self.rgb_path, self.rgb_list[idx]))
            rgb = Image.open(os.path.join(self.rgb_path, self.rgb_list[idx])).convert("RGB")
            rgb = rgb.resize((self.width, self.height))
            inputs["rgb"] = self.rgb_transform(rgb)
        
        if 'depth' in self.input_modality:
            depth = np.load(os.path.join(self.depth_path, self.depth_list[idx])).astype(np.float32)
            depth = np.nan_to_num(depth)
            depth = cv2.resize(depth, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
            # print("depth_shape:", depth.shape, "len(depth.shape):", len(depth.shape))
            len_depth_shape = len(depth.shape)
            depth_shape = depth.shape
            if len_depth_shape == 2 or depth_shape[-1] == 1:
                # 动态计算深度图像的最小值和最大值
                # self.min_depth = depth.min().item()
                # self.max_depth = depth.max().item()

                # 替换无效值
                depth[depth == 9999.0] = 0  # 替换为 NaN
                # 过滤极端无效值
                # depth[depth <= -0.1] = 0  # 过滤掉极端无效值

                depth = torch.from_numpy(depth).unsqueeze(-1).permute(2, 0, 1)

                self.min_depth = depth.min().item()  # 计算有效值的最小值
                self.max_depth = depth.max().item()  # 计算有效值的最大值
                self.range_val = self.max_depth - self.min_depth
                if self.range_val == 0:
                    self.range_val = 1e-6  # 避免除以零

                # create corresponding validity mask
                val_mask = torch.ones([self.height, self.width])
                val_mask[np.where(depth[0] == 0.0)] = 0
                val_mask = val_mask.unsqueeze(0)

                # depth clip & normalization
                depth[depth < self.min_depth] = self.min_depth
                depth[depth > self.max_depth] = self.max_depth
                depth = (depth - self.min_depth) / (self.range_val)

                # to imitate noise of raw depth map, add random erase + S&P noise
                # if "raw" in self.input_modality:
                #     depth, val_mask = RandomErasing(depth, val_mask)
                #     depth, val_mask = SaltPepperNoise(depth, val_mask)
                inputs["depth"] = torch.repeat_interleave(depth, 3, 0)
                # print("depth_shape:", inputs["depth"].shape)
                inputs["val_mask"] = val_mask
                # print("valmask_shape:", inputs["val_mask"].shape)  # valmask_shape: torch.Size([1, 512, 512])

            elif len_depth_shape == 3 and depth_shape[-1] == 3:
                # 替换无效值
                depth[depth == 9999.0] = 0  # 替换为 NaN
                # 过滤极端无效值
                # depth[depth <= -0.1] = 0  # 过滤掉极端无效值

                # create corresponding validity mask
                val_mask = torch.ones([self.height, self.width])
                val_mask[np.where(depth[:, :, -1] == 0.0)] = 0
                val_mask = val_mask.unsqueeze(0)

                inputs["depth"] = torch.from_numpy(depth).permute(2, 0, 1)
                # print("depth_shape:", inputs["depth"].shape)
                inputs["val_mask"] = val_mask
                # print("valmask_shape:", inputs["val_mask"].shape)

            elif len_depth_shape == 3 and depth_shape[-1] > 3: # 多通道光谱指数数据
                # 替换无效值
                depth[depth == 9999.0] = 0  # 替换为 NaN
                threshold = 0
                a = 2
                b = 19
                c = 0
                # e_renzhensi_rsi: a = 2 b = 19 c = 0
                # # 设置每个波段的上下限（根据数据设定）
                # min_a, max_a = np.min(depth[:, :, a]) + 0.00001, np.max(depth[:, :, a])
                # min_b, max_b = np.min(depth[:, :, b]) + 0.00001, np.max(depth[:, :, b])
                # min_c, max_c = np.min(depth[:, :, c]) + 0.00001, np.max(depth[:, :, c])
                # # 归一化操作
                # depth[:, :, a] = (depth[:, :, a] - min_a) / (max_a - min_a) * 255
                # depth[:, :, b] = (depth[:, :, b] - min_b) / (max_b - min_b) * 255
                # depth[:, :, c] = (depth[:, :, c] - min_c) / (max_c - min_c) * 255

                # rsi_combination = [depth[:, :, a], depth[:, :, b], depth[:, :, c]]
                threshold_chm = threshold * depth[:, :, 0]
                rsi_combination = np.stack([depth[:, :, a] * threshold_chm,
                                            depth[:, :, b] * threshold_chm,
                                            depth[:, :, c] * threshold_chm], axis=-1)
                # print("rsi_combination:", rsi_combination.shape)
                # create corresponding validity mask
                val_mask = torch.ones([self.height, self.width])
                val_mask[np.where(rsi_combination[:, :, -1] == 0.0)] = 0
                val_mask = val_mask.unsqueeze(0)

                inputs["depth"] = torch.from_numpy(rsi_combination).permute(2, 0, 1)
                # print("depth_shape:", inputs["depth"].shape)
                inputs["val_mask"] = val_mask

        
        if inputs["rgb"] is not None and inputs["depth"] is not None:
            img = torch.cat((inputs["rgb"], inputs["depth"]), 0)
        elif inputs["rgb"] is not None:
            img = inputs["rgb"]
        elif inputs["depth"] is not None:
            img = inputs["depth"]
        if inputs["val_mask"] is not None:
            img = torch.cat((img, inputs["val_mask"]), 0)

        seg_mask_path = os.path.join(self.seg_path, self.seg_list[idx])
        seg_mask = Image.open(seg_mask_path).convert("L")
        seg_mask = seg_mask.resize((self.width, self.height), Image.NEAREST)
        seg_mask = np.array(seg_mask)       
        # instances are encoded as different colors
        obj_ids = np.unique(seg_mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        # split the color-encoded mask into a set
        # of binary masks
        seg_masks = seg_mask == obj_ids[:, None, None]
        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        temp_obj_ids = []
        temp_masks = []
        boxes = []
        for i in range(num_objs):
            pos = np.where(seg_masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            if int(xmax-xmin) < 1 or int(ymax-ymin) < 1 :
                continue
            temp_masks.append(seg_masks[i])
            temp_obj_ids.append(obj_ids[i])
            boxes.append([xmin, ymin, xmax, ymax])
        obj_ids = temp_obj_ids
        seg_masks = np.asarray(temp_masks)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = []
        for obj_id in obj_ids:
            if 1 <= obj_id:
                labels.append(1) 
            else:
                print("miss value error")
                exit(0)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        seg_masks = torch.as_tensor(seg_masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        try:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])  # 正常计算面积
        except:
            area = 0
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = seg_masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        return img, target

    def __len__(self):
        return len(self.rgb_list)