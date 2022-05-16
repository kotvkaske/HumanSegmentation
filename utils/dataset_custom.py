import os
import torch
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class SemanticSegmentationDataset(Dataset):
    """
    Датасет для семантической сегментации.
    Input:
    files_x - список строк (пути к изображениям) исходных картинок
    transform_x - преобразование каждого изображения
    files_y - список строк исходных масок
    transform_y - преобразование масок
    type_of_data - train/val/test
    """
    def __init__(self, files_x, transform_x, files_y=None, transform_y=None, type_of_data='train'):
        super(SemanticSegmentationDataset, self).__init__()
        self.files_x = files_x
        self.files_y = files_y
        self.transform_x = transform_x
        self.transform_y = transform_y
        self.type_of_data = type_of_data

    def __len__(self):
        return len(self.files_x)

    def __getitem__(self, index):
        x_label = Image.open(self.files_x[index])
        x_label = self.transform_x(x_label)
        if self.type_of_data == 'test':
            return x_label
        y_label = Image.open(self.files_y[index])
        y_label = self.transform_y(y_label)
        # y_label = (y_label[0, :, :] != 0).int()
        return x_label, y_label

    def show_pic(self, index,type_of_data='train'):
        x = Image.open(self.files_x[index])
        if self.type_of_data=='test':
            return x
        y = Image.open(self.files_y[index])
        return x, y
