from utils.prompt_points import label_to_point_prompt_global
from torch.utils.data import Dataset
import pandas as pd
import cv2
import os
import random
import numpy as np
import albumentations as alb
from tqdm import tqdm
import torch
import glob
import tifffile
from utils.mask_convertor import DendriteSWC


class DatasetLoader(Dataset):
    def __init__(self, 
                 prompt_positive_num=-1, 
                 prompt_negative_num=-1, 
                 is_training=True,
                 shuffle = False,
                 data_dir = ''):
        
        self.prompt_positive_num = prompt_positive_num
        self.prompt_negative_num = prompt_negative_num
        self.is_training = is_training
        self.shuffle = shuffle
        
        files = glob.glob(f'{data_dir}/*')
        images = []
        labels = []
        for file in files:
            with tifffile.TiffFile(file) as tif:
                for i in range(0,len(tif.pages),3):
                    images.append(np.expand_dims(tif.pages[i].asarray(), axis=0))
                    label = tif.pages[i+1].asarray()
                    label = (label - label.min()) / ((label.max() - label.min()) + 1e-5)
                    labels.append(label)

        if self.shuffle:
            combined = list(zip(images, labels))
            # Shuffle the combined list
            random.shuffle(combined)
            shuffled_images, shuffled_labels = zip(*combined)
            images = list(shuffled_images)
            labels = list(shuffled_labels)

        self.images = images
        self.labels = labels

        self.transform = alb.Compose([
            alb.RandomBrightnessContrast(p=0.25),    
            alb.Rotate(limit=10, border_mode=cv2.BORDER_REFLECT, p=0.5),
            alb.RandomRotate90(p=0.5),
            alb.HorizontalFlip(p=0.5),
            alb.VerticalFlip(p=0.5),
            alb.Blur(p=0.2)], p=1,
            additional_targets={
            'mask': 'mask'}
            )

        self.add_gaus_noise = alb.Compose([
            alb.GaussNoise(p=0.5)], p=1,
            additional_targets={
            'mask': 'mask'}
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        """
        Retrieves and processes the image and corresponding data at the given index.

        Parameters:
        index (int): Index of the data to retrieve.

        Returns:
        tuple: Contains the processed image, prompt points, prompt types, selected component, and index.
            - image (torch.Tensor): Normalized image tensor with shape (1, C, H, W).
            - prompt_points (torch.Tensor): Tensor of prompt points coordinates with shape (1, N, 2).
            - prompt_type (torch.Tensor): Tensor indicating the type of each prompt point with shape (1, N).
            - selected_component (torch.Tensor): Tensor of the selected component with shape (1, H, W).
            - index (int): The index of the retrieved data.
        """

        image, prompt_points, prompt_type, selected_component = self.get_sam_item(self.images[index], self.labels[index])  
        image = (image - image.min()) / ((image.max() - image.min()) + 1e-5)
        image = torch.from_numpy(image[None,...])
        prompt_points = torch.from_numpy(prompt_points[None,...])
        prompt_type = torch.from_numpy(prompt_type[None,...])
        selected_component = torch.from_numpy(selected_component[None,...])

        if prompt_points.shape[-1] == 0:
            prompt_points = torch.tensor(np.array([[0,0]])[None,...])
            prompt_type = torch.tensor(np.array([0.])[None,...])

        return image, prompt_points, prompt_type, selected_component, index

    def get_sam_item(self, image, label):
        
        """
        Applies transformations to the input image and label, and generates prompt points for training.

        Parameters:
        image (ndarray): The input image array with shape (C, H, W).
        label (ndarray): The corresponding label/mask array with shape (H, W).

        Returns:
        tuple: Contains the transformed image, prompt points, prompt types, and the selected component.
            - image (ndarray): Transformed image array with shape (C, H, W).
            - prompt_points (ndarray): Array of prompt points coordinates.
            - prompt_type (ndarray): Array indicating the type of each prompt point (positive or negative).
            - selected_component (ndarray): The selected component from the label.
        """

        if self.is_training:
            transformed = self.transform(**{"image": image.transpose((1,2,0)), "mask": label[np.newaxis,:].transpose((1,2,0))})
            image, label = transformed["image"].transpose((2,0,1)), transformed["mask"].transpose((2,0,1))[0]
        ppn, pnn = self.prompt_positive_num, self.prompt_negative_num
        selected_component, prompt_points_pos, prompt_points_neg = label_to_point_prompt_global(label, ppn, pnn)

        prompt_type = np.array([1] * len(prompt_points_pos) + [0] * len(prompt_points_neg))
        prompt_points = np.array(prompt_points_pos + prompt_points_neg)

        if self.is_training:
            # print('Applying Gauss Noise')
            transformed = self.add_gaus_noise(**{"image": image.transpose((1,2,0)), "mask": label[np.newaxis,:].transpose((1,2,0))})
            image, _ = transformed["image"].transpose((2,0,1)), transformed["mask"].transpose((2,0,1))[0]
        
        return image, prompt_points, prompt_type, selected_component

class TestDatasetLoader(Dataset):
    def __init__(self, mask_name, filename = ''):

        data_dir = '/'.join(filename.split('/')[:-1])

        images = []
        with tifffile.TiffFile(filename) as tif:
            for i in range(len(tif.pages)):
                images.append(np.expand_dims(tif.pages[i].asarray(), axis=0))

        self.images = images

        if len(mask_name)>2:
            
            swc_file = f'{data_dir}/{mask_name}'

            converter = DendriteSWC()
            mask_tiffile_path = converter.generate_tif(swc_file, filename)

            masks = []
            with tifffile.TiffFile(mask_tiffile_path) as tif:
                for i in range(len(tif.pages)):
                    masks.append(np.expand_dims(tif.pages[i].asarray(), axis=0))

            self.masks = masks
        else:
            self.masks = None

    def __len__(self):
        return len(self.images), len(self.masks)

    def __getitem__(self, index):
        '''
        Pulls one item from images and masks
        '''
        image = self.images[index]
        image = (image - image.min()) / ((image.max() - image.min())  + 1e-5)
        image = torch.from_numpy(image[None, ...])

        if self.masks == None:
            return image, None, index

        mask = self.masks[index]
        mask = (mask - mask.min()) / ((mask.max() - mask.min()) + 1e-5)
        mask = torch.from_numpy(mask[None, ...])

        return image, mask, index
'''    
Un-comment them to test

if __name__=="__main__":
    dataset_params = [1, 1, True, True, "../datasets/DeepD3_Training/"]
    dataset = DatasetLoader(*dataset_params)
    for image, prompt_points, prompt_type, selected_component, idx in dataset:
        print(image.shape)
        print(selected_component.shape)
        print(prompt_points.shape)
        print(prompt_type.shape)
        break

    dataset_params = ['Dendrite_U.swc', "../datasets/DeepD3_Benchmark/DeepD3_Benchmark.tif"]
    dataset = TestDatasetLoader(*dataset_params)
    for image, label in dataset:
        print(image.shape)
        print(label.shape)
'''