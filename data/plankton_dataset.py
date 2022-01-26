import os
import glob
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
#from skimage import exposure
from PIL import Image

class planktondataset(Dataset):
    """Store the SAR data into a torch dataset like object. 

    Args:
        Dataset (class): pytorch dataset object 
    """

    def __init__(self, root, transforms=None):
        """
        Args:
            path (str): absolute path of the data files 
        """

        self.root = root
        self.files_paths = glob.glob(os.path.join(root, '*/*'))
        self.transforms = transforms


    def __getitem__(self, idx):
        """Retrieve the i-th item of the dataset

        Args:
            idx (int): idx-th item to retrieve

        Returns:
            image_input, image_target: the low resolution image and the high resolution image
        """
        #self.filepaths[idx] is in the form 'train/082_Aglaura/184.jpg'
        image_input = Image.open(self.files_paths[idx])
        target = self.files_paths[idx].split('/')[-2][1:3]
        #Possible solution to error ?
        # target = torch.zeros(86)
        # target[target_idx] = 1
        if self.transforms is None:
            return image_input, int(target)
        return self.transforms(image_input), int(target)

    def __len__(self):
        """Operator len that returns the size of the dataset 

        Returns:
            int: length of the dataset
        """
        return len(self.files_paths)


class plankton_test_dataset(Dataset):
    """Store the SAR data into a torch dataset like object. 

    Args:
        Dataset (class): pytorch dataset object 
    """

    def __init__(self, root, transforms=None):
        """
        Args:
            path (str): absolute path of the data files 
        """

        self.root = root
        self.files_paths = glob.glob(os.path.join(root, '*/*'))
        self.transforms = transforms


    def __getitem__(self, idx):
        """Retrieve the i-th item of the dataset

        Args:
            idx (int): idx-th item to retrieve

        Returns:
            image_input, image_target: the low resolution image and the high resolution image
        """
        #self.filepaths[idx] is in the form 'train/082_Aglaura/184.jpg'
        image_input = Image.open(self.files_paths[idx])
        target = self.files_paths[idx].split('/')[-2][1:3]
        name_image = self.files_paths[idx].split('/')[-1]
        #Possible solution to error ?
        # target = torch.zeros(86)
        # target[target_idx] = 1
        if self.transforms is None:
            return image_input, target, name_image
        return self.transforms(image_input), target, name_image

    def __len__(self):
        """Operator len that returns the size of the dataset 

        Returns:
            int: length of the dataset
        """
        return len(self.files_paths)