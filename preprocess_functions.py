import os
import torch
import torchvision.datasets
import torchvision.transforms as transforms
from data.plankton_dataset import planktondataset, plankton_test_dataset

def check_key(d, key, valid_values):
    if not key in d:
        raise KeyError('Missing key {} in dictionnary {}'.format(key, d))
    if not d[key] in valid_values:
        raise ValueError("Key {}: got \"{}\" , expected one of {}".format(key, d[key], valid_values))


def validate_image_transform_params(image_transform_params: dict):
    """
    {'image_mode'='none'}
    {'image_mode'='shrink', output_image_size={'width':.., 'height': ..}}
    {'image_mode'='crop'  , output_image_size={'width':.., 'height': ..}}
    """
    check_key(image_transform_params, 'image_mode', ['none', 'shrink', 'crop'])

    if(image_transform_params['image_mode'] == 'none'):
        return
    else:
        assert('output_image_size' in image_transform_params)
        assert(type(image_transform_params['output_image_size']) is dict)
        assert('width' in image_transform_params['output_image_size'])
        assert('height' in image_transform_params['output_image_size'])


def make_image_transform(image_transform_params: dict,
                         transform: object):
    """
    image_transform_params :
        {'image_mode'='none'}
        {'image_mode'='shrink', output_image_size={'width':.., 'height': ..}}
        {'image_mode'='crop'  , output_image_size={'width':.., 'height': ..}}
    transform : a torchvision.transforms type of object
    """
    validate_image_transform_params(image_transform_params)

    resize_image = image_transform_params['image_mode']
    if resize_image == 'none':
        preprocess_image = None
    elif resize_image == 'shrink':
        preprocess_image = transforms.Resize((image_transform_params['output_image_size']['width'],
                                              image_transform_params['output_image_size']['height']))
    elif resize_image == 'crop':
        preprocess_image = transforms.CenterCrop((image_transform_params['output_image_size']['width'],
                                                  image_transform_params['output_image_size']['height']))

    if preprocess_image is not None:
        if transform is not None:
            image_transform = transforms.Compose([preprocess_image, transform])
        else:
            image_transform = preprocess_image
    else:
        image_transform = transform

    return image_transform


def make_trainval_dataset(dataset_dir: str,
                          image_transform_params: dict,
                          transform: object,
                          valid_ratio: float = 0.2):

    image_transform  = make_image_transform(image_transform_params, transform)

    dataset_train_and_valid = planktondataset(root=dataset_dir,
                                     transforms = image_transform)

    nb_train = int((1.0 - valid_ratio) * len(dataset_train_and_valid))
    nb_valid =  len(dataset_train_and_valid) - nb_train
    dataset_train, dataset_val = torch.utils.data.dataset.random_split(dataset_train_and_valid, [nb_train, nb_valid])

    return dataset_train, dataset_val


def make_test_dataset(dataset_dir: str,
                          image_transform_params: dict,
                          transform: object):

    image_transform  = make_image_transform(image_transform_params, transform)

    dataset_test = plankton_test_dataset(root=dataset_dir,
                                transforms = image_transform)

    return dataset_test

