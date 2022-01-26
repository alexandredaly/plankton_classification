# External modules
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import pprint
# Local modules
import preprocess_functions
import utils

# The datasets is already downloaded on the cluster
dataset_dir = "train_try/"

# The post-processing of the image
image_transform = None #transforms.ToTensor()

img_idx = 2

# How do we preprocessing the image (e.g. none, crop, shrink)
image_transform_params = {'image_mode': 'none'}
train_dataset, valid_dataset = preprocess_functions.make_trainval_dataset(
    dataset_dir=dataset_dir,
    image_transform_params=image_transform_params,
    transform=image_transform)

img, target = train_dataset[img_idx]
print("The image from the dataset is of type {}".format(type(img)))

print("Saving an image as bird.jpeg")
# img.save('bird.jpeg')

img = np.asarray(img[0])
print(img.shape)

image_transform_params = {'image_mode': 'shrink',
                          'output_image_size': {'width': 224, 'height': 224}}
train_dataset, valid_dataset = preprocess_functions.make_trainval_dataset(
    dataset_dir=dataset_dir,
    image_transform_params=image_transform_params,
    transform=image_transform)
shrink_img = np.asarray(train_dataset[img_idx][0])

image_transform_params = {'image_mode': 'crop',
                          'output_image_size': {'width': 224, 'height': 224}}
train_dataset, valid_dataset = preprocess_functions.make_trainval_dataset(
    dataset_dir=dataset_dir,
    image_transform_params=image_transform_params,
    transform=image_transform)
crop_img = np.asarray(train_dataset[img_idx][0])


# Displaying an image
fig = plt.figure(figsize=(15, 5))
axes = fig.subplots(1, 3)
axes[0].imshow(img, aspect='equal')
axes[0].get_xaxis().set_visible(False)
axes[0].get_yaxis().set_visible(False)
axes[0].set_title('Original image')

axes[1].imshow(shrink_img, aspect='equal')
axes[1].get_xaxis().set_visible(False)
axes[1].get_yaxis().set_visible(False)
axes[1].set_title('Shrink')

axes[2].imshow(crop_img, aspect='equal')
axes[2].get_xaxis().set_visible(False)
axes[2].get_yaxis().set_visible(False)
axes[2].set_title('Crop')

plt.savefig('preprocess_images.png', bbox_inches='tight')
plt.show()
