import torchvision.models
import torchvision.transforms as transforms
import torch.nn
import numpy as np
import pandas as pd
import preprocess_functions
from tqdm import tqdm

model_path = "logs/resnet50_finetuning_nonweightedloss_batch128_Adam_2/best_model.pt"
model = torchvision.models.resnet50(pretrained=True)
# for param in model.parameters():
#     param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
#We aadapt our model too 1-channel images by chaanging the input
model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
# Here's the classifying head of our model that we are going to train
model.fc = torch.nn.Linear(2048, 86)

device = torch.device('cuda')
model = model.to(device)

model.load_state_dict(torch.load(model_path))

# Switch to eval mode 
model.eval()

# The post-processing of the image
imagenet_preprocessing = transforms.Normalize(mean=[0.5],
                                            std=[0.5])
image_transform = transforms.Compose([transforms.ToTensor(), imagenet_preprocessing]) 

# Choose type of transformation to apply to training dataset
image_transform_params = {'image_mode': 'shrink',
                        'output_image_size': {'width': 224, 'height': 224}}

test_dataset = preprocess_functions.make_test_dataset(dataset_dir="../../Datasets/challenge_kaggle/test",
                                                    image_transform_params=image_transform_params,
                                                    transform=image_transform)

batch_size = 128
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                        batch_size=batch_size,
                                        shuffle=False,                # <-- this reshuffles the data at every epoch
                                        num_workers=4)

with torch.no_grad():
    df = None
    for i, (inputs, targets, names_images) in tqdm(enumerate(test_dataloader)):
        inputs = inputs.to(device)
        #targets = targets.to(device)

        # Compute the forward pass through the network up to the loss
        outputs = model(inputs)
        outputs = outputs.cpu()
        outputs = np.argmax(outputs, axis=1)

        if df is None:
            df = pd.DataFrame(data = {'imgname' : list(names_images), 'label' : outputs.numpy()})
        else:
            df_new = pd.DataFrame(data = {'imgname' : list(names_images), 'label' : outputs.numpy()})
            df = pd.concat([df, df_new])
    df.to_csv('predictions_final.csv', index=False)