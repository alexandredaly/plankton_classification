import sys
import os
import yaml
import numpy as np
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
import pandas as pd
# Local modules
from data.plankton_dataset import planktondataset
import utils
import preprocess_functions
import models


def main(image_transform, image_transform_params, batch_size, model, optimizer, f_loss, nb_epochs, training_description):
    """
    Main function to launch.
    train mode launches train() function --> stores best model at logs/<training_description>/best_model.pt
    test mode launches test() function --> generates csv prediction file

    Args:
        image_transform    -- torch.transforms Module
        image_transform_params    -- dict - Type of preprocessing to apply to images
        batch_size             -- int
        model                  -- A torch.nn object
        optimizer              -- A torch.optim.Optimzer object
        f_loss                 -- A loss Module
        nb_epochs              -- int
        training_description   -- str
    """

    try:
        if sys.argv[1] == 'train':
            # Launch best model training
            train(sys.argv[2], image_transform, image_transform_params, batch_size, model, optimizer, f_loss, nb_epochs, training_description)
        elif sys.argv[1] == 'test':
            # Launch best model testing
            test(sys.argv[2], sys.argv[3], image_transform_params, image_transform, batch_size, model)
        else:
            # The argument is not 'train' or 'test'
            print("Must indicate train or test")
    except IndexError:
        # No argument precised
        print("No arguments precised")


def train(dataset_dir, image_transform, image_transform_params, batch_size, model, optimizer, f_loss, nb_epochs, training_description):
    """
    Train a model for <nb_epochs> epochs, iterating over the loader
    using the f_loss to compute the loss and the optimizer
    to update the parameters of the model. At each epoch, if the score improves then the models parameters
    are saved at "logs/<training_description>/best_model.pt"

    Arguments :

        dataset_dir     -- dataset filepath
        image_transform    -- torch.transforms Module
        image_transform_params    -- dict - Type of preprocessing to apply to images
        batch_size             -- int
        model                  -- A torch.nn object
        optimizer              -- A torch.optim.Optimzer object
        f_loss                 -- A loss Module
        nb_epochs              -- int
        training_description   -- str

    Returns :
    """
    # Load dataset
    train_dataset, valid_dataset = preprocess_functions.make_trainval_dataset(
        dataset_dir=dataset_dir,
        image_transform_params=image_transform_params,
        transform=image_transform)

    num_threads = 4  # Loading the dataset is using 4 CPU threads

    #####################################
    # DATA AUGMENTATION
    ###################
    # weights = utils.get_weights(dataset_dir)
    # weights_sampler = utils.get_weights_sampler(train_dataset, weights)
    # weights_sampler = torch.FloatTensor(weights_sampler)
    # sampler = torch.utils.data.sampler.WeightedRandomSampler(weights_sampler, len(weights_sampler))
    #####################################

    # Create dataloaders
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,  # <-- this reshuffles the data at every epoch
                                                   #sampler = sampler,
                                                   num_workers=num_threads)

    valid_dataloader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=False,  # <-- this reshuffles the data at every epoch
                                                   num_workers=num_threads)

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = model.to(device)

    # Create the directory "./logs" if it does not exist
    top_logdir = os.path.join(os.getcwd(), "logs/")
    if not os.path.exists(top_logdir):
        os.mkdir(top_logdir)
    # Where to store the logs
    logdir = utils.generate_unique_logpath(
        top_logdir, training_description)
    print("Logging to {}".format(logdir))
    if not os.path.exists(logdir):
        os.mkdir(logdir)

    # Define the callback object
    model_checkpoint = utils.ModelCheckpoint(logdir + "/best_model.pt", model)

    # TRain over the givenn number of epochs
    for t in tqdm(range(nb_epochs)):
        print("Epoch {}".format(t))
        utils.train_epoch(model, train_dataloader, f_loss, optimizer, device)

        val_loss, val_acc, macro_f1 = utils.test_epoch(
            model, valid_dataloader, f_loss, device)
        print(" Validation : Loss : {:.4f}, Acc : {:.4f}, macro_f1 : {:.4F}".format(
            val_loss, val_acc, macro_f1))
        # If macro f1-score improves then we update the saved model
        model_checkpoint.update_on_score(macro_f1)


def test(path_to_checkpoint, test_dataset_dir, image_transform_params, image_transform, batch_size, model):
    """
    Test model with parameters saved at path_to_checkpoint on test dataset and generate csv prediction file

    Arguments :

        testdataset_dir     -- test dataset filepath (str)
        image_transform    -- torch.transforms Module
        image_transform_params    -- dict - Type of preprocessing to apply to images
        batch_size             -- int
        model                  -- A torch.nn object

    Returns :
    """
    # Make datasets
    test_dataset = preprocess_functions.make_test_dataset(dataset_dir=test_dataset_dir,
                                                          image_transform_params=image_transform_params,
                                                          transform=image_transform)

    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,                # <-- this reshuffles the data at every epoch
                                                  num_workers=4)

    device = torch.device('cuda')
    model = model.to(device)

    model.load_state_dict(torch.load(path_to_checkpoint))

    # Switch to eval mode
    model.eval()

    with torch.no_grad():
        # Initialize the dataframe of predictions
        df = None
        for i, (inputs, names_images) in tqdm(enumerate(test_dataloader)):
            inputs = inputs.to(device)

            # Compute the forward pass through the network up to the loss
            outputs = model(inputs)
            outputs = outputs.cpu()
            outputs = np.argmax(outputs, axis=1)

            # Complete the dataframe of predictions
            if df is None:
                df = pd.DataFrame(data={'imgname': list(
                    names_images), 'label': outputs.numpy()})
            else:
                df_new = pd.DataFrame(data={'imgname': list(
                    names_images), 'label': outputs.numpy()})
                df = pd.concat([df, df_new])
        # Save dataframe of predictions to csv
        df.to_csv('predictions_{}.csv'.format(path_to_checkpoint.split('/')[1]), index=False)


if __name__ == '__main__':
    batch_size = 128  # Using minibatches of 128 samples
    nb_epochs = 20 # We train our model over 20 epochs

    # Choose the post-processing of the image - normalization is required for using resnet
    imagenet_preprocessing = transforms.Normalize(mean=[0.5],
                                                  std=[0.5])
    image_transform = transforms.Compose(
        [transforms.ToTensor(), imagenet_preprocessing])

    # Choose type of transformation to apply to training dataset
    image_transform_params = {'image_mode': 'shrink',
                              'output_image_size': {'width': 224, 'height': 224}}
    #Choose model
    model = models.resnet50()

    # Instanciate the loss
    f_loss = utils.get_loss('crossentropy')

    # Instanciate optimizer
    optimizer = utils.get_optimizer('adam', model)

    # Location to save models 
    training_description = "resnet50_finetuning_nonweightedloss_batch128_Adam"

    main(image_transform, image_transform_params, batch_size, model, optimizer, f_loss, nb_epochs, training_description)
