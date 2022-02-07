# Plankton_classification_teamA001

## Requirements

Before running any code, make sure to install all the dependancies by running
```
pip3 install -r requirements.txt
```

## Train a model

The folowing command is training the best model that you propose with the training set images given in the path PATH_TO_TRAINING_SET
```
python3 main.py train PATH_TO_TRAINING_SET
```

All training parameters (image_transform, image_transform_params, batch_size, model, optimizer, f_loss, nb_epochs and training_description) can be defined at the beginning of the " if __name__=="__main__" " in main.py

## Train a model

The folowing command is loading the model whose checkpoint fullpath is PATH_TO_CHECKPOINT, testing the model on the images provided in PATH_TO_TEST_SET and outputting the label csv file ready for submission
```
python3 main.py test PATH_TO_CHECKPOINT PATH_TO_TEST_SET
```



