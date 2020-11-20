# Flowers Image Classifier
Final project for the Udacity Nanodegree program: AI programming with Python. Image classifier to recognize different species of flowers.

The classifier works with a dataset of 102 flower categories. The dataset has to be split into three parts: training, validation, and testing. The dataset can be downloaed from here: http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html. The dataset can be placed in any directory path but this has to be passed in the trainer when calling it (see details below).

Code is written in Python and uses Pytorch for the training the neural network.

The code is organized in two parts:

1.- Trainer files that preprocess the images and train the image classifier on your dataset
2.- Predictor files that is able to predicts image content (provided that its class is one of the categories the classifier has been trained in)

The trainer trains a classifier for flower images using a pretrained CNN model and ad hoc classifier. Three CNN architectures are available for the user: VGG-16, DenseNet161, AlexNet. It saves the model as a checkpoint once trained. It prints out training loss, validation loss, and validation accuracy as the network trains.
Argparse module is used for passing the code the command line arguments provided by the user when running the program from a terminal window. If the user fails to provide some or all of the arguments, then the default values are used instead.

    Expected Call indicated with <> next: python train.py <directory with images> --save_dir <directory for checkpoint file> --arch <model> --learning_rate <value> --epochs <value> --dropout <value>
    
    Example call: python train.py flowers/ --arch VGG-16 --epoch 8 --gpu YES
    
The predictor makes predictions about a flower image type input by the user. It uses a pretrained network, which is input from the user and loaded CNN model and ad hoc classifier. Three CNN architectures are available for the user: VGG-16, DenseNet161, AlexNet. It returns the most K likely flower types (K as required from user). And it also identifies the name of the flower type using a mapping of categories to real names (cat_to_name.json file).
Argparse module is used for passing the code the command line arguments provided by the user when running the program from a terminal window. If the user fails to provide some or all of the arguments, then the default values are used instead.

    Expected Call indicated with <> next: python predict.py <image path> <checkpoint file path> --category_names <cat_to_name.json file>  --arch <model> --topk <value>
    
    Example call:python predict.py image_06054.jpg densenet_model_checkpoint.pth --topk 3 --arch densenet161
    
For using the network, first train it with any of the CNN options and a different set of hyperparameters by means of the train.py file as indicated before. Use of GPU is recomended for training VGG-16 and DenseNet161 CNNs. Then carry out the prediction of any flower image by calling the predict.py file as inidicated before. 

## How to install the code.
Just clone the repository in your working directory, open a terminal shell there and call train.py or predict.py
