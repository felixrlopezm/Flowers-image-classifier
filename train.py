# python3
#
# PROGRAMMER: Félix Ramón López Martínez
# DATE CREATED: 18/11/2020
# REVISED DATE:
# PURPOSE: This code trains a classifier for flower images using a pretrained
#          CNN model and ad hoc classifier.
#          Three CNN architectures are available for the user: VGG-16,
#          DenseNet161, AlexNet
#          It saves the model as a checkpoint once trained.
#          It prints out training loss, validation loss, and validation accuracy
#          as the network trains.
#
# User argparse Expected Call indicated with <> next:
#      python train.py <directory with images> --save_dir
#      <directory for checkpoint file> --arch <model> --learning_rate <value>
#      --epochs <value> --dropout <value>
#   Example call:
#    python train.py flowers/ --arch VGG-16 --epoch 8 --gpu YES
##

# Imports python modules
from torchvision import datasets, transforms, models
import torch
from torch import nn, optim
from time import time
#from PIL import Image
#import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Imports functions created for this program
from train_functions import get_input_args
from train_functions import load_pretrained_model
from train_functions import load_classifier
from train_functions import save_checkpoint

def main():
    # Getting input arguments by user when running the program from terminal
    in_arg = get_input_args()

    # Defining directory paths
    data_dir = in_arg.data_directory
    training_dir = data_dir + '/train'
    validation_dir = data_dir + '/valid'

    # Defining transforms for the training and validation sets
    training_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                              transforms.RandomHorizontalFlip(p=0.33),
                                              transforms.RandomVerticalFlip(p=0.33),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])

    # Loading the datasets with ImageFolder for the training and validation sets
    training_dataset = datasets.ImageFolder(training_dir, transform = training_transforms)
    validation_dataset = datasets.ImageFolder(validation_dir, transform = validation_transforms)

    #  Defining the training and validation dataloaders
    batch_size = 50
    training_loader = torch.utils.data.DataLoader(training_dataset, batch_size, shuffle = True)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size, shuffle = True)

    ## BUILDING THE NETWORK
    # Load pre-trained model acc. to --arch parameter
    model = load_pretrained_model(in_arg.arch.lower())
    print('state_keys 1', model.state_dict().keys())

    # Freezing model parameters to not backpropagate through them
    for param in model.parameters():
        param.requires_grad = False

    # Loading and substituing the odel classifier acc. to -arch parameter
    model.classifier = load_classifier(in_arg.arch.lower(), in_arg.dropout)
    print('state_keys 2', model.state_dict().keys())

    ## TRAINING THE NETWORK
    # Use GPU if it's available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Assigning hyperparameters
    lr = in_arg.learning_rate
    epochs = in_arg.epochs

    # Definition of the error function: negative log likelihood loss
    criterion = nn.NLLLoss()

    # Definition of the optimizer algorithm: Adam algorithm
    optimizer = optim.Adam(model.classifier.parameters(), lr)

    # Sending the model to the existing device
    model.to(device);

    # Start time counter just before starting training
    start = time()

    training_losses, validation_losses, accuracy_log = [], [], []

    # Setting model in training mode
    model.train()

    print('\nSTART TRAINING')

    # Training loop
    for e in range(epochs):
        running_loss = 0
        step = 0
        start_epoch = time()

        for images, labels in training_loader:
            step += 1
            # Move input and label tensors to the default device
            images, labels = images.to(device), labels.to(device)

            # Clearing the gradients every training loop
            optimizer.zero_grad()

            # Forward pass
            log_ps = model(images)

            # Loss calculation
            loss = criterion(log_ps, labels)

            #Backpropagration and Gradient Descent
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Print progress
            if step % 30 == 0:
                print('------------------------------------------------------------------')
                print('Epoch: {}/{}.. '.format(e+1, epochs),
                      'Step: {}/{}'.format(step, len(training_loader)),
                      'Training Loss: {:.3f}.. '.format(running_loss/len(training_loader)))
                print(f'Total time per step: {((time() - start_epoch) / step):.3f} seconds')
                print(f'Total time: {(time() - start):.3f} seconds')

        else:
            # Validation pass
            validation_loss = 0
            accuracy = 0

            # Setting model to evaluation mode
            model.eval()

            # Turn off gradients for validation
            with torch.no_grad():
                for images, labels in validation_loader:
                    # Move input and label tensors to the default device
                    images, labels = images.to(device), labels.to(device)

                    # Forward pass
                    log_ps = model(images)

                    # Validation loss calculation
                    validation_loss += criterion(log_ps, labels)

                    # Calculating the accuracy during the validation pass
                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                # set model back to train mode
                model.train()

                training_losses.append(running_loss / len(training_loader))
                validation_losses.append(validation_loss.item() / len(validation_loader))
                accuracy_log.append(accuracy / len(validation_loader) * 100)

                # Print out the validation accuracy
                print('===================================================================================')
                print('Epoch: {}/{}.. '.format(e+1, epochs),
                      'Training Loss: {:.3f}.. '.format(running_loss / len(training_loader)),
                      'Validation Loss: {:.3f}.. '.format(validation_loss.item() / len(validation_loader)),
                      'Validation Accuracy: {:.1f}%'.format(accuracy / len(validation_loader) * 100))

                start_epoch = time()

    # Total time of the training process
    print(f'Total time per epoch: {((time() - start) / epochs):.3f} seconds')
    print(f'Total training time: {(time() - start):.3f} seconds\n')

    # Saving model's state_state_dict and its architecture in checkpoint file
    model.to('cpu')
    save_checkpoint(in_arg.arch.lower(), in_arg.dropout, training_dataset.class_to_idx,
                    model.state_dict())

    # Printing a summary of training and validation losses and accuracy
    epoch_labels = ['epoch_{}'.format(x) for x in range(1,epochs+1)]
    accuracy_log = ('{:.1f}%'.format(x) for x in accuracy_log)

    summary_dict = {'Training loss': training_losses,
               'Validation loss': validation_loss.item(),
               'Validation Accuracy': accuracy_log}
    summary = pd.DataFrame(summary_dict, index=epoch_labels)
    print(summary)

    print('\nEND')

# Call to main function to run the program
if __name__ == "__main__":
    main()
