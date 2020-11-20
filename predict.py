# python3
#
# PROGRAMMER: Félix Ramón López Martínez
# DATE CREATED: 19/11/2020
# REVISED DATE:
# PURPOSE: This code make predictions about a flower image type.
#          It uses a pretrained network, which is input from the user and loaded
#          CNN model and ad hoc classifier.
#          Three CNN architectures are available for the user: VGG-16,
#          DenseNet161, AlexNet
#          It returns the most K likely flower types (K as required from user)
#          And it also identifies the name of the flower type using a mapping
#          of categories to real names.
#
# User argparse Expected Call indicated with <> next:
#      python predict.py <image path> <checkpoint file path>
#      --category_names <cat_to_name.json file>  --arch <model>  --topk <value>
#   Example call:
#    python predict.py image_06054.jpg alexnet_model_checkpoint.pth
#    --topk 3 --arch alexnet
##

# Imports python modules
import json

# Imports functions created for this program
from predict_functions import get_input_args
from predict_functions import process_image
from predict_functions import load_checkpoint_file
from predict_functions import predict
from predict_functions import plotting

# Main program function
def main():

    # Getting input arguments by user when running the program from terminal
    in_arg = get_input_args()
    print('Arguments in')

    # Preprocessing image format before prediction
    img = process_image(in_arg.image_path)
    print('Image processed')

    # Load checkpoint file and rebuilding model
    model = load_checkpoint_file(in_arg.checkpoint_file, in_arg.arch)
    print('Rebuilt model and reloaded model state')

    # Predicting top K probabilities and their corresponding classes
    ps_topk, cat_topk = predict(img, model, in_arg.topk)
    print('Prediction done')

    # Creating a mapping dictionary between image category and its name
    with open(in_arg.category_names, 'r') as f:
        cat_to_name_dict = json.load(f)

    # Transforming image categories to labels for plotting
    labels = [cat_to_name_dict[clas] for clas in cat_topk]
    ps_topk_formatted = ["%.1f" % (ps * 100) for ps in ps_topk]

    # Printing prediction
    print('Top {} predicted categories and probabilities'.format(in_arg.topk))
    print('    Categories: {}'.format(labels))
    print('    Probabilities, %: ', ps_topk_formatted)

    print('END')

# Call to main function to run the program
if __name__ == "__main__":
    main()
