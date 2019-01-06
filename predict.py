# Imports here
import sys
import json
import torch
from torch import nn
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# loads a checkpoint and rebuilds the model
def load_model(filepath):
    """
    Loads trained models from a checkpoint.
    """
    checkpoint = torch.load(filepath, map_location='cpu')
    if 'VGG_19' in filepath:
        model = models.vgg19(pretrained=True)
    elif 'DenseNet_161' in filepath:
        model = models.densenet161(pretrained=True)
    elif 'ResNet_152' in filepath:
        model = models.resnet152(pretrained=True)

    # Freezes parameteres
    for param in model.parameters():
        param.requires_grad = False
    
    # Modify Classifier
    try:
        n_inputs = getattr(model, model.last_layer_attr).in_features # Check if returns a Sequential
    except AttributeError:
        n_inputs = getattr(model, model.last_layer_attr)[-1].in_features
    attr = getattr(model, model.last_layer_attr)
    if isinstance(attr, nn.Sequential): # Check if returns a Sequential
        attr[-1] = nn.Linear(n_inputs, 102)
    else:
        attr = nn.Linear(n_inputs, 102)
    
    # Modifies last layer attributes
    setattr(model, model.last_layer_attr, attr)

    # Load other info
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.idx_to_class = {idx:class_ for class_, idx in model.class_to_idx.items()}
    model.name = checkpoint['name']
    model.optimizer_state  = checkpoint['optimizer_state']
    model.eval()
    
    return model

def load_cat_to_name(file_path):
    """
    Load file to convert categories is names.

    Args: 
        file_path: File directory.
    
    Outputs:
        cat_to_name: dictionary maping classes to names.
    """
    # Load file
    with open(file_path, 'r') as f:
        cat_to_name = json.load(f)
    
    return cat_to_name


def check_accuracy(model, validation_loader, train_on_gpu=False):
    """
    Check models overall accuracy in validation dataset.
    """
    # Validations step
    with torch.no_grad():
        correct = np.array([])
        for inputs, labels in validation_loader:
            # move tensors to GPU if CUDA is available
            if train_on_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()

            output = model(inputs)
            _, pred = torch.max(output, 1)    
            # compare predictions to true label
            correct_tensor = pred.eq(labels.data.view_as(pred))
            # Turns result in numpy array
            if train_on_gpu:
                results = np.squeeze(correct_tensor.cpu().numpy())
            else:
                results = np.squeeze(correct_tensor.numpy())
            correct = np.append(correct,
                                results)

   # Prints
    print(model.name + ":")
    print(f"Accuracy {correct.sum() / len(correct) * 100:.2f}% ({correct.sum()}/{len(correct)}). ")


def process_image(image):
    """ 
    Scales, crops, and normalizes a PIL image for a PyTorch model, returns an Numpy array.
    
    Args:
        image: PIL image.

        img: Numpy array.
    """
    # Resize
    size = (255, 255)
    image.thumbnail(size, Image.ANTIALIAS)
    
    # Crop center
    new_size = 224
    width, height = image.size  # Get dimensions
    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2
    image = image.crop((left, top, right, bottom))
    
    # Convert RGB
    image = image.convert("RGB")
    
    # Normalize
    np_image = np.array(image) / 255.
    std = np.array([.229, .224, .225])
    mean = np.array([.485, .456, .406])
    img = (np_image - mean) / std # un-normalize
    
    # Transpose
    img = img.transpose((2, 0, 1))

    return img

def imshow(image, ax=None):
    """
    Imshow for Tensor. Removes all processing from image.
    
    Args:
        image: PyTorch Tensor.
        ax= matplotlib axes.
    """
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


def predict(image_path, model, cat_to_name, topk=5, gpu=False):
    """
    Predict the class (or classes) of an image using a trained deep learning model.

    Args:
        image_path: image directory
        model: model used for prediction.
        cat_to_name: Categories to name of the classes.
        topk: number of categoies showed.
    """
    # Prepare image
    img = Image.open(image_path)
    np_img = process_image(img)
    tensor_img = torch.tensor(np_img).unsqueeze(0).double()
    
    # Predictions
    model.eval()    # garantes prediction mode
    model.double()  # Some windows specific errors
    with torch.no_grad():
        if gpu:
            # Move to GPU
            model.cuda()
            tensor_img.cuda()
        model_output = model(tensor_img)
        out = nn.functional.softmax(model_output, dim=1)
    
    # Results
    probs, idxs = out.topk(topk)
    idxs = idxs.numpy().squeeze(0)      # idxs as numpy array
    probs = probs.numpy().squeeze(0)    # probs as numpy array
    classes = [model.idx_to_class[i] for i in idxs]
    
    # Print predicted class name
    print(cat_to_name[classes[0]] + " : " + str(probs[0]))
    
    return list(probs), list(classes)


def show_probs(img_path, model, cat_to_name):
    """
    Show image and its probabilities.

    Args: 
        img_path: image directory
        model: model used for prediction.
        cat_to_name: Categories to name of the classes.
    """
    # Load image
    img = Image.open(img_path)
    np_img = process_image(img)
    tensor_data = torch.tensor(np_img)
    
    # Prediction
    probs, classes = predict(img_path, model, 5)
    names = [cat_to_name[c] for c in classes]
    
    # Plots
    fig, axs = plt.subplots(2, 1, figsize=(5, 12))
    axs[0].set_ticks=[]
    axs[0].set_axis_off()
    imshow(tensor_data, ax=axs[0])
    axs[1].barh(list(reversed(names)), list(reversed(probs)))


if __name__ == "__main__":

    # Get model architecture
    if len(sys.argv) > 1:
        model_arch = sys.argv[1]
    else:
        model_arch = 'best' # loads best architecture tested
    
    # Load trained Models - TOP 3 Architectures
    if 'resnet' in model_arch.lower():
        model = load_model('checkpoint_ResNet_152.pth')
    elif 'vgg' in model_arch.lower():
        model = load_model('checkpoint_VGG_19.pth')
    else: # default model
        model = load_model('checkpoint_DenseNet_161.pth') # Best model

    cat_to_name = load_cat_to_name('cat_to_name.json')

    # Print predictions of 'test.jpg' Image    
    print(predict('test.jpg', model, cat_to_name,5))

    # Show results
    show_probs('test.jpg', model, cat_to_name)

