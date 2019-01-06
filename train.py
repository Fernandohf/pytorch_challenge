
# Imports here
import sys
import os
import torch
from torch import nn
from torch import optim
import torchvision
from torchvision import transforms, models
import matplotlib.pyplot as plt
import numpy as np

def check_gpu():
    """
    Return if GPU is available or not.
    """
    # check if CUDA is available
    train_on_gpu = torch.cuda.is_available()

    if not train_on_gpu:
        print('CUDA is not available.  Training on CPU ...')
    else:
        print('CUDA is available!  Training on GPU ...')
    
    return train_on_gpu


# ## Load the data

def get_dataloaders(folder = 'flower_data', batch_size=32):
    """
    Return train and test dataloader and the class to idx dictionary.
    All transforms are enabled.
    
    Args:
        folder: String with the folder name that contains the train and test folders.
    """
    # System agnostic directories
    local_dir = os.getcwd()
    data_dir = os.path.join(local_dir, folder)
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')


    # Transforms for the training and validation sets
    data_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                        transforms.RandomRotation(10),
                                        transforms.Resize(255),
                                        transforms.RandomCrop(224),
                                        #transforms.RandomResizedCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize((.485, .456, .406), (.229, .224, .225))
                                        ])

    #Load the datasets with ImageFolder
    train_data = torchvision.datasets.ImageFolder(train_dir, transform=data_transforms)
    valid_data = torchvision.datasets.ImageFolder(valid_dir, transform=data_transforms)

    # print out some data stats
    print('Num training images: ', len(train_data))
    print('Num test images: ', len(valid_data))

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(train_data,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=0) 
    valid_loader = torch.utils.data.DataLoader(valid_data,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=0)

    return train_loader, valid_loader, train_data.class_to_idx


# helper function to un-normalize and display an image
def imshow(img):
    """
    Un-normalize and display numpy image
    """
    std = np.array([.229, .224, .225]).reshape(3,1,1)
    mean = np.array([.485, .456, .406]).reshape(3,1,1)
    img = img * std + mean      # un-normalize
    img = np.clip(img, 0, 1)
    plt.imshow(np.transpose(img, (1, 2, 0)))  


def prepare_model(model, n_out):
    """
    Prepares the model for training.
    
    Args: 
        model: A model from torchvision.models
    """
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
        attr[-1] = nn.Linear(n_inputs, n_out)
    else:
        attr = nn.Linear(n_inputs, n_out)
    
    # Modifies last layer attributes
    setattr(model, model.last_layer_attr, attr)


def train_model(model, loaders, train_on_gpu=False, optimizer='sgd', criterion='crossentropy', lr=0.001, n_epochs=2, print_every=100):
    """
    Trains and validates a previously prepared model.

    Args:
        model: The model ready to be trained.
        loaders: Dictionary with the keys 'train' and 'test' for training and validation data.
        train_on_gpu: Either training on GPU or not.
        optimizer: the optimizer used (SGD or ADAM).
        criterion: the loss function used (NLLLoss of CrossEntropy).
        lr: Learning Rate
        n_epochs: Number of Epochs.
        print_every: Show training / validation loss every print_every batches.
    """
    # Initial print
    print('#'*len(model.name))
    print(model.name.upper())
    print('#'*len(model.name))

    # Move model to gpu
    if train_on_gpu:
        model.cuda()
    
    # Specify loss function (categorical cross-entropy)
    if criterion == 'logits':
        criterion = nn.NLLLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    
    # specify optimizer (stochastic gradient descent) and learning rate = 0.001
    if optimizer == 'SGD':
        optimizer = optim.SGD(getattr(model, model.last_layer_attr).parameters(), lr=lr)
    else:
        optimizer = optim.Adam(getattr(model, model.last_layer_attr).parameters(), lr=lr)

    # Check if is an already trained model
    if hasattr(model, 'optimizer_state'):
        optimizer.load_state_dict(model.optimizer_state)

    # number of epochs to train the model
    for epoch in range(1, n_epochs + 1):

        # keep track of training and validation loss
        train_loss = 0.

        # model by default is set to train
        model.train()
        for batch_i, (data, target) in enumerate(loaders['train']):
            # move tensors to GPU if CUDA is available
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
            
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            
            # calculate the batch loss
            loss = criterion(output, target)
            
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            
            # perform a single optimization step (parameter update)
            optimizer.step()
            
            # update training loss 
            train_loss += loss.item()
            
            # Show partial results
            if (batch_i + 1) % print_every == 0:
                test_loss = 0
                model.eval()
                correct = np.array([])
                # Validations step
                with torch.no_grad():
                    for inputs, labels in loaders['test']:
                        # move tensors to GPU if CUDA is available
                        if train_on_gpu:
                            inputs, labels = inputs.cuda(), labels.cuda()
                        
                        output_val = model(inputs)
                        val_loss = criterion(output_val, labels)
                        test_loss += val_loss.item()

                        # Predicted index
                        _, pred = torch.max(output_val, 1)    
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
                print(f"Epoch {epoch}/{n_epochs}.. "
                      f"Train loss: {train_loss/(print_every * data.size()[0]):.3f}.. "
                      f"Test loss: {test_loss/len(loaders['test']):.3f}.. "
                      f"Accuracy: {correct.sum()/len(correct) * 100:.2f} % ({correct.sum()}/{len(correct)}). ")
                
                # Make sure model is in training mode
                model.train()
    
    # Saves training state
    model.optimizer_state = optimizer.state_dict
    model.class_to_idx = loaders['class_to_idx']


def save_model(model, file_name="checkpoint"):
    """
    Saves the given model with all the needed info.

    Args: 
        model: Model to be saved.
        file_name: initial file name of the saved model.
    """
    # Saves model 
    checkpoint = {'name': model.name,
                  'class_to_idx': model.class_to_idx,
                  'optimizer_state': model.optimizer_state,
                  'state_dict': model.state_dict()}

    torch.save(checkpoint, file_name + '_' + model.name + '.pth')


# Prepares and perform the training of te best model.                  
if __name__ == "__main__":

    # Get model architecture
    if len(sys.argv) > 1:
        model_arch = sys.argv[1]
    else:
        model_arch = 'best' # loads best architecture tested
    
    # Load Pretrained Models - TOP 3 Architectures
    if 'resnet' in model_arch.lower():
        model = models.resnet152(pretrained=True)
        # Add attributes to model
        model.name = 'ResNet_152'
        model.last_layer_attr = 'fc'
    elif 'vgg' in model_arch.lower():
        model = models.vgg19(pretrained=True)
        # Add attributes to model
        model.name = 'VGG_19'
        model.last_layer_attr = 'classifier'
    
    else: # default model
        model = models.densenet161(pretrained=True) # Best model
        # Add attributes to model
        model.name = 'DenseNet_161'
        model.last_layer_attr = 'classifier'
    
    # Data Loaders
    train_loader, valid_loader, class_to_idx = get_dataloaders('flower_data')
    loaders = {'train': train_loader, 'test': valid_loader, 'class_to_idx': class_to_idx}

    # Prepare Models
    prepare_model(model, 102)

    # Check GPU if available
    gpu_status = check_gpu()

    # Train model
    train_model(model, loaders, train_on_gpu=gpu_status,
                optimizer='cross_entropy', criterion='adam',
                lr=0.001, n_epochs=100, print_every=100)

    # Save model
    save_model(model)
