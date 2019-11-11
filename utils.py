import argparse
import json
import torch
import torchvision

from collections import OrderedDict
from PIL import Image
from torch import nn, optim
from torchvision import datasets, transforms, models


supported_arches = ['vgg11', 'vgg13', 'vgg16', 'vgg19']


# Parse command line parameters for predict.py
def get_predict_args():
    """
    Parses command line arguments provided by the user
    
    Basic usage:
        python predict.py /path/to/image checkpoint
    
    Options:
        python predict.py input checkpoint --top_k 3
        python predict.py input checkpoint --category_names cat_to_name.json
        python predict.py input checkpoint --gpu
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument('image_path', type = str, default = './flowers/test/10/image_07117.jpg',
                        help = 'image file to predict')
    
    parser.add_argument('checkpoint_path', type = str, default = 'checkpoint.pth',
                        help = 'file path to load model checkpoints')     
    
    parser.add_argument('--top_k', type = int, default = '3',
                        help ='returns the top numbers of predicted classses')
    
    parser.add_argument('--category_names', type = str, default = './cat_to_name.json',
                        help ='file path of category to name')
    
    parser.add_argument('--gpu', type = bool, default = True,
                        help = 'true to enable gpu mode')
    
    return parser.parse_args() 


def get_labels(in_args):
    category_names_file = in_args.category_names
    print('Getting category names from category_names={}'.format(category_names_file))
    
    with open(category_names_file, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name


def predict(model, in_args):
    probs, classes = predict_class(model, in_args)
    cat_to_name = get_labels(in_args)
    
    labels = []
    for c in classes:
        labels.append(cat_to_name[c])
        
    for i in range(in_args.top_k):
        print('top {} predict: label = {}, probability = {}'.format(i+1, labels[i], probs[i]))
    
    return None


def predict_class(model, in_args):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    ''' 
    topk = in_args.top_k
    use_gpu = in_args.gpu
    image_path = in_args.image_path
    
    print('Predicting image image_path={}, topk={}, gpu={}'.format(image_path, topk, use_gpu))
    
    # Process image
    img = process_image(in_args)
    img = img.unsqueeze(0)  
    
    # Use GPU if it's available
    device = get_device(in_args)
    img = img.to(device)   
    model.to(device)
    
    model.eval()    

    # Calculate the class probabilities (softmax) for img
    with torch.no_grad():
        output = model.forward(img)

        ps = torch.exp(output)
        top_p, top_class = ps.topk(topk)

        classes = []       
        for key, value in model.class_to_idx.items():
            for idx in top_class[0].tolist():
                if idx == value:
                    classes.append(key)
                    
    return top_p[0].tolist(), classes


# process image
def process_image(in_args):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        return an image tensor
    '''
    image_path = in_args.image_path
    print('Processing image from image_path={}'.format(image_path))
    
    image = Image.open(image_path)
    
    image_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])
    
    image_tensor = image_transforms(image)
    
    return image_tensor 


# load model from saved checkpoint
def load_checkpoint(in_args):
    save_dir = in_args.checkpoint_path
    
    print('Loading network ... save_dir={}'.format(save_dir))
    
    checkpoint = torch.load(save_dir)
    arch = checkpoint['arch']
    
    model = getattr(models, arch)(pretrained=True)
    
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']    
    model.load_state_dict(checkpoint['state_dict'])
    
    print('Model loaded from checkpoint. model={}'.format(model))
    
    return model


# Save the model
def save_checkpoint(model, in_args, image_datasets):
    save_dir = in_args.save_dir
    print('Saving network ... save_dir={}'.format(save_dir))
    
    device = get_device(in_args)
    model.to(device)    
    
    model.class_to_idx = image_datasets['train'].class_to_idx
    
    checkpoint = {'arch': in_args.arch,
              'class_to_idx': model.class_to_idx,
              'classifier' : model.classifier,
              'state_dict': model.state_dict()}
    
    torch.save(checkpoint, 'checkpoint.pth')
    print('Network saved. model={}'.format(model))
    
    return None


# Test the model
def test_network(model, dataloaders, in_args):
    print('Testing network ... use_gpu={}'.format(in_args.gpu))
    
    test_loss = 0
    accuracy = 0
    
    device = get_device(in_args)
    criterion = nn.NLLLoss()
    
    # Validation mode
    model.eval()
    
    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs, labels = inputs.to(device), labels.to(device)

            log_ps = model.forward(inputs)
            loss = criterion(log_ps, labels)
            test_loss += loss.item()

            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print("Test loss: {:.3f} ...".format(test_loss/len(dataloaders['test'])),
          "Test accuracy: {:.3f} ".format(accuracy/len(dataloaders['test'])))

    return None


def get_device(in_args):
    return torch.device("cuda" if torch.cuda.is_available() and in_args.gpu else "cpu")


# Train the model
def train_network(in_args, model, dataloaders):
    learning_rate = in_args.learning_rate
    epochs = in_args.epochs
    use_gpu = in_args.gpu
    print('Training network ... learning_rate={}, epochs={}, gpu={}'.format(learning_rate, epochs, use_gpu))
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    # Use GPU if it's available
    device = get_device(in_args)
    model.to(device)
    
    validate_every = 50
    step = 0
    running_loss = 0 

    for epoch in range(epochs):
        for inputs, labels in dataloaders['train']:
            step += 1
            
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
        
            log_ps = model.forward(inputs)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Validates
            if step % validate_every == 0:
                validation_loss = 0
                accuracy = 0
                model.eval()

                with torch.no_grad():
                    for inputs, labels in dataloaders['validation']:
                        inputs, labels = inputs.to(device), labels.to(device)

                        log_ps = model.forward(inputs)
                        loss = criterion(log_ps, labels)
                        validation_loss += loss.item()

                        ps = torch.exp(log_ps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()


                print("Epoch {}/{} ... ".format(epoch+1, epochs),
                      "Step {} ... ".format(step),
                      "Train loss: {:.3f} ... ".format(running_loss/validate_every),
                      "Validation loss: {:.3f} ... ".format(validation_loss/len(dataloaders['validation'])),
                      "Validation accuracy: {:.3f} ".format(accuracy/len(dataloaders['validation'])))

                model.train()
                running_loss = 0
    
    print('Model trained. ')
    
    return None


# Create network
def create_network(in_args):
    arch = in_args.arch
    hidden_units = in_args.hidden_units
    print('Creating network ... arch={}, hidden_units={}'.format(arch, hidden_units))

    if arch not in supported_arches:
        print('Unsupported architect, please choose from {}'.format(supported_arches))
        exit(1)
    
    model = getattr(models, arch)(pretrained=True)
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
        
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, hidden_units)),
        ('relu', nn.ReLU()),
        ('drop', nn.Dropout(0.3)),
        ('fc2', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    
    model.classifier = classifier
    
    print('Network created, model={}'.format(model))
   
    return model
    

# Load data
def load_data(in_args):
    data_dir = in_args.data_directory    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    print('Loading data ... data_dir={}'.format(data_dir))
    
    # transforms
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])
    
    validation_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])

    test_transforms = validation_transforms

    data_transforms = {'train': train_transforms,
                  'validation': validation_transforms,
                  'test': test_transforms}
    
    # Load the dataset
    train_datasets = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
    validation_datasets = datasets.ImageFolder(valid_dir, transform=data_transforms['validation'])
    test_datasets = datasets.ImageFolder(test_dir, transform=data_transforms['test'])
    
    image_datasets = {'train': train_datasets, 
                  'validation': validation_datasets, 
                  'test': test_datasets}
    
    # Define dataloaders
    train_loaders = torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True)
    validation_loaders = torch.utils.data.DataLoader(image_datasets['validation'], batch_size=64)
    test_loaders = torch.utils.data.DataLoader(image_datasets['test'], batch_size=64)

    dataloaders = {'train': train_loaders, 
               'validation': validation_loaders, 
               'test': test_loaders}
    
    return dataloaders, image_datasets


# Parse command line parameters
def get_train_args():
    """
    Parses command line arguments provided by the user
    
    Basic usage:
        python train.py data_directory
    
    Options:
        python train.py data_dir --save_dir save_directory
        python train.py data_dir --arch "vgg13"
        python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
        python train.py data_dir --gpu
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument('data_directory', type = str, default = './flowers/',
                        help='path containing data set')
    
    parser.add_argument('--save_dir', type = str, default = 'checkpoint.pth',
                        help = 'directory to save checkpoints')    
    
    parser.add_argument('--arch', type = str, default = 'vgg19',
                        help = 'pre-trained model, choose from supported architects {}'.format(supported_arches))   
    
    parser.add_argument('--learning_rate', type = float, default = 0.001,
                        help = 'learning rate for gradient decent')
    
    parser.add_argument('--hidden_units', type = int, default = 4096,
                        help = 'number of unit in the neutral network hidden layer')
    
    parser.add_argument('--epochs', type = int, default = 4,
                        help = 'number of interation in training')
    
    parser.add_argument('--gpu', type = bool, default = True,
                        help = 'true to enable gpu mode')   
    
    return parser.parse_args()