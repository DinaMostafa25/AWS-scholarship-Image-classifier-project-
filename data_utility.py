import torch
from torchvision import models, datasets, transforms
def load_data(filepath):
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    # transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                                         ])
    val_test_transforms = transforms.Compose([transforms.Resize(225),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                                             ])

    #Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform =train_transforms)
    val_data = datasets.ImageFolder(valid_dir, transform =val_test_transforms)
    test_data = datasets.ImageFolder(test_dir, transform =val_test_transforms)

    #the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size = 32, shuffle = True)
    validloader = torch.utils.data.DataLoader(val_data, batch_size = 32 ) 
    testloader = torch.utils.data.DataLoader(test_data, batch_size = 32 )
    
    print('Data loaded ^^')
    return train_data, trainloader, validloader, testloader

def process_image(image):
    img = Image.open(image)
    val_test_transforms = transforms.Compose([transforms.Resize(225),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                                         ])
    res = val_test_transforms(img)
    return res 

    

    