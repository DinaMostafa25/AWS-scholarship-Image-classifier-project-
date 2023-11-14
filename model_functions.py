import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torchvision import models, datasets, transforms
def network (arch, hidden):
    print("Building network...........\nArchticture:{}\nHidden_unites{}".format(arch,hidden))
    model = models.densenet121(pretrained =  True)
    for param in model.parameters():
        param.requires_grad = False
    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([('fc1',nn.Linear(1024,500)),
                                            ('relu',nn.ReLU()),
                                            ('dropout',nn.Dropout(p=0.2)),
                                            ('fc2', nn.Linear(500,102)),
                                            ('output', nn.LogSoftmax(dim=1))
    ]))
    model.classifier = classifier
    print("\n\n\nBuilding is DONE ^^")
    return model
def train_network(model, epochs, lr, trainloader, validloader, GPU):
    print("Building network...........\nepoches:{}\nlearning_rate{}\n GPU used:{}".format(epochs,lr,GPU))
    
    if GPU:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
     
    criterion = nn.NLLLoss()
    
    optimizer = optim.Adam(model.classifier.parameters(), lr = 0.003)
    
    model.to(device)    
    
    #training model 
    steps = 0
    running_loss = 0
    print_every = 10
    train_losses, test_losses = [],[]
    
    for epoch in range(epochs):
        running_loss = 0
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            log_ps = model.forward(inputs)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if steps % print_every == 0:
                val_loss = 0 
                val_accuracy =  0
                model.eval()
                with torch.no_grad():
                    for inputs, targets in validloader:
                        inputs, targets = inputs.to('cuda'), targets.to('cuda')
                        outputs = model.forward(inputs)
                        batch_loss = criterion(outputs, targets)
                        val_loss += batch_loss.item()
                        #calcualts accuracy
                        ps = torch.exp(outputs)
                        top_p, top_class = ps.topk(1, dim =1)
                        eq = top_class== targets.view(*top_class.shape)
                        val_accuracy += torch.mean(eq.type(torch.FloatTensor)).item()
                train_losses.append(running_loss/len(trainloader))
                test_losses.append(val_loss/len(validloader))

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Test loss: {val_loss/len(validloader):.3f}.. "
                      f"Test accuracy: {val_accuracy/len(validloader):.3f}")
                running_loss =0
                model.train()
            
    print("\n\n\nTraining is DONE ^^")
    return model, criterion


def test_network(model, test_loader, criterion, GPU):
    print("Testing network...........\nGPU used:{}".format(GPU))
    
    if GPU:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
     
    # validation on the test set
    test_loss = 0 
    accuracy =  0
    test_losses=[]
    model.eval() 
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            outputs = model.forward(inputs)
            batch_loss = criterion(outputs, targets)
            test_loss += batch_loss.item()
            #calcualts accuracy
            ps = torch.exp(outputs)
            top_p, top_class = ps.topk(1, dim =1)
            eq = top_class== targets.view(*top_class.shape)
            accuracy += torch.mean(eq.type(torch.FloatTensor)).item()
    test_losses.append(test_loss/len(testloader))

    print(f"Test loss: {test_loss/len(testloader):.3f}.. "
          f"Test accuracy: {accuracy/len(testloader):.3f}")

    #model.train()
    print("\n\n\nTesting is DONE ^^")

def save_check_point(model, arch, hidden, epoches, lr, save_dir ):
    check_point = {
    'archticture':arch,
    'hidden_units': hidden,
    'epochs':epochs,
    'learning_rate':lr,
    'model_state_dict': model.state_dict(),
    'class_to_idx' : train_data.class_to_idx
    }
    check_point_path = save_dir+'checkpoint.pth'
    torch.save(check_point,'checkpoint.pth')
    
    print('Model saved ^^... in {}'.format(check_point_path))
    
def load_model(filepath):

    checkpoint = torch.load(filepath)
    model = network(check_point['archticture'],check_point['hidden_units'])
    model.load_state_dict(check_point['model_state_dict'])
    model.class_to_idx = check_point['class_to_idx']
    
    return model

def predict(processed_img, model, topk=5):
    model.eval()
    img = process_image(image_path).unsqueeze(0)
    with torch.no_grad():
        outputs = model.forward(img)
        ps = torch.exp(outputs)
        top_p, top_class = ps.topk(topk, dim=1)
        pred_classes = list()
        idx_to_class ={}
        for i in model.class_to_idx:
            idx_to_class[model.class_to_idx[i]] = i
        for _class in top_class.numpy()[0] :
            pred_classes.append(idx_to_class[_class])
    return top_p.numpy()[0], pred_classes
    


        
    

    
    