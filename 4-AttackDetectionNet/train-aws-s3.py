# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

# Dataset
data_dir = '/root/deepfake-framework/4-AttackDetectionNet/data/train'
data_s3 = 's3://raw-videos-gleonato/AttackDetectionNet/data/train/'

# image input
frame_height=704
frame_weight=704

# Model parms
epochs = 6
batch_size = 64
steps = 0
running_loss = 0
print_every = 1
train_losses, test_losses = [], []


# Carrega os datasets de treinamento 
def load_split_train_test(datadir, valid_size = .2):
    # Transforma resize dos frames
    train_transforms = transforms.Compose([transforms.Resize([frame_height,frame_weight]),
                                       transforms.ToTensor(),
                                       ])
    test_transforms = transforms.Compose([transforms.Resize([frame_height,frame_weight]),
                                      transforms.ToTensor(),
                                         ])
    train_data = datasets.ImageFolder(datadir,transform=train_transforms)
    test_data = datasets.ImageFolder(datadir,transform=test_transforms)
    num_train = len(train_data)
    # print('Input shape:' + train_data[0][0].shape)
    indices = list(range(num_train))

    split = int(np.floor(valid_size * num_train))
    # print(split)

    np.random.shuffle(indices)
    # print(indices)

    from torch.utils.data.sampler import SubsetRandomSampler
    train_idx, test_idx = indices[split:], indices[:split]
    print(len(train_idx), len(test_idx))
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    trainloader = torch.utils.data.DataLoader(train_data,
                   sampler=train_sampler, batch_size=batch_size)
    testloader = torch.utils.data.DataLoader(test_data,
                   sampler=test_sampler, batch_size=batch_size)
    return trainloader, testloader

trainloader, testloader = load_split_train_test(data_s3, .2)
print(trainloader.dataset.classes)
# print(testloader.dataset)

# Testa para GPU e Carrega Resnet50 

device = torch.device("cuda" if torch.cuda.is_available() 
                                  else "cpu")
print(device)
model = models.resnet50(pretrained=True)
# print(model)

#  freeze Resnet50 pre-trained layers, so we don’t backprop through them during training. 
for param in model.parameters():
    param.requires_grad = False

# Then, we re-define the final fully-connected the layer, the one that we’ll train with our images. We also create the criterion (the loss function) and pick an optimizer (Adam in this case) and learning rate.

model.fc = nn.Sequential(nn.Linear(2048, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, 10),
                                 nn.LogSoftmax(dim=1))
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.003)
model.to(device)
# print(model)


# Train the model
for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        writer.add_scalar("Loss/train", loss, epoch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    writer.add_scalar("Loss/train", batch_loss, epoch)
                    test_loss += batch_loss.item()
                    
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            train_losses.append(running_loss/len(trainloader))
            test_losses.append(test_loss/len(testloader))                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(testloader):.3f}.. "
                  f"Test accuracy: {accuracy/len(testloader):.3f}")
            running_loss = 0
            model.train()
            writer.flush()
    torch.save(model, 'AttackNetModel.pth-{}-{}'.format(epoch,os.timestamp))
# torch.save(model, 'AttackNetModel.pth')
