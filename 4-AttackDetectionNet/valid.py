import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

# VAlidation dataset

data_dir = '/home/leonato/Projects/deepfake-framework/4-AttackDetectionNet/data/test'
# classes = ['attacked!','notattacked!']
# classes = []
# print(classes)

test_transforms = transforms.Compose([transforms.Resize([704,704]),
                                      transforms.ToTensor(),
                                     ])

# test_transforms = transforms.Compose([transforms.Resize(224),
#                                       transforms.ToTensor(),
#                                      ])


# Testa GPU e carrega modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load('AttackNetModel.pth')
# model = torch.load('oldmodels/AttackNetModel-704x704-1epoch.pth')
model.eval()

def predict_image(image):
    image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = image_tensor
    # print(input)
    input = input.to(device)
    output = model(input)
    index = output.data.cpu().numpy().argmax()
    return index

def get_random_images(num):
    data = datasets.ImageFolder(data_dir, transform=test_transforms)
    print(data.classes)
    classes = data.classes
    indices = list(range(len(data)))
    np.random.shuffle(indices)
    idx = indices[:num]
    # print(idx)
    # print(data.imgs[idx[1]])
    from torch.utils.data.sampler import SubsetRandomSampler
    sampler = SubsetRandomSampler(idx)
    # print(sampler)
    loader = torch.utils.data.DataLoader(data, 
                   sampler=sampler, batch_size=num)
    dataiter = iter(loader)
    images, labels = dataiter.next()
    return images, labels, classes, data.imgs,idx


to_pil = transforms.ToPILImage()
images, labels, classes, dataimgs, idx = get_random_images(6)
fig=plt.figure(figsize=(10,10))
attacked = 0
notattacked = 0
for ii in range(len(images)):
    image = to_pil(images[ii])
    index = predict_image(image)
    sub = fig.add_subplot(1, len(images), ii+1)
    res = int(labels[ii]) == index
    sub.set_title(str(classes[index]) + ":" + str(res))
    # plt.axis('off')
#     plt.imshow(image)
# plt.show()
    prefix = '/home/leonato/Projects/deepfake-framework/4-AttackDetectionNet/data/test/'
    # print('img: '+ str(dataimgs[ii])[len(prefix):] + ' classification: ' + classes[index])
    print('img: '+ str(dataimgs[idx[ii]])[len(prefix)+1:] + ' classification: ' + classes[index])
    if index == 0:
        attacked += 1 
    else: 
        notattacked += 1
print('\nResults: \n Attacked: ' + str(attacked) + ' videos \n' + ' Not attacked: ' + str(notattacked)+' videos')