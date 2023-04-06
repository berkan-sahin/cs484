from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision
from torchvision import models, transforms
from torchvision.io import read_image, ImageReadMode
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import KFold

class HockeyDataset(Dataset):

    def __init__(self, data_dir, transformer) -> None:
        self.data_dir = data_dir
        self.transformer = transformer


    def __len__(self):
        return 192

    def __getitem__(self, index) -> Tuple[any, any]:
        class_idx = index / 64
        if (class_idx < 1):
            class_name = 'freehit'
        elif (class_idx < 2):
            class_name = 'goal'
        elif (class_idx < 3):
            class_name = 'penaltycorner'
        else:
            class_name = 'penaltyshot' # should never happen

        img_name = os.path.join(self.data_dir, class_name, f'{(index % 64) + 1}.jpg')
        image = read_image(img_name).to(torch.float32)
        image = self.transformer(image)
        return image, class_idx

        
        


preprocess = transforms.Compose([
    #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.Resize(256, antialias=True),
    transforms.CenterCrop(224),
    #transforms.ToTensor(),
])


if __name__ == '__main__':
    fold = 6
    epochs = 10
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)
    vgg16 = models.vgg16()
    vgg16.load_state_dict(torch.load('vgg16.pth'))
    dataset = HockeyDataset('dataset', preprocess)
    kfold = KFold(n_splits=fold, shuffle=True)

    for (fold, (train, test)) in enumerate(kfold.split(dataset)):
        print("Train: ", train, "Validation: ", test)
        trainloader = DataLoader(dataset, batch_size=4, sampler=train, num_workers=4)
        valloader = DataLoader(dataset, batch_size=4, sampler=test, num_workers=4)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(vgg16.parameters(), lr=0.001)
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                optimizer.zero_grad()
                outputs = vgg16(inputs)
                loss = criterion(outputs, labels.long())
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if i % 10 == 9:
                    print('Fold %d [%d, %5d] loss: %.3f' % (fold ,epoch + 1, i + 1, running_loss / 10))
                    running_loss = 0.0

            correct = 0
            total = 0
            with torch.no_grad():
                for data in valloader:
                    images, labels = data
                    outputs = vgg16(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            print('Accuracy of the network on the 64 test images: %d %%' % (100 * correct / total))
        print('Finished Training')


    
    



    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(vgg16.parameters(), lr=0.001)
    features, labels = next(iter(dataloader))
    print(features.shape)
    print(labels.shape)
    img = torchvision.utils.make_grid(features)
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()
"""

