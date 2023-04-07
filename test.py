from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import torchvision
from torchvision import models, transforms
from torchvision.io import read_image, ImageReadMode
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import KFold, train_test_split
import time
import copy


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
            class_name = 'penaltyshot'  # should never happen

        img_name = os.path.join(self.data_dir, class_name,
                                f'{(index % 64) + 1}.jpg')
        image = read_image(img_name).to(torch.float32)
        image = self.transformer(image)
        return image, class_idx


preprocess = transforms.Compose([
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.Resize(256, antialias=True),
    transforms.CenterCrop(224),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    # transforms.ToTensor(),
])


if __name__ == '__main__':
    fold = 6
    epochs = 10
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)
    vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    # vgg16.load_state_dict(torch.load('vgg16.pth'))
    # reset the last layer
    vgg16.classifier[-1] = nn.Linear(vgg16.classifier[-1].in_features, 3)
    vgg16 = vgg16.to(device)
    dataset = HockeyDataset(
        'dataset', models.VGG16_Weights.IMAGENET1K_V1.transforms(antialias=True))
    # kfold = KFold(n_splits=fold, shuffle=True)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train, test = random_split(dataset, [train_size, test_size])

    begin = time.time()
    # for (train, test) in train_test_split(dataset, test_size=0.2):
    print("Train: ", train, "Validation: ", test)
    trainloader = DataLoader(train, batch_size=4, shuffle=True, num_workers=4)
    valloader = DataLoader(test, batch_size=4, shuffle=True, num_workers=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(vgg16.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.StepLR(
           optimizer, step_size=5, gamma=0.1)

    best_weight = copy.deepcopy(vgg16.state_dict())
    best_acc = 0.0
    for epoch in range(epochs):
            # Training phase
            vgg16.train()
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in trainloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = vgg16(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels.long())
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                scheduler.step()

            epoch_loss = running_loss / len(train)
            epoch_acc = running_corrects.double() / len(train)
            print('Epoch: {} train Loss: {:.4f} Acc: {:.4f}'.format(epoch, epoch_loss, epoch_acc))
            running_loss = 0.0
            running_corrects = 0
            vgg16.eval()
            with torch.no_grad():
                for images, labels in valloader:
                    optimizer.zero_grad()
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = vgg16(images)
                    _, predicted = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels.long())
                    running_loss += loss.item() * images.size(0)
                    running_corrects += torch.sum(predicted == labels.data)
            
            epoch_loss = running_loss / len(test)
            epoch_acc = running_corrects.double() / len(test)
            print('Epoch: {} eval Loss: {:.4f} Acc: {:.4f}'.format(epoch, epoch_loss, epoch_acc))

            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_weight = copy.deepcopy(vgg16.state_dict())


    time_elapsed = time.time() - begin
    print(f"Finished Training, took {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print('Best accuracy: {:4f}'.format(best_acc))
    vgg16.load_state_dict(best_weight)



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
