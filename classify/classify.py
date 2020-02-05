#!/usr/bin/env python

###

import torch
import torchvision
import numpy as np
import os
import cv2

# from __future__ import print_function, division
# import pandas as pd
# from skimage import io, transform
# import matplotlib.pyplot as plt
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms, utils

def line_to_label(line):
    d = {'False' : 0, 'True' : 1}
    return d[line.split()[1]]

def batch_to_tensor(batch):
    return torch.tensor([line_to_label(label) for label in batch])

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size.
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = cv2.resize(image, self.output_size)
        return {'image': image, 'label': label}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))

        return {'image': torch.from_numpy(image),
                #'label': torch.from_numpy(np.array([1, 0]) if label == 'False' else np.array([0, 1]))
                #'label': torch.zeros(1, dtype=torch.long) if label == 'False' else torch.ones(1, dtype=torch.long)
                #'label': line_to_label(label) #0 if label == 'False' else 1
                'label': label #0 if label == 'False' else 1
        }

        # return {'image': torch.from_numpy(image),
        #         'label': torch.from_numpy(label)}

class SimpleClassifyDataset(torch.utils.data.Dataset):
    """Simple dataset for classification"""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
        """
        self.root_dir = root_dir
        self.transform = transform
        with open(os.path.join(root_dir, "annotation.txt")) as f:
            self.content = [line.strip() for line in f.readlines()]
            #self.content = [(s[0], bool(s[1]), bool(s[2]), float(s[3])) for s in self.content]

    def __len__(self):
        return len(self.content)

    def __getitem__(self, s):

        if (not isinstance(s, slice)):
            s = (s, s + 1, 1)
        else:
            s = s.indices(len(self.content))

        samples = []
        for idx in range(*s):
            img_name = os.path.join(self.root_dir, "%07d.jpg" % idx)
            image = np.float32(cv2.imread(img_name))
            sample = {'image': image, 'label': self.content[idx]}
            if self.transform:
                sample = self.transform(sample)
            samples.append(sample)

        if len(samples) == 1:
            return samples[0]
        else:
            return samples

import torch.nn as nn
import torch.nn.functional as F

batch_size_from_env = int(os.environ.get('BATCH_SIZE', 128))
epochs_from_env = int(os.environ.get('TRAIN_EPOCHS', 100))

def main():
    full_dataset = SimpleClassifyDataset("/storage/datasets/0001_flare_fog_rain",
                                         torchvision.transforms.Compose([Rescale((128, 128)),
                                                                         ToTensor()]))

    trainval_idx = int(len(full_dataset) * 0.5)
    trainloader = torch.utils.data.DataLoader(full_dataset[0:trainval_idx], batch_size=batch_size_from_env,
                                              shuffle=True, num_workers=2)

    testloader = torch.utils.data.DataLoader(full_dataset[trainval_idx:len(full_dataset)], batch_size=batch_size_from_env,
                                             shuffle=False, num_workers=2)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = torchvision.models.resnet18(num_classes=2)
    net.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    #optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

    for epoch in range(epochs_from_env):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; # OLD: data is a list of [inputs, labels]
            # inputs, labels = data

            inputs = data['image'].to(device)
            labels = batch_to_tensor(data['label']).to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 20 == 0:    # print every 20 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

                # # early exit
                # break

    # validate
    hit = 0
    N_val = 0
    for val_batch in testloader:
        #dataiter = iter(testloader)
        #val_batch = dataiter.next()
        images = val_batch['image'].to(device)
        labels = batch_to_tensor(val_batch['label']).to(device)

        outputs = net(images)
        _, predicted = torch.max(outputs, 1)

        # def imshow(img):
        #     #img = img / 2 + 0.5     # unnormalize
        #     #npimg = img.numpy()
        #     cv2.imshow('img', np.transpose(npimg, (1, 2, 0)))
        #     cv2.waitKey(0)

        for i in range(len(images)):
            hit += predicted[i].item() == labels[i].item()
            N_val += 1

            if (False):
            #if (predicted[i].item() != labels[i].item()):
                print(predicted[i], labels[i])
                #print (val_batch['label'][i].split()[0])
                cv2.imshow('img', cv2.imread(val_batch['label'][i].split()[0]))
                cv2.waitKey(0)

    print ("Accuracy = ", hit / N_val)

if __name__ == '__main__':
    main()
