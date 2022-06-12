import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
import copy
import lightly
import numpy as np

import lightly.models as models
import lightly.loss as loss
import lightly.data as data

from lightly.models.utils import deactivate_requires_grad
from lightly.models.utils import update_momentum
from lightly.models.utils import batch_shuffle
from lightly.models.utils import batch_unshuffle


DATAPATH_tr = '../data/CIFAR10/train/'
DATAPATH_tt = '../data/CIFAR10/test/'


collate_fn = lightly.data.SimCLRCollateFunction(
    input_size=32,
    normalize={'mean':[0.4914, 0.4822, 0.4465],'std':[0.2023, 0.1994, 0.2010]}
)


def define_transforms(img_size=32):

    train_classifier_transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(img_size, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )
    ])
    test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((img_size, img_size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )
    ])
    
    return train_classifier_transforms, test_transforms


def get_dataloader(c, *transforms):
    
    train_classifier_transforms, test_transforms = transforms
    
    dataset_train_ssl = lightly.data.LightlyDataset(input_dir=c.DATAPATH_tr)

    ## 50% of labeled dataset for training classifier 
    train_index = list(np.random.choice(list(range(len(dataset_train_ssl))), len(dataset_train_ssl)//2, replace=False))
    dataset_train_classifier = lightly.data.LightlyDataset(
        input_dir=c.DATAPATH_tr,
        transform=train_classifier_transforms,
        filenames=np.array(dataset_train_ssl.get_filenames())[train_index]
    )

    dataset_test = lightly.data.LightlyDataset(
        input_dir=c.DATAPATH_tt,
        transform=test_transforms
    )
    
    dataloader_train_ssl = torch.utils.data.DataLoader(
        dataset_train_ssl,
        batch_size=c.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=c.num_workers
        )

    dataloader_train_classifier = torch.utils.data.DataLoader(
        dataset_train_classifier,
        batch_size=c.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=c.num_workers
        )

    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=c.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=c.num_workers
        )
    
    return dataloader_train_ssl, dataloader_train_classifier, dataloader_test