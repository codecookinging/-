from __future__ import absolute_import
import torch.utils.data as data
import torch
import numpy as np
from torchvision import transforms
import os
from PIL import Image

class dataload(data.Dataset):
    def __init__(self,img_path,input_transform=None,target_transform=None,is_train=True):
        super(dataload,self).__init__()

        self.train = is_train
        self.img_names,self.labels = [],[]
        with open(img_path,'r') as f:
            lines = [i.strip() for i in f.readlines()]
            for line in lines:
                name,tag = line.split('|')
                self.img_names.append(name)
                self.labels.append(int(tag))
        print('Loading {} {} images'.format(
            'training' if self.train else 'test',len(self.img_names)))
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self,index):
        img = Image.open(self.img_names[index])
        target = self.labels[index]
        if self.input_transform is not None:
            img = self.input_transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img,target

    def __len__(self):
        return len(self.img_names)

def get_data(train_list,test_list,batch_size,use_cuda=True):
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    test_trsf = transforms.Compose([transforms.Resize([512, 512]), transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    test_data = dataload(test_list, test_trsf, is_train=False)
    test_loader = data.DataLoader(test_data, batch_size=batch_size, shuffle=False, **kwargs)
    trsf = transforms.Compose([transforms.Resize([512,512]),transforms.RandomApply([transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip(),transforms.RandomGrayscale()]),
                               transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    train_data = dataload(train_list, trsf,)
    train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True, **kwargs)
    return train_loader, test_loader

if __name__ == '__main__':
    # test dataloader
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    train_dir = '../train_list.txt'
    test_dir = '../val_list.txt'
    train_load,test_load = get_data(train_dir,test_dir,2)
    for data,target in train_load:
        import ipdb
        # ipdb.set_trace()

        print(len(train_load))
        print(data.size())
        print(target.size())
        exit()


