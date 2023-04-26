import torch
import torchvision
import os
import numpy as np
from torchvision.datasets import CIFAR100
from torch.utils.data import Dataset, DataLoader
import argparse
from dataset.utils import Util
import pickle
from PIL import Image

class FewshotDataset(Dataset):
    def __init__(self, args):
        self.root = args["root"]
        self.dataset_dir = args["dataset_dir"]
        self.seed = args["seed"]

        self.dataset_dir = os.path.join(self.root,self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "image_data")
        self.split_path = os.path.join(self.dataset_dir, "split.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        Util.mkdir_if_missing(self.split_fewshot_dir)

        self.shots = args["shots"]
        self.prepocess = args["preprocess"]
        self.all_data = Util.read_split(self.split_path)


        '''
        [['Faces/image_0274.jpg', 0, 'face'], ['Faces/image_0339.jpg', 0, 'face'], ['Faces/image_0124.jpg', 0, 'face'],
        .........
        ['yin_yang/image_0055.jpg', 99, 'yin_yang'], ['yin_yang/image_0031.jpg', 99, 'yin_yang']]
        '''
        self.all_train = self.all_data["train"]
        # print(self.all_train)
        preprocessed = os.path.join(self.split_fewshot_dir,f"shot_{self.shots}.pkl")
        if os.path.exists(preprocessed):
            print(f"Loading preprocessed few-shot data from {preprocessed}")
            with open(preprocessed,"rb") as file:
                content = pickle.load(file)
                self.new_train_data = content["new_train_data"]
                self.classes = content["classes"]
        else:
            self.new_train_data,self.classes = self.construct_few_shot_data()

            print(f"Saving preprocessed few-shot data to {preprocessed}")
            content = {"new_train_data":self.new_train_data,"classes":self.classes}
            with open(preprocessed, "wb") as file:
                pickle.dump(content, file, protocol=pickle.HIGHEST_PROTOCOL)


    def __len__(self):
        return len(self.new_train_data)

    def __getitem__(self, idx):

        return {"image": self.new_train_data[idx][0], "label": self.new_train_data[idx][1]}

    def construct_few_shot_data(self):
        new_train_data = []
        train_shot_count={}
        classes_dict = {}
        all_indices = [_ for _ in range(len(self.all_train))]
        np.random.seed(self.seed)
        np.random.shuffle(all_indices)

        for index in all_indices:
            label,classname = self.all_train[index][1],self.all_train[index][2]

            if label not in train_shot_count:
                train_shot_count[label]=0
                classes_dict[label]=classname
            if train_shot_count[label]<self.shots:
                tmp = self.all_train[index]
                image_path = os.path.join(self.image_dir,tmp[0])
                # convert img to compatible tensors
                tmp_data = [self.prepocess(Image.open(image_path)),tmp[1]]
                new_train_data.append(tmp_data)
                train_shot_count[label] += 1
        classes = [classes_dict[i] for i in range(len(classes_dict))]
        return new_train_data,classes




def load_train(batch_size=1,seed=42,shots=16,preprocess=None,root=None,dataset_dir=None):
    args = {"shots":shots,"preprocess":preprocess,"root":root,"dataset_dir":dataset_dir,"seed":seed}
    train_data = FewshotDataset(args)
    train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True,num_workers=4)
    return train_data,train_loader

class TestDataset(Dataset):
    def __init__(self, args):
        self.prepocess = args["preprocess"]
        self.root = args["root"]
        self.dataset_dir = args["dataset_dir"]
        self.dataset_dir = os.path.join(self.root,self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "image_data")
        self.split_path = os.path.join(self.dataset_dir, "split.json")
        self.all_data = Util.read_split(self.split_path)
        self.all_test = self.all_data["test"]
#-------------------------------------------------------------------------------------------
        self.test_data_dir = os.path.join(self.dataset_dir, "test_data")
        Util.mkdir_if_missing(self.test_data_dir)
        preprocessed = os.path.join(self.test_data_dir,f"test.pkl")
        if os.path.exists(preprocessed):
            print(f"Loading preprocessed test data from {preprocessed}")
            with open(preprocessed,"rb") as file:
                content = pickle.load(file)
                self.all_test = content
        else:
            for tmp in self.all_test:
                image_path = os.path.join(self.image_dir, tmp[0])
                tmp[0] = self.prepocess(Image.open(image_path))
            print(f"Saving preprocessed test data to {preprocessed}")
            content = self.all_test
            with open(preprocessed, "wb") as file:
                pickle.dump(content, file, protocol=pickle.HIGHEST_PROTOCOL)
#-----------------------------------------------------------------------------------------------


    def __len__(self):
        return len(self.all_test)

    def __getitem__(self, idx):
        return {"image": self.all_test[idx][0], "label": self.all_test[idx][1]}

def load_test(batch_size=1,preprocess=None,root=None,dataset_dir=None):
    args = {"preprocess":preprocess,"root":root,"dataset_dir":dataset_dir}
    test_data = TestDataset(args)
    test_loader = DataLoader(test_data,batch_size=batch_size,shuffle=True,num_workers=4)
    return test_data,test_loader

