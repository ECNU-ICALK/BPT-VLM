import os
from collections import OrderedDict
import json
import numpy as np
text_file = "/root/autodl-tmp/dataset/classnames.txt"
image_dir = "/root/autodl-tmp/dataset/imagenet/image_data"


def read_classnames(text_file):
    """
    <folder name>: <class name>.
    """
    classnames = OrderedDict()
    with open(text_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(" ")
            folder = line[0]
            classname = " ".join(line[1:])
            classnames[folder] = classname
    return classnames

def read_data(classnames, split_dir):
    split_dir = os.path.join(image_dir, split_dir)
    folders = sorted(f.name for f in os.scandir(split_dir) if f.is_dir())
    items = []

    for label, folder in enumerate(folders):
        path = os.path.join(split_dir, folder)
        #no hidden folders
        imnames = [f for f in os.listdir(path) if not f.startswith(".")]
        if split_dir=="train":
            np.random.seed(42)
            np.random.shuffle(imnames)
            imnames = imnames[:64]
        classname = classnames[folder]
        for imname in imnames:
            impath = os.path.join(split_dir, folder, imname)
            # item = Datum(impath=impath, label=label, classname=classname)
            item = [impath,label,classname]
            items.append(item)
    return items



classnames = read_classnames(text_file)
train = read_data(classnames, "train")
test = read_data(classnames,"val")
all_data = {"train":train,"test":test}
json.dump(all_data,open("split.json","w"),indent=4)