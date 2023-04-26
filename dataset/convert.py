import json
import os
import shutil


source_path = "/home/yu/dataset/caltech101"
target_path = "/home/yu/dataset/caltech101_Gen"
# image_train_source_path = "cars_train"
# image_test_source_path = "cars_test"

if not os.path.exists(target_path):
    os.makedirs(target_path)


f = json.load(open(source_path+"/split_zhou_Caltech101.json","r",encoding="utf-8"))
# print(f)
if not os.path.exists(target_path + "/image_data"):
    os.makedirs(target_path + "/image_data")

for i in range(len(f["train"])):
    f["train"][i][0]=f["train"][i][0].replace(" ","_")
    f["train"][i][-1]=f["train"][i][-1].replace(" ","_")
    # f["train"][i][0] = f["train"][i][0].replace("/", "_")
    # f["train"][i][-1] = f["train"][i][-1].replace("/", "_")
    f["train"][i][0] = f["train"][i][0].replace("-", "_")
    f["train"][i][-1] = f["train"][i][-1].replace("-", "_")

    if not os.path.exists(target_path+"/image_data/"+f["train"][i][-1]):
        os.makedirs(target_path+"/image_data/"+f["train"][i][-1])
    shutil.copy(source_path+"/101_ObjectCategories/"+f["train"][i][0], target_path+"/image_data/"+f["train"][i][-1]+"/"+f["train"][i][0].split("/")[-1])
    f["train"][i][0] = f["train"][i][-1] + "/" + f["train"][i][0].split("/")[-1]

for i in range(len(f["test"])):
    f["test"][i][0]=f["test"][i][0].replace(" ","_")
    f["test"][i][-1]=f["test"][i][-1].replace(" ","_")
    # f["test"][i][0]=f["test"][i][0].replace("/","_")
    # f["test"][i][-1]=f["test"][i][-1].replace("/","_")
    f["test"][i][0]=f["test"][i][0].replace("-","_")
    f["test"][i][-1]=f["test"][i][-1].replace("-","_")

    if not os.path.exists(target_path+"/image_data/"+f["test"][i][-1]):
        os.makedirs(target_path+"/image_data/"+f["test"][i][-1])
    shutil.copy(source_path+"/101_ObjectCategories/"+f["test"][i][0], target_path+"/image_data/"+f["test"][i][-1]+"/"+f["test"][i][0].split("/")[-1])
    f["test"][i][0] = f["test"][i][-1] + "/" + f["test"][i][0].split("/")[-1]

json.dump(f,open(target_path+"/split.json","w"),indent=4)