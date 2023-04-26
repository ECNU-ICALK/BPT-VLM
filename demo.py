import torch
import argparse
import os
import yaml
from model.Shallow_Prompt_CLIP import PromptCLIP_Shallow
from model.Deep_Prompt_CLIP import PromptCLIP_Deep
from model.Shallow_Prompt_ReCLIP import PromptReCLIP_Shallow
import time
from collections import defaultdict
import json
import argparse
import random
import copy
from PIL import Image
from tqdm import tqdm
from model.reclip.interpreter import *
from model.reclip.executor import *
from model.reclip.parse import Parse
from torch.nn import functional as F

# Global_Variables
__classification__ = ["CIFAR100","caltech101","StanfordCars","OxfordPets","UCF-101","DTD","EuroSAT",
                      "Food101","SUN397","ImageNet"]
__pypop__ = ["shallow_lmcmaes","shallow_mmes","shallow_dcem","shallow_maes"]
__dataset__ = "/home/yu/dataset"
__output__ = "/home/yu/dataset/result"
__backbone__ = "ViT-B/32"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Parser
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_dir", required=True, type=str)
parser.add_argument("--checkpoint_name", default="ImageNet", type=str)
parser.add_argument("--task_name", default="caltech101", type=str)
parser.add_argument("--split", default=None, type=str)
parser.add_argument("--opt", default="shallow_cma", type=str)
parser.add_argument("--backbone",default="ViT-B-32")
args = parser.parse_args()

# Configuration
if "shallow" in args.opt:
    config_file = os.path.join("./configs","shallow_prompt.yaml")
else:
    config_file = os.path.join("./configs", "deep_prompt.yaml")
cfg = yaml.load(open(config_file), Loader=yaml.FullLoader)
cfg["opt_name"] = args.opt
cfg["data_dir"] = __dataset__
cfg["output_dir"] = __output__
cfg["opt_name"] = args.opt
cfg["backbone"] = __backbone__
cfg["parallel"] = False
for k,v in cfg[args.task_name].items():
    cfg[k]=v

task_cfg_file = os.path.join("./configs","refcoco.yaml")
task_cfg = yaml.load(open(task_cfg_file), Loader=yaml.FullLoader)

# Load the Model
assert os.path.exists(args.checkpoint_dir), "No such checkpoint directory path {}".format(args.checkpoint_dir)
checkpoint_path = os.path.join(args.checkpoint_dir,args.checkpoint_name,args.checkpoint_name+'_'+args.opt+'_'+args.backbone+'.pth')
content = torch.load(checkpoint_path)

prompt_clip = None
if "shallow" in args.opt:
    if args.task_name in __classification__:
        prompt_clip = PromptCLIP_Shallow(args.task_name,cfg)
    else:
        prompt_clip = PromptReCLIP_Shallow(args.task_name,cfg)
else:
    prompt_clip = PromptCLIP_Deep(args.task_name,cfg)
    #TODO
prompt_clip.best_prompt_image,prompt_clip.best_prompt_text = content["best_prompt_image"],content["best_prompt_text"]

# Run Test
if args.task_name in __classification__:
    text_context = prompt_clip.get_text_information()
    image_context = prompt_clip.get_image_information()
    prompt_clip.text_encoder.set_context(text_context)
    prompt_clip.image_encoder.set_context(image_context)
    acc = prompt_clip.test()
    print(acc)
else:
    assert os.path.exists(task_cfg["input_dir_path"]), "No such input directory path {}".format(task_cfg["input_dir_path"])
    input_file_path = os.path.join(task_cfg["input_dir_path"],task_cfg["refcoco_file"][args.task_name][args.split])

    if not os.path.exists(task_cfg["output_dir_path"]):
        os.makedirs(task_cfg["output_dir_path"])
    output_file_path = os.path.join(task_cfg["output_dir_path"], args.opt+"_"+task_cfg["refcoco_file"][args.task_name][args.split])
    output_file = open(output_file_path,'w')

    print("[{} -- Input Json File]: {}".format(args.task_name,input_file_path))
    print("[{} -- Output Json File]: {}".format(args.task_name, output_file_path))
    print("[{} -- Image Root]: {}".format(args.task_name, task_cfg["image_root"]))

    with open(input_file_path) as f:
        lines = f.readlines()
        data = [json.loads(line) for line in lines]
    f.close()

    executor = ClipExecutor(task_name=args.task_name,task_cfg=task_cfg,prompt_clip=prompt_clip)
    method = Parse(task_cfg)
    correct_count = 0
    total_count = 0
    batch_count = 0
    batch_boxes = []
    batch_gold_boxes = []
    batch_gold_index = []
    batch_file_names = []
    batch_sentences = []
    for datum in tqdm(data):
        if "coco" in datum["file_name"].lower():
            file_name = "_".join(datum["file_name"].split("_")[:-1])+".jpg"
        else:
            file_name = datum["file_name"]
        img_path = os.path.join(task_cfg["image_root"], file_name)
        img = Image.open(img_path).convert('RGB')
        gold_boxes = [Box(x=ann["bbox"][0], y=ann["bbox"][1], w=ann["bbox"][2], h=ann["bbox"][3]) for ann in datum["anns"]]
        if isinstance(datum["ann_id"], int) or isinstance(datum["ann_id"], str):
            datum["ann_id"] = [datum["ann_id"]]
        assert isinstance(datum["ann_id"], list)
        gold_index = [i for i in range(len(datum["anns"])) if datum["anns"][i]["id"] in datum["ann_id"]]
        for sentence in datum["sentences"]:
            boxes = gold_boxes
            env = Environment(img, boxes, executor, (task_cfg["mdetr"] is not None and not task_cfg["mdetr_given_bboxes"]), str(datum["image_id"]))
            if task_cfg["shuffle_words"]:
                words = sentence["raw"].lower().split()
                random.shuffle(words)
                result = method.execute(" ".join(words), env=env)
            else:
                result = method.execute(caption=sentence["raw"].lower().rstrip('.'), env=env)
            boxes = env.boxes
            print(sentence["raw"].lower().rstrip('.'))
            correct = False
            for g_index in gold_index:
                if iou(boxes[result["pred"]], gold_boxes[g_index]) > 0.5:
                    correct = True
                    break
            if correct:
                result["correct"] = 1
                correct_count += 1
            else:
                result["correct"] = 0
            if task_cfg["detector_file"]:
                argmax_ious = []
                max_ious = []
                for g_index in gold_index:
                    ious = [iou(box, gold_boxes[g_index]) for box in boxes]
                    argmax_iou = -1
                    max_iou = 0
                    if max(ious) >= 0.5:
                        for index, value in enumerate(ious):
                            if value > max_iou:
                                max_iou = value
                                argmax_iou = index
                    argmax_ious.append(argmax_iou)
                    max_ious.append(max_iou)
                argmax_iou = -1
                max_iou = 0
                if max(max_ious) >= 0.5:
                    for index, value in zip(argmax_ious, max_ious):
                        if value > max_iou:
                            max_iou = value
                            argmax_iou = index
                result["gold_index"] = argmax_iou
            else:
                result["gold_index"] = gold_index
            result["bboxes"] = [[box.left, box.top, box.right, box.bottom] for box in boxes]
            result["file_name"] = file_name
            result["probabilities"] = result["probs"]
            result["text"] = sentence["raw"].lower()
            total_count += 1
            est_acc = 100 * correct_count / total_count
            result["est_acc"] = est_acc
            # Serialize numpy arrays for JSON.
            for key in result:
                if isinstance(result[key], np.ndarray):
                    result[key] = result[key].tolist()
                if isinstance(result[key], np.int64):
                    result[key] = result[key].item()
            output_file.write(json.dumps(result)+"\n")
            print(f"est_acc: {est_acc:.3f}")


    output_file.close()
    print(f"acc: {100 * correct_count / total_count:.3f}")

    logs = {}
    logs["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    logs["opt"] = args.opt
    logs["task_name"] = args.task_name
    logs["split"] = args.split
    logs["acc"] = 100 * correct_count / total_count
    stats = method.get_stats()
    if stats:
        pairs = sorted(list(stats.items()), key=lambda tup: tup[0])
        for key, value in pairs:
            if isinstance(value, float):
                print(f"{key}: {value:.5f}")
            else:
                print(f"{key}: {value}")
            # logs[key] = value
    for key in logs:
        if isinstance(logs[key], np.ndarray):
            logs[key] = logs[key].tolist()
        if isinstance(logs[key], np.int64):
            logs[key] = logs[key].item()

    if task_cfg["log_file"]:
        log_file_path = os.path.join(task_cfg["output_dir_path"],task_cfg["log_file"])
        if not os.path.exists(log_file_path):
            with open(log_file_path, 'w') as f:
                print("The log json file is created")
            f.close()

        with open(log_file_path) as f:
            log_lines = f.readlines()
            log_data = [json.loads(line) for line in log_lines]
        f.close()

        log_data.append(logs)
        log_output = open(log_file_path, "w")
        for x in log_data:
            log_output.write(json.dumps(x) + '\n')
        log_output.close()


pass