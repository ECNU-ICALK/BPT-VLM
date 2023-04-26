import torch
import argparse
import yaml
from tqdm import tqdm
from algorithm.CMA_ES import shallow_cma
from algorithm.LM_CMA_ES import Shallow_LMCMAES
from algorithm.MMES import Shallow_MMES
from algorithm.LMMAES import Shallow_LMMAES
from model.Shallow_Prompt_CLIP import PromptCLIP_Shallow
import numpy as np
import time

__classification__ = ["CIFAR100","caltech101","StanfordCars","OxfordPets","UCF-101","DTD","EuroSAT",
                      "Food101","SUN397","ImageNet"]
__pypop__ = ["shallow_lmcmaes","shallow_mmes","shallow_dcem","shallow_maes"]
__dataset__ = "/home/yu/dataset"
__output__ = "/home/yu/dataset/result"
# __output__ = "/home/yu/result"
__backbone__ = "ViT-B/32"

parser = argparse.ArgumentParser()
parser.add_argument("--task_name", default="CIFAR100", type=str)
parser.add_argument("--opt", default="shallow_cma", type=str)
parser.add_argument("--parallel", action='store_true', help='Whether to allow parallel evaluation')



args = parser.parse_args()
assert "shallow" in args.opt, "Only shallow prompt tuning is supported in this file."
cfg = yaml.load(open("./configs/shallow_prompt.yaml"), Loader=yaml.FullLoader)

cfg["opt_name"] = args.opt
cfg["data_dir"] = __dataset__
cfg["output_dir"] = __output__
cfg["opt_name"] = args.opt
cfg["backbone"] = __backbone__

for k,v in cfg[args.task_name].items():
    cfg[k]=v
cfg["parallel"] = args.parallel

device = "cuda" if torch.cuda.is_available() else "cpu"
intrinsic_dim_L = cfg["intrinsic_dim_L"]
intrinsic_dim_V = cfg["intrinsic_dim_V"]



# Eval function and Settings(if needed)+
def fitness_eval(prompt_zip):
    prompt_text_list = prompt_clip.generate_text_prompts([x[:intrinsic_dim_L] for x in [prompt_zip]])
    prompt_image_list = prompt_clip.generate_visual_prompts([x[intrinsic_dim_L:] for x in [prompt_zip]])
    fitnesses = [prompt_clip.eval(x).item() for x in zip(prompt_text_list, prompt_image_list)]


    if prompt_clip.num_call % (prompt_clip.test_every) == 0:
        print("-------------------------Epoch {}---------------------------".format(prompt_clip.num_call/prompt_clip.test_every))
    if prompt_clip.num_call % (prompt_clip.popsize) == 0:
        print("Evaluation of Individual: {}, Generation: {}".format(prompt_clip.num_call % prompt_clip.popsize,
                                                                int(prompt_clip.num_call / prompt_clip.popsize)))
    if prompt_clip.num_call % prompt_clip.test_every == 0:
        print("current loss: {}".format(prompt_clip.min_loss))
        print("Best Prompt Embedding - Acc : " + str(prompt_clip.best_accuracy))

    return fitnesses[0]

ndim_problem = intrinsic_dim_L + intrinsic_dim_V
pro = {'fitness_function': fitness_eval,
       'ndim_problem': ndim_problem}
opt_cfg = {'fitness_threshold': 1e-10,
           'seed_rng': 0,
           'max_runtime':20800,
           'x': 0 * np.ones((ndim_problem,)),  # mean
           'sigma': cfg['sigma'],
           'verbose_frequency': 5,
           'n_individuals': cfg["popsize"],
           'is_restart': False}



# Load algorithm
opt = None
if args.opt == "shallow_cma":
    opt = shallow_cma(cfg)
elif args.opt == "shallow_lmcmaes":
    opt = Shallow_LMCMAES(pro, opt_cfg)
elif args.opt == "shallow_mmes":
    opt = Shallow_MMES(pro, opt_cfg)
elif args.opt == "shallow_lmmaes":
    opt = Shallow_LMMAES(pro,opt_cfg)


# Build CLIP model
if args.task_name in __classification__:
    prompt_clip = PromptCLIP_Shallow(args.task_name,cfg)
print('Population Size: {}'.format(cfg["popsize"]))

# Black-box prompt tuning

if args.opt in __pypop__:
    if args.task_name in __classification__:
        text_context = prompt_clip.get_text_information()
        image_context = prompt_clip.get_image_information()
        prompt_clip.text_encoder.set_context(text_context)
        prompt_clip.image_encoder.set_context(image_context)
        res = opt.optimize()
else:
    if args.task_name in __classification__:
        text_context = prompt_clip.get_text_information()
        image_context =prompt_clip.get_image_information()
        prompt_clip.text_encoder.set_context(text_context)
        prompt_clip.image_encoder.set_context(image_context)
        while not opt.stop():
            solutions = opt.ask()
            #prompt_list [popsize, n_cls, embedding_dim]  tokenized_prompts [n_cls, embedding_dim]
            prompt_text_list= prompt_clip.generate_text_prompts([x[:intrinsic_dim_L] for x in solutions])
            prompt_image_list = prompt_clip.generate_visual_prompts([x[intrinsic_dim_L:] for x in solutions])
            if cfg["parallel"]:
                fitnesses = prompt_clip.eval([prompt_text_list, prompt_image_list])
                fitnesses = [x.item() for x in tqdm(fitnesses,ncols=50)]
            else:
                fitnesses = [prompt_clip.eval(x).item() for x in tqdm(zip(prompt_text_list, prompt_image_list))]
            # output current loss and acc
            if prompt_clip.num_call % prompt_clip.test_every == 0:
                print("current loss: {}".format(prompt_clip.min_loss))
                print("Best Prompt Embedding - Acc : " + str(prompt_clip.best_accuracy))
            opt.tell(solutions, fitnesses)
    else:
        image_context =prompt_clip.get_image_information()
        prompt_clip.image_encoder.set_context(image_context)
        while not opt.stop():
            solutions = opt.ask()
            prompt_text_list = prompt_clip.generate_text_prompts([x[:intrinsic_dim_L] for x in solutions])
            prompt_image_list = prompt_clip.generate_visual_prompts([x[intrinsic_dim_L:] for x in solutions])
            if cfg["parallel"]:
                fitnesses = prompt_clip.eval([prompt_text_list, prompt_image_list])
                fitnesses = [x.item() for x in tqdm(fitnesses, ncols=50)]
            else:
                fitnesses = [prompt_clip.eval(x).item() for x in tqdm(zip(prompt_text_list, prompt_image_list))]
            # output current loss and acc
            if prompt_clip.num_call % prompt_clip.test_every == 0:
                print("current loss: {}".format(prompt_clip.min_loss))
                print("Best Prompt Embedding - Acc : " + str(prompt_clip.best_accuracy))
            opt.tell(solutions, fitnesses)




acc = prompt_clip.test()
pass