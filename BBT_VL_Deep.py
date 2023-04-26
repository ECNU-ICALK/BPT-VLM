import torch
import argparse
import yaml
from tqdm import tqdm
from algorithm.CMA_ES import shallow_cma,deep_cma
from model.Deep_Prompt_CLIP import PromptCLIP_Deep

__classification__ = ["CIFAR100","caltech101","StanfordCars","OxfordPets","UCF-101","DTD","EuroSAT",
                      "Food101","SUN397","ImageNet","refcoco"]
__pypop__ = ["shallow_lmcmaes","shallow_mmes","shallow_dcem","shallow_lmmaes"]
__dataset__ = "/home/yu/dataset"
__output__ = "/home/yu/dataset/result"
# __output__ = "/home/yu/result"
__backbone__ = "ViT-B/32"


parser = argparse.ArgumentParser()

parser.add_argument("--task_name", default="CIFAR100", type=str)
parser.add_argument("--opt", default="deep_cma", type=str)
parser.add_argument("--parallel", action='store_true', help='Whether to allow parallel evaluation')

args = parser.parse_args()
assert "deep" in args.opt, "Only deep prompt tuning is supported in this file."

cfg = yaml.load(open("./configs/deep_prompt.yaml"), Loader=yaml.FullLoader)
cfg["opt_name"] = args.opt
cfg["data_dir"] = __dataset__
cfg["output_dir"] = __output__
cfg["opt_name"] = args.opt
cfg["backbone"] = __backbone__
cfg["budget"] = cfg[args.task_name]["budget"]
cfg["test_every"] = cfg[args.task_name]["test_every"]
cfg["maxiter"] = cfg["budget"] // (cfg["popsize"] * cfg["num_prompt_layer"])
cfg["parallel"] = args.parallel


device = "cuda" if torch.cuda.is_available() else "cpu"
intrinsic_dim_L = cfg["intrinsic_dim_L"]
intrinsic_dim_V = cfg["intrinsic_dim_V"]



opt = None
if args.opt == "deep_cma":
    opt = deep_cma(cfg)

prompt_clip = PromptCLIP_Deep(args.task_name,cfg)
print('Population Size: {}'.format(opt.opt_setting["popsize"]))

context = prompt_clip.get_text_information()
prompt_clip.text_encoder.set_context(context)
context = prompt_clip.get_image_information()
prompt_clip.image_encoder.set_context(context)
for _ in range(cfg['maxiter']):
    for i in range(cfg["num_prompt_layer"]):
        # perform prompt tuning on ith layer
        solutions = opt.ask(i)
        prompt_text_list = prompt_clip.generate_text_prompts([x[:intrinsic_dim_L] for x in solutions],i)
        prompt_image_list = prompt_clip.generate_visual_prompts(
            [x[intrinsic_dim_L:] for x in solutions],i)
        if cfg["parallel"]:
            fitnesses = prompt_clip.eval([prompt_text_list, prompt_image_list],i)
            fitnesses = [x.item() for x in tqdm(fitnesses,ncols=50)]
            # num_call = prompt_clip.num_call * prompt_clip.popsize if prompt_clip.parallel else prompt_clip.num_call
            if prompt_clip.num_call % prompt_clip.test_every == 0:
                print("current loss: {}".format(prompt_clip.min_loss))
                print("Best Prompt Embedding - Acc : " + str(prompt_clip.best_accuracy))
        else:
            fitnesses = [prompt_clip.eval(x,i).item() for x in tqdm(zip(prompt_text_list,prompt_image_list))]
        opt.tell(solutions,fitnesses,i)


acc = prompt_clip.test()
pass