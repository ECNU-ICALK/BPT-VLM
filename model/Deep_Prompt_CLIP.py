#TODO
import os
import torch
from torch.nn import functional as F
import numpy as np
import clip
from torchvision.datasets import CIFAR100
from dataset.cifar100 import load_train_cifar100, load_test_cifar100
from model.deep_encoder import TextEncoder_Deep,VisionEncoder_Deep
from model.analysis_utils import Analysis_Util
from dataset.general import load_train,load_test
class PromptCLIP_Deep:
    def __init__(self,task_name,cfg):
        self.task_name = task_name
        self.opt_name = cfg["opt_name"]
        self.data_dir = cfg["data_dir"]
        self.output_dir = cfg["output_dir"]
        self.backbone = cfg["backbone"]
        self.batch_size = cfg["batch_size"]
        self.popsize = cfg["popsize"]
        self.parallel = cfg["parallel"]
        self.k_shot = cfg["k_shot"]
        self.seed = cfg["seed"]
        self.num_call = 0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32",device=self.device)
        self.load_dataset()

        self.acc = []
        self.loss = []
        self.best_accuracy = 0
        # Text Encoder
        self.n_prompt_tokens_L = cfg["n_prompt_tokens_L"]
        self.intrinsic_dim_L = cfg["intrinsic_dim_L"]
        self.ctx_dim_L = self.model.ln_final.weight.shape[0]
        self.text_encoder = TextEncoder_Deep(self.model)
        # Image Encoder
        self.n_prompt_tokens_V = cfg["n_prompt_tokens_V"]
        self.ctx_dim_V = self.model.visual.width
        self.intrinsic_dim_V = cfg["intrinsic_dim_V"]
        self.image_encoder = VisionEncoder_Deep(self.model)
        self.image_encoder.n_prompt_tokens_V = self.n_prompt_tokens_V

        self.loss_type = cfg["loss_type"]
        self.init_prompt = None
        self.imsize = self.image_encoder.input_resolution
        self.logit_scale = self.model.logit_scale
        self.dtype = self.model.dtype
        self.num_prompt_layer = cfg["num_prompt_layer"]
        assert(self.num_prompt_layer<=self.model.transformer.layers)

        #Init with random tensor
        self.best_prompt_text = torch.zeros(self.num_prompt_layer,self.n_prompt_tokens_L,self.ctx_dim_L,device=self.device,dtype=self.dtype)
        self.best_prompt_image = torch.zeros(self.num_prompt_layer,self.n_prompt_tokens_V,self.ctx_dim_V,device=self.device,dtype=self.dtype)
        self.min_loss = None
        self.test_every = cfg["test_every"] if self.parallel else cfg["test_every"]*self.popsize
        self.sigma = cfg["sigma"]
        self.alpha = cfg["alpha"]

        # Lauguage Linear Layers
        self.linear_L = torch.nn.ModuleList([torch.nn.Linear(self.intrinsic_dim_L, self.n_prompt_tokens_L * self.ctx_dim_L,
                                      bias=False,device=self.device,dtype=self.dtype) for _ in range(self.num_prompt_layer)])
        embedding = self.model.token_embedding.weight.cpu()
        mu_hat = np.mean(embedding.reshape(-1).detach().cpu().numpy())
        std_hat = np.std(embedding.reshape(-1).detach().cpu().numpy())
        mu = 0.0
        std = std_hat / (np.sqrt(self.intrinsic_dim_L) * self.sigma)
        print('[Embedding] mu: {} | std: {} [RandProj]  mu: {} | std: {}'.format(mu_hat, std_hat, mu, std))
        for p in self.linear_L[0].parameters():
            torch.nn.init.normal_(p, mu, std)
        self.intermediate_stats_L = [(mu,std)]
        # Vision Linear Layers
        self.linear_V = torch.nn.ModuleList([torch.nn.Linear(self.intrinsic_dim_V, self.n_prompt_tokens_V * self.ctx_dim_V,
                                        bias=False, device=self.device, dtype=self.dtype) for _ in range(self.num_prompt_layer)])
        conv = self.model.visual.conv1.weight.cpu()
        mu_hat = np.mean(conv.reshape(-1).detach().cpu().numpy())
        std_hat = np.std(embedding.reshape(-1).detach().cpu().numpy())
        #mu = 0.0
        mu = mu_hat*3072/self.intrinsic_dim_V
        std = std_hat * np.sqrt(3072/self.intrinsic_dim_V) * self.sigma
        print('[Conv] mu: {} | std: {} [RandProj]  mu: {} | std: {}'.format(mu_hat, std_hat, mu, std))
        for p in self.linear_V[0].parameters():
            torch.nn.init.normal_(p, mu, std)
        self.intermediate_stats_V = [(mu, std)]


    def store_best_prompt(self,prompt_text,prompt_image,i):
        self.best_prompt_text[i] = prompt_text
        self.best_prompt_image[i] = prompt_image
    def use_best_prompt(self,prompt_text,prompt_image,i):
        tmp_text_prompt = self.best_prompt_text.clone()
        tmp_image_prompt = self.best_prompt_image.clone()
        if self.parallel:
            tmp_text_prompt = [x for x in tmp_text_prompt]
            tmp_image_prompt = [x for x in tmp_image_prompt]
        tmp_text_prompt[i] = prompt_text
        tmp_image_prompt[i] = prompt_image
        return tmp_text_prompt,tmp_image_prompt

    def get_text_information(self,caption=None):
        prompt_prefix = " ".join(["X"] * self.n_prompt_tokens_L)
        if caption is None:

            classnames = [name.replace("_", " ") for name in self.classes]
            pattern_prompts = [prompt_prefix + " " + name + "." for name in classnames]
            tokenized_pattern_prompts= torch.cat([clip.tokenize(p) for p in pattern_prompts]).to(self.device)
            with torch.no_grad():
                init_pattern_embedding = self.model.token_embedding(tokenized_pattern_prompts).type(self.dtype)
            context = {"n_cls":self.n_cls, "n_prompt_tokens_L":self.n_prompt_tokens_L,
                       "init_pattern_embedding":init_pattern_embedding, "tokenized_pattern_prompts":tokenized_pattern_prompts,
                       "batch_size":self.batch_size,"pop_size":self.popsize,"parallel":self.parallel}
        else:
            pattern_prompt = prompt_prefix + caption + "."
            tokenized_pattern_prompts = torch.cat([clip.tokenize(pattern_prompt)]).to(self.device)
            with torch.no_grad():
                init_pattern_embedding = self.model.token_embedding(tokenized_pattern_prompts).type(self.dtype)
            context = {"n_cls":1,"n_prompt_tokens_L":self.n_prompt_tokens_L,
                       "init_pattern_embedding":init_pattern_embedding, "tokenized_pattern_prompts":tokenized_pattern_prompts,"batch_size":self.batch_size,
                       "pop_size":self.popsize,"parallel":self.parallel}
        return context

    def get_image_information(self):
        context = {"n_prompt_tokens_V": self.n_prompt_tokens_V,
                   "batch_size": self.batch_size, "pop_size": self.popsize, "parallel": self.parallel}
        return context


    def generate_text_prompts(self,intrinsic_vectors,layer_id):
        prompt_list = []
        for vector in intrinsic_vectors:
            z = torch.tensor(vector,device=self.device,dtype=self.dtype)
            # [intrinsic_dim_L,] -> [n_prompt_token,ctx_dim]
            z = self.linear_L[layer_id](z).reshape(self.n_prompt_tokens_L,-1)
            if self.init_prompt is not None:
                z = z + self.init_prompt  # Az + p_0
            prompt_list.append(z)
        return prompt_list

    def generate_visual_prompts(self,intrinsic_vectors,layer_id):
        visual_prompt_list = []
        for vector in intrinsic_vectors:
            z = torch.tensor(vector,device=self.device,dtype=self.dtype)
            # [intrinsic_dim_L,] -> [n_prompt_token,ctx_dim]
            z = self.linear_V[layer_id](z).reshape(self.n_prompt_tokens_V,-1)
            visual_prompt_list.append(z)
        return visual_prompt_list
    def init_rest_linear_L(self,hidden_states_L):
        for i,h in enumerate(hidden_states_L[1:]):
            print('[Language Layer {} initializing]'.format(i+2))
            hidden = h.clone().reshape(-1).detach().cpu().numpy()
            hidden = hidden.astype(np.float32)
            mu_hat = np.mean(hidden)
            std_hat = np.std(hidden)
            max_h = np.max(hidden)
            min_h = np.min(hidden)
            print(' - Before clipping: mu=%.4f, std=%.4f, min=%.4f, max=%.4f' % (
                mu_hat, std_hat, min_h, max_h))
            # Clipping outliers
            clip_round = 0
            while clip_round < 5:
                clip_round += 1
                min_bound = mu_hat - 3 * std_hat
                max_bound = mu_hat + 3 * std_hat
                hidden = np.clip(hidden, min_bound, max_bound)
                mu_hat = np.mean(hidden)
                std_hat = np.std(hidden)
                max_h = np.max(hidden)
                min_h = np.min(hidden)
                print(' - After clipping (round %d): mu=%.4f, std=%.4f, min=%.4f, max=%.4f' % (
                    clip_round, mu_hat, std_hat, min_h, max_h))
            # Calculating std dev for the random projection
            mu = 0.0
            std = self.alpha * std_hat / (np.sqrt(self.intrinsic_dim_L) * self.sigma)
            # temp = intrinsic_dim - std_hat * std_hat
            # mu = mu_hat / temp
            # std = std_hat / np.sqrt(temp)
            print(' - Random Projection: mu=%.4f, std=%.4f' % (mu, std))
            for p in self.linear_L[i + 1].parameters():
                torch.nn.init.normal_(p, mu, std)
            self.intermediate_stats_L.append((mu, std))
        assert len(self.intermediate_stats_L) == self.num_prompt_layer

    def init_rest_linear_V(self,hidden_states_V):
        for i,h in enumerate(hidden_states_V[1:]):
            print('[Vision Layer {} initializing]'.format(i+2))
            hidden = h.clone().reshape(-1).detach().cpu().numpy()
            hidden = hidden.astype(np.float32)
            mu_hat = np.mean(hidden)
            std_hat = np.std(hidden)
            max_h = np.max(hidden)
            min_h = np.min(hidden)
            print(' - Before clipping: mu=%.4f, std=%.4f, min=%.4f, max=%.4f' % (
                mu_hat, std_hat, min_h, max_h))
            # Clipping outliers
            clip_round = 0
            while clip_round < 5:
                clip_round += 1
                min_bound = mu_hat - 3 * std_hat
                max_bound = mu_hat + 3 * std_hat
                hidden = np.clip(hidden, min_bound, max_bound)
                mu_hat = np.mean(hidden)
                std_hat = np.std(hidden)
                max_h = np.max(hidden)
                min_h = np.min(hidden)
                print(' - After clipping (round %d): mu=%.4f, std=%.4f, min=%.4f, max=%.4f' % (
                    clip_round, mu_hat, std_hat, min_h, max_h))
            # Calculating std dev for the random projection
            mu = 0.0
            std = self.alpha * std_hat / (np.sqrt(self.intrinsic_dim_V) * self.sigma)
            # temp = intrinsic_dim - std_hat * std_hat
            # mu = mu_hat / temp
            # std = std_hat / np.sqrt(temp)
            print(' - Random Projection: mu=%.4f, std=%.4f' % (mu, std))
            for p in self.linear_V[i + 1].parameters():
                torch.nn.init.normal_(p, mu, std)
            self.intermediate_stats_V.append((mu, std))
        assert len(self.intermediate_stats_V) == self.num_prompt_layer

    def metric(self,logits,label):
        ce_loss = F.cross_entropy(logits, label, reduction='none')
        final_loss = 0
        if self.loss_type == "ce":
            final_loss = torch.sum(ce_loss)
        elif self.loss_type == "focal":
            gamma = 2
            pt = torch.exp(-ce_loss)
            focal_loss = (1 - pt) ** gamma * ce_loss
            final_loss = torch.sum(focal_loss)
        return final_loss

    @torch.no_grad()
    def eval(self,prompt_zip,layer_id):
        prompt_text,prompt_image = prompt_zip[0],prompt_zip[1] # dtype = list if parallel else tuple
        prompts_text,prompts_image = self.use_best_prompt(prompt_text,prompt_image,layer_id)
        # (num_prompt_tokens, ctx_dim) -> (num_prompt_layer, num_prompt_token, ctx_dim)
        self.num_call += 1
        loss = 0
        if self.parallel:
            loss = [0]*self.popsize
            self.text_encoder.opt_layer = self.image_encoder.opt_layer = layer_id

        text_features,text_hidden_states = self.text_encoder(prompts_text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        if len(self.intermediate_stats_L)==1:
            self.init_rest_linear_L(text_hidden_states)
        for batch in self.train_loader:
            image,label = self.parse_batch(batch)
            image_features,image_hidden_states = self.image_encoder(image,prompts_image)
            image_features = image_features / image_features.norm(dim=-1,keepdim=True)

            if len(self.intermediate_stats_V)==1:
                self.init_rest_linear_V(image_hidden_states)

            logit_scale = self.logit_scale.exp()
            if self.parallel:
                B = int(image_features.shape[0]/self.popsize)
                for i in range(self.popsize):
                    start_text = i * self.n_cls
                    start_image = i * B
                    tmp_text_features = text_features[start_text:start_text+self.n_cls]
                    tmp_image_features = image_features[start_image:start_image+B]
                    tmp_logits =  logit_scale*tmp_image_features@tmp_text_features.t()
                    loss[i]+=self.metric(tmp_logits,label)
            else:
                logits = logit_scale*image_features@text_features.t()
                loss +=self.metric(logits,label)

        epoch_min_loss = None
        if self.parallel:
            loss = [x/len(self.train_data) for x in loss]
            epoch_min_loss = min(loss)
        else:
            loss /= len(self.train_data)
            epoch_min_loss = loss if epoch_min_loss == None else min(loss,epoch_min_loss)
        self.loss.append(loss)

        if self.min_loss is None or epoch_min_loss<self.min_loss:
            self.min_loss = epoch_min_loss
            if self.parallel:
                index = loss.index(epoch_min_loss)
                self.store_best_prompt(prompt_text[index], prompt_image[index], layer_id)
            else:
                self.store_best_prompt(prompt_text,prompt_image,layer_id)


        #num_call = self.num_call * self.popsize if self.parallel else self.num_call
        if self.num_call % self.test_every == 0:
            acc = self.test()
            self.acc.append(acc)
            self.best_accuracy = max(acc,self.best_accuracy)
            output_dir = os.path.join(self.output_dir,self.task_name)
            fname = "{}_{}_{}.pth".format(self.task_name, self.opt_name, self.backbone.replace("/","-"))
            content = {"task_name":self.task_name,"opt_name":self.opt_name,"backbone":self.backbone,"best_accuracy":self.best_accuracy,"acc":self.acc,
                       "best_prompt_text":self.best_prompt_text,"best_prompt_image":self.best_prompt_image,"loss":self.loss,"num_call":self.num_call,
                       "Linear_L":self.linear_L.state_dict(),"Linear_V":self.linear_V.state_dict()}
            Analysis_Util.save_results(content,output_dir,fname)
            # ---------------save_results-----------------------------------
        return loss

    @torch.no_grad()
    def test(self):
        correct = 0.
        parallel = self.parallel
        self.parallel=self.text_encoder.parallel = self.image_encoder.parallel = False
        for batch in self.test_loader:
            image,label = self.parse_batch(batch)
            text_features,text_hidden_states = self.text_encoder(self.best_prompt_text)
            image_features,image_hidden_states = self.image_encoder(image,self.best_prompt_image)

            image_features = image_features / image_features.norm(dim=-1,keepdim=True)
            text_features = text_features / text_features.norm(dim=-1,keepdim=True)
            logit_scale = self.logit_scale.exp()
            logits = logit_scale*image_features@text_features.t()
            prediction = logits.argmax(dim=-1)
            correct += (prediction == label).float().sum()
        acc = correct/len(self.test_data)
        self.parallel = self.text_encoder.parallel = self.image_encoder.parallel = parallel
        return acc

    def load_dataset(self):
        if self.task_name == 'CIFAR100':
            self.dataset = CIFAR100(os.path.expanduser("~/.cache"), transform=self.preprocess, download=True)
            self.classes = self.dataset.classes
            self.n_cls = len(self.classes)
            self.train_data,self.train_loader = load_train_cifar100(batch_size=self.batch_size,shots=self.k_shot,preprocess=self.preprocess)
            self.test_data, self.test_loader = load_test_cifar100(batch_size=self.batch_size, preprocess=self.preprocess)
        elif self.task_name == 'StanfordCars':
            self.train_data,self.train_loader = load_train(batch_size=self.batch_size,seed=self.seed,shots=self.k_shot,preprocess=self.preprocess,
                                                           root=self.data_dir,dataset_dir="Cars_Gen")
            self.test_data,self.test_loader = load_test(batch_size=self.batch_size,preprocess=self.preprocess,
                                                           root=self.data_dir,dataset_dir="Cars_Gen")
            self.classes = self.train_data.classes
            self.n_cls = len(self.classes)
        elif self.task_name == 'OxfordPets':
            self.train_data,self.train_loader = load_train(batch_size=self.batch_size,seed=self.seed,shots=self.k_shot,preprocess=self.preprocess,
                                                           root=self.data_dir,dataset_dir="OxfordPets_Gen")
            self.test_data,self.test_loader = load_test(batch_size=self.batch_size,preprocess=self.preprocess,
                                                           root=self.data_dir,dataset_dir="OxfordPets_Gen")
            self.classes = self.train_data.classes
            self.n_cls = len(self.classes)
        elif self.task_name == 'UCF-101':
            self.train_data,self.train_loader = load_train(batch_size=self.batch_size,seed=self.seed,shots=self.k_shot,preprocess=self.preprocess,
                                                           root=self.data_dir,dataset_dir="UCF-101_Gen")
            self.test_data,self.test_loader = load_test(batch_size=self.batch_size,preprocess=self.preprocess,
                                                           root=self.data_dir,dataset_dir="UCF-101_Gen")
            self.classes = self.train_data.classes
            self.n_cls = len(self.classes)
        elif self.task_name == 'DTD':
            self.train_data,self.train_loader = load_train(batch_size=self.batch_size,seed=self.seed,shots=self.k_shot,preprocess=self.preprocess,
                                                           root=self.data_dir,dataset_dir="DTD_Gen")
            self.test_data,self.test_loader = load_test(batch_size=self.batch_size,preprocess=self.preprocess,
                                                           root=self.data_dir,dataset_dir="DTD_Gen")
            self.classes = self.train_data.classes
            self.n_cls = len(self.classes)
        elif self.task_name == 'EuroSAT':
            self.train_data,self.train_loader = load_train(batch_size=self.batch_size,seed=self.seed,shots=self.k_shot,preprocess=self.preprocess,
                                                           root=self.data_dir,dataset_dir="EuroSAT_Gen")
            self.test_data,self.test_loader = load_test(batch_size=self.batch_size,preprocess=self.preprocess,
                                                           root=self.data_dir,dataset_dir="EuroSAT_Gen")
            self.classes = self.train_data.classes
            self.n_cls = len(self.classes)
        elif self.task_name == 'Food101':
            self.train_data,self.train_loader = load_train(batch_size=self.batch_size,seed=self.seed,shots=self.k_shot,preprocess=self.preprocess,
                                                           root=self.data_dir,dataset_dir="Food101_Gen")
            self.test_data,self.test_loader = load_test(batch_size=self.batch_size,preprocess=self.preprocess,
                                                           root=self.data_dir,dataset_dir="Food101_Gen")
            self.classes = self.train_data.classes
            self.n_cls = len(self.classes)

        elif self.task_name == 'caltech101':
            self.train_data,self.train_loader = load_train(batch_size=self.batch_size,seed=self.seed,shots=self.k_shot,preprocess=self.preprocess,
                                                           root=self.data_dir,dataset_dir="caltech101_Gen")
            self.test_data,self.test_loader = load_test(batch_size=self.batch_size,preprocess=self.preprocess,
                                                           root=self.data_dir,dataset_dir="caltech101_Gen")
            self.classes = self.train_data.classes
            self.n_cls = len(self.classes)
        elif self.task_name == 'SUN397':
            self.train_data,self.train_loader = load_train(batch_size=self.batch_size,seed=self.seed,shots=self.k_shot,preprocess=self.preprocess,
                                                           root=self.data_dir,dataset_dir="SUN397_Gen")
            self.test_data,self.test_loader = load_test(batch_size=self.batch_size,preprocess=self.preprocess,
                                                           root=self.data_dir,dataset_dir="SUN397_Gen")
            self.classes = self.train_data.classes
            self.n_cls = len(self.classes)
        elif self.task_name == 'ImageNet':
            self.train_data,self.train_loader = load_train(batch_size=self.batch_size,seed=self.seed,shots=self.k_shot,preprocess=self.preprocess,
                                                           root=self.data_dir,dataset_dir="imagenet")
            self.test_data,self.test_loader = load_test(batch_size=self.batch_size,preprocess=self.preprocess,
                                                           root=self.data_dir,dataset_dir="imagenet")
            self.classes = self.train_data.classes
            self.n_cls = len(self.classes)


    def parse_batch(self,batch):
        image = batch["image"]
        label = batch["label"]
        image = image.to(device=self.device, dtype=self.dtype)
        label = label.to(device=self.device)
        if self.parallel:
            image = image.repeat(self.popsize, 1, 1, 1)
        return image, label

