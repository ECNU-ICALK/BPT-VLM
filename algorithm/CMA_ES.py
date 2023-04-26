import cma
class shallow_cma:
    def __init__(self,cfg):
        self.opt_setting = {
            'seed': cfg["seed"],
            'popsize': cfg["popsize"],
            'maxiter': cfg["budget"] / cfg["popsize"],
            'verbose': -1,
        }
        if cfg["bound"] > 0:
            self.opt_setting['bounds'] = [-1 * cfg["bound"], 1 * cfg["bound"]]
        self.intrinsic_dim_L = cfg["intrinsic_dim_L"]
        self.intrinsic_dim_V = cfg["intrinsic_dim_V"]
        self.sigma = cfg["sigma"]
        self.es = cma.CMAEvolutionStrategy((self.intrinsic_dim_L + self.intrinsic_dim_V) * [0], self.sigma , inopts=self.opt_setting)

    def ask(self):
        return self.es.ask()

    def tell(self,solutions, fitnesses):
        return self.es.tell(solutions,fitnesses)

    def stop(self):
        return self.es.stop()


class deep_cma:
    def __init__(self,cfg):
        self.opt_setting = {
            'seed': cfg["seed"],
            'popsize': cfg["popsize"],
            'maxiter': cfg["maxiter"],
            'verbose': -1,
        }
        if cfg["bound"] > 0:
            self.opt_setting['bounds'] = [-1 * cfg["bound"], 1 * cfg["bound"]]
        self.intrinsic_dim_L = cfg["intrinsic_dim_L"]
        self.intrinsic_dim_V = cfg["intrinsic_dim_V"]
        self.sigma = cfg["sigma"]
        self.es_list = [cma.CMAEvolutionStrategy((self.intrinsic_dim_L + self.intrinsic_dim_V) * [0],
                                                 self.sigma , inopts=self.opt_setting) for i in range(cfg['num_prompt_layer'])]

    def ask(self,prompt_id):
        return self.es_list[prompt_id].ask()


    def tell(self,solutions, fitnesses,prompt_id):
        return self.es_list[prompt_id].tell(solutions,fitnesses)


