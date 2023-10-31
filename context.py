import torch
import trainer


class Context:
    """
    Contains complete information about current state of the training process.
    """
    def __init__(self, config: dict):
        self.trainer = trainer.from_config(config)
        self.rnd_ds = torch.Generator() # generator for dataset
        self.rnd_sampler = torch.Generator() # generator for sampler
        self.rnd_loader = torch.Generator() # generator for loader
        self.total_iters = 0
        
        self.rnd_ds.manual_seed(config["seed"])
        self.rnd_sampler.manual_seed(config["seed"])
        self.rnd_loader.manual_seed(config["seed"])
                
    def state_dict(self):
        state = {"trainer": self.trainer.state_dict(),
                 "rnd_ds": self.rnd_ds.get_state(),
                 "rnd_sampler": self.rnd_sampler.get_state(),
                 "rnd_loader": self.rnd_loader.get_state(),
                 "total_iters": self.total_iters}
        return state
    
    def load_state_dict(self, state_dict):
        self.trainer.load_state_dict(state_dict["trainer"])
        self.rnd_ds.set_state(state_dict["rnd_ds"])
        self.rnd_sampler.set_state(state_dict["rnd_sampler"])
        self.rnd_loader.set_state(state_dict["rnd_loader"])
        self.total_iters = state_dict["total_iters"]
        