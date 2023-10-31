import torch
from torch.utils.data import Dataset, Sampler, SequentialSampler, DataLoader
import torchaudio
from typing import Dict, List
import csv
import os
#from tqdm import tqdm
from context import Context


class DatasetBasic(Dataset):
    """Basic abstract dataset."""
    
    # def __init__(self, test_set: Dict[str, str], mode: str, sr: int):
    #     assert isinstance(test_set, dict)
    #     assert mode in ["test", "train"]
        
    #     self.test_set = test_set # {name: relative_path}
    #     self.sr = sr
    #     self.mode = mode
    
    def __init__(self, mode: str, sr: int):
        assert mode in ["test", "train"]
        self.sr = sr
        self.mode = mode
    
    def weights(self):
        """Returns a list of sample weights."""
        assert self.mode == "train"
        # by default - uniformly distributed weights 
        return [1./len(self)]*len(self)
    
    # def load_test(self, name: str):
    #     assert name in self.test_set.keys()
    #     fn = os.path.join(self.path, "wavs", f"{self.test_set[name]}.wav")
    #     return torchaudio.load(fn)[0][0]
    
    # def test_list(self):
    #     return self.test_set.keys()


class LJSpeech(DatasetBasic):
    
    def __init__(self, path: str, test_set: Dict[str, str] = {}, 
                 mode: str="test", crop_len: int=0, 
                 generator: torch.Generator=None):
        
        # super(LJSpeech, self).__init__(test_set, mode, 22050)
        super(LJSpeech, self).__init__(mode, 22050)
        
        self.path = path
        
        if mode == "train":
            self.crop_len = crop_len
            self.generator = generator
            self._create_train_files(test_set)
        
        else:
            self._create_test_files(test_set)
    
    def _create_train_files(self, test_set):
        
        self.files = []
        metadata_fn = os.path.join(self.path, "metadata.csv")
        
        print("loading metadata... (may take a few minutes)")
        
        with open(metadata_fn, newline="") as f:
            reader = csv.reader(f, delimiter="|", quoting=csv.QUOTE_NONE)
            for id,_,_ in reader: #tqdm(reader)
                fn = os.path.join(self.path, "wavs", f"{id}.wav")
                num_frames = torchaudio.info(fn).num_frames
                
                # removing too short waveforms from the filelist
                if num_frames >= self.crop_len and not id in test_set.values():
                    self.files.append([id, num_frames])
                    
    def _create_test_files(self, test_set):
        self.files = []
        for name, id in test_set.items():
            self.files.append([id, name])
                
    def __getitem__(self, idx):
        
        if self.mode == "train":
            id, num_frames = self.files[idx]
            fn = os.path.join(self.path, "wavs", f"{id}.wav")
            
            i = torch.randint(high=num_frames-self.crop_len, size=[1], 
                              generator=self.generator).item()
            
            return torchaudio.load(fn, i, self.crop_len)[0][0]
        
        else:
            id, name = self.files[idx]
            fn = os.path.join(self.path, "wavs", f"{id}.wav")

            return torchaudio.load(fn)[0][0], name
            
                
    def __len__(self):
        return len(self.files)
    
    def weights(self):
        assert self.mode == "train"
        
        weights = [num_frames for _, num_frames in self.files]
        weights = torch.tensor(weights, dtype=torch.float)
        
        return (weights / weights.sum()).tolist()


class WeightedSampler(Sampler[List[int]]):
    """Weighted batch sampler."""
    
    def __init__(self, batch_size: int, num_batches: int,
                 weights: torch.Tensor, 
                 generator: torch.Generator=None):

        self.batch_size = batch_size
        self.num_batches = num_batches
        self.rnd = generator
        self.weights = weights
        
    def __iter__(self):
        for i in range(self.num_batches):
            yield torch.multinomial(self.weights, self.batch_size, True, 
                                    generator=self.rnd).tolist()
            
    def __len__(self):
        return self.num_batches
    
    
def train_loader(config: dict, iters: int, ctx: Context) -> DataLoader:
    """Creates a training DataLoader from the config."""

    cfg = config["data"]
    assert cfg["dataset"] in ["LJSpeech"]
    
    # dataset creating
    if cfg["dataset"] == "LJSpeech":
        crop_len = (22050 * cfg["crop_len"]) // cfg["sample_rate"]
        ds = LJSpeech(cfg["path"], cfg["test_wav"], "train", 
                      crop_len, ctx.rnd_ds)
    else:
        raise NotImplementedError("Unidentified dataset.")
    
    # sampler and loader creating
    sampler = WeightedSampler(cfg["batch_size"], iters, 
                              torch.Tensor(ds.weights()), ctx.rnd_sampler)
    loader = DataLoader(ds, batch_sampler=sampler,
                        # pin_memory=True, pin_memory_device=device,
                        num_workers=cfg["num_workers"],
                        generator=ctx.rnd_loader)
    
    return loader
        

def test_loader(config: dict) -> DataLoader:
    """Creates a testing DataLoader from the config."""

    cfg = config["data"]
    assert cfg["dataset"] in ["LJSpeech"]
    
    # dataset creating
    if cfg["dataset"] == "LJSpeech":
        ds = LJSpeech(cfg["path"], cfg["test_wav"], "test")
    else:
        raise NotImplementedError("Unidentified dataset.")
    
    return DataLoader(ds, sampler=SequentialSampler(ds))


if __name__ == "__main__":
    # test
    import yaml
    import time
    
    config_path = "config/mb_train.yaml"
    
    rnd = torch.Generator()
    
    t = time.time()
    cfg = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    
    ctx = Context(cfg)
    loader = train_loader(cfg, 10, ctx)
         
    print("start")
    t0 = time.time()
    a = 0
    for i,x in enumerate(loader):
        a += x.pow(2).mean()
        # time.sleep(0.001)
        # print(x[2:4,3:5])
    t1 = time.time()
    print("elapsed time: {:.2f}".format(t1-t0))
    print("meta: {:.2f}".format(t0-t))