import torch
from torch.utils.tensorboard import SummaryWriter
from torchaudio.functional import resample, spectrogram, amplitude_to_DB
from cmap import Colormap
import atexit
import yaml
import os
import melgan
from context import Context
from data import test_loader


class Logger:
    """Class for tracking and serializing the training process."""
    
    TRAIN_CFG_FNAME = "train.yaml"
    CHECKPOINT_FNAME = "context.pt"
    RESULTS_TRACKING_DIR = "results"
    MELGAN_DIR = "models"
    MELGAN_CFG_FNAME = "melgan.yaml"
    MELGAN_FNAME_FMT = lambda iter: f"melgan_iter{iter}.pt"
    AUDIO_ORIG_FMT = lambda name: f"{name}_wav/orig"
    AUDIO_PRED_FMT = lambda name: f"{name}_wav/pred"
    SPEC_ORIG_FMT = lambda name: f"{name}_spec/orig"
    SPEC_PRED_FMT = lambda name: f"{name}_spec/pred"
    
    TRACK_PRINT = lambda i, d, stft: f"iter: {i} | dur: {d:.2f} sec | stft-loss: {stft:.4f}"
    TRACK_FULL_PRINT = lambda i, d, stft, D, G: \
        f"iter: {i} | dur: {d:.2f} sec | stft-loss: {stft:.4f} | D-loss: {D:.4f} | G-loss: {G:.4f}"
    
        
    def __init__(self, path):
        self.initialized = False
        self.path = path
                    
    def _exit_handler(self):
        """Executed when the program ends or is interrupted by 
        pressing the "CTRL+C"."""
        # self.save_checkpoint()
        self.writer.close()
        
    def create(self, config: dict, ctx: Context) -> None:
        """Creates a new folder for tracking and serializing 
        the training process."""
        assert not self.initialized
        assert not os.path.isdir(self.path)
        assert not config is None
        assert not ctx is None
        
        loader = test_loader(config)
        
        # create folder for models
        os.makedirs(os.path.join(self.path, self.MELGAN_DIR))
        # save melgan-config
        mg_fn = os.path.join(self.path, self.MELGAN_DIR, self.MELGAN_CFG_FNAME)
        yaml.dump(melgan_config(config), open(mg_fn, 'w'))
                
        # save train-config
        yaml.dump(config, open(os.path.join(self.path, self.TRAIN_CFG_FNAME), 'w'))
        
        # save checkpoint (context)
        ctx_path = os.path.join(self.path, self.CHECKPOINT_FNAME)
        torch.save(ctx.state_dict(), ctx_path)
        
        # run logger and save test-files
        results_path = os.path.join(self.path, self.RESULTS_TRACKING_DIR)
        self.writer = SummaryWriter(results_path, flush_secs=10)
        sr = config["data"]["sample_rate"]
        for x, name in loader:
            x, name = x[0], name[0]
            x = resample(x, loader.dataset.sr, sr)
            X = spec_img(x, 512, 128)
            self.writer.add_audio(Logger.AUDIO_ORIG_FMT(name), x, 0, sr)
            self.writer.add_image(Logger.SPEC_ORIG_FMT(name), X, 0, None, 'HWC')
        self.writer.flush()
                
        self.config = config
        self.ctx = ctx
        self.loader = loader
        self.initialized = True
        
        print(f"folder created: '{self.path}'.")
        atexit.register(self._exit_handler)
        
    def load(self) -> (dict, Context):
        """Loads the state of the training process from an existing folder."""
        assert not self.initialized
        assert os.path.isdir(self.path)
        
        # load config
        cfg_path = os.path.join(self.path, self.TRAIN_CFG_FNAME)
        config = yaml.load(open(cfg_path, "r"), Loader=yaml.FullLoader)
        
        # load checkpoint (context)
        ctx_path = os.path.join(self.path, self.CHECKPOINT_FNAME)
        ctx = Context(config)
        ctx.trainer.to(config["device"])
        ctx.load_state_dict(torch.load(ctx_path))
        
        loader = test_loader(config)
        
        results_path = os.path.join(self.path, self.RESULTS_TRACKING_DIR)
        self.writer = SummaryWriter(results_path, flush_secs=10)
        
        self.config = config
        self.ctx = ctx
        self.loader = loader
        self.initialized = True

        print(f"checkpoint loaded from: '{ctx_path}'.")
        atexit.register(self._exit_handler)
        
        return config, ctx
                
    def save_checkpoint(self):
        """Saves checkpoint."""
        assert self.initialized
        
        ctx_path = os.path.join(self.path, self.CHECKPOINT_FNAME)
        torch.save(self.ctx.state_dict(), ctx_path)
        print(f"checkpoint saved in: '{ctx_path}'.")
    
    def save_results(self):
        """Saves the melgan-model, generated test samples and 
        their spectrograms."""
        assert self.initialized
        i = self.ctx.total_iters
        state = self.ctx.trainer.G.model.to("cpu").state_dict()
        mg_path = os.path.join(self.path, self.MELGAN_DIR, 
                               Logger.MELGAN_FNAME_FMT(i))
        # save melgan
        torch.save(state, mg_path)
        print(f"melgan saved in: '{mg_path}'.")
        
        vododer = melgan.from_config(self.config)
        vododer.G.load_state_dict(state)
        
        # save generated test samples
        sr = self.config["data"]["sample_rate"]
        for x, name in self.loader: 
            x, name = x[0], name[0]
            x = resample(x, self.loader.dataset.sr, sr)
            x = vododer.decode(vododer.encode(x))
            X = spec_img(x, 512, 128)
            self.writer.add_audio(Logger.AUDIO_PRED_FMT(name), x, i, sr)
            self.writer.add_image(Logger.SPEC_PRED_FMT(name), X, i, None, 'HWC')
        self.writer.flush()
        
        self.ctx.trainer.G.model.to(self.config["device"])
    
    def track_losses(self, dt, stft_loss, D_loss=None, G_loss=None):
        assert self.initialized
        i = self.ctx.total_iters
        self.writer.add_scalar("STFT_loss", stft_loss, i)
        if not D_loss is None and not G_loss is None:
            self.writer.add_scalar("D_loss", D_loss, i)
            self.writer.add_scalar("G_loss", G_loss, i)
            print(Logger.TRACK_FULL_PRINT(i, dt, stft_loss, D_loss, G_loss))
        else:
            print(Logger.TRACK_PRINT(i, dt, stft_loss))
        self.writer.flush()
        
        
def spec_img(x, nfft=512, hop=128):
    """
    Converts the waveform into a power spectrogram of the "viridis" color map.
    """
    assert x.ndim == 1
    
    norm = lambda x: (x-x.min())/(x.max()-x.min())
    color = lambda x, cm: cm[(norm(x) * (len(cm)-1)).long()]
    
    cm = torch.tensor(Colormap('viridis').info.data)
    win = torch.hann_window(nfft)
    X = spectrogram(x, 0, win, nfft, hop, len(win), 2, False)
    
    return color(amplitude_to_DB(X, 10.0, 1e-10, 0), cm).flip(0)


def melgan_config(config: dict) -> dict:
    """Creates a melgan-config (for the vocoder) from the training config."""
    res = {"data": { "dataset": config["data"]["dataset"], 
                     "sample_rate": config["data"]["sample_rate"]}}
    res["mel"] = config["mel"]
    res["melgan"] = config["melgan"]
    res["pqmf"] = config["pqmf"]
    return res
