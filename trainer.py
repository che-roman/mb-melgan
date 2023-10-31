import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from pqmf import PQMF
from modules import Generator, Discriminator, MelEncoder, STFTTotalLoss, \
                    GeneratorLoss, DiscriminatorLoss


class ModelOptimizerWrap(nn.Module):
    
    def __init__(self, model, start_iter=0, lr=1e-4, betas=(0.5, 0.9)):
        
        super(ModelOptimizerWrap, self).__init__()
        milestones = [start_iter + 100000*(i+1) for i in range(6)]
        
        self.model = model
        self.opt  = Adam(model.parameters(), lr=lr, betas=betas)
        self.sched  = MultiStepLR(self.opt, milestones, gamma=0.5)
    
    def forward(self, x):
        return self.model(x)
    
    def update(self, loss):
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.sched.step()
    
    def state_dict(self):
        self.opt.zero_grad()
        state = {"model": self.model.state_dict(),
                 "opt"  : self.opt.state_dict(),
                 "sched": self.sched.state_dict()}
        return state
    
    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict["model"])
        self.opt.load_state_dict  (state_dict["opt"])
        self.sched.load_state_dict(state_dict["sched"])
        

class Trainer():
    
    def __init__(self, G: ModelOptimizerWrap, D: ModelOptimizerWrap, 
                 mel: MelEncoder, pqmf: PQMF, lambda_param=2.5):
        
        super(Trainer, self).__init__()
        
        self.G = G
        self.D = D
        self.mel = mel
        self.pqmf = pqmf
        
        self.G_loss = GeneratorLoss()
        self.D_loss = DiscriminatorLoss()
        self.T_loss = STFTTotalLoss(G.model.bands > 1)
        
        self.lambda_param = lambda_param
                        
    def pretrain(self, wav):
        """Generator pre-training procedure."""

        wav, mel = self._prepare_input(wav)
        
        stft_loss = self._G_pretrain(wav, mel)

        return stft_loss
    
    def train(self, wav):
        """GAN training procedure."""

        wav, mel = self._prepare_input(wav)
                        
        D_loss            = self._D_train(wav, mel)
        G_loss, stft_loss = self._G_train(wav, mel)
        
        return stft_loss, D_loss, G_loss
    
    def _prepare_input(self, wav):

        assert wav.ndim == 2
        mel = self.mel(wav)
        wav = wav.unsqueeze(1)

        return wav, mel
    
    def _predict(self, mel):

        band_pred = self.G(mel)
        full_pred = band_pred 
        
        if self.G.model.bands > 1:
            full_pred = self.pqmf.inverse(band_pred)

        return full_pred, band_pred
    
    def _G_pretrain(self, wav, mel):

        full_pred, band_pred = self._predict(mel)
        band_real = self.pqmf(wav) if self.G.model.bands>1 else None

        stft_loss = self.T_loss(wav, full_pred, band_real, band_pred)
        self.G.update(stft_loss)

        return stft_loss.detach()

    def _G_train(self, wav, mel):

        full_pred, band_pred = self._predict(mel)
        band_real = self.pqmf(wav) if self.G.model.bands>1 else None

        stft_loss = self.T_loss(wav, full_pred, band_real, band_pred)
        G_loss_mean = self.G_loss(self.D(full_pred))
        G_loss = G_loss_mean * self.lambda_param + stft_loss
        
        self.G.update(G_loss)
        return G_loss_mean.detach(), stft_loss.detach()
        
    def _D_train(self, wav, mel):
        
        with torch.no_grad():
            pred, _ = self._predict(mel)
        
        D_loss = self.D_loss(self.D(wav), self.D(pred.detach()))
        self.D.update(D_loss)
        
        return D_loss.detach()
    
    def to(self, device):
        self.G.to(device)
        self.D.to(device)
        self.pqmf.to(device)
        self.mel.to(device)
        self.T_loss.to(device)
        return self
    
    def state_dict(self):
        return {"G": self.G.state_dict(), "D": self.D.state_dict()}
        
    def load_state_dict(self, state_dict):
        self.G.load_state_dict(state_dict["G"])
        self.D.load_state_dict(state_dict["D"])
    
    
def from_config(config: dict) -> Trainer:
    """Creates Trainer from config."""
    
    sr = config["data"]["sample_rate"]
    mel = config["mel"]
    melgan = config["melgan"]
    pqmf = config["pqmf"]
    lr = float(config["learning_rate"])
    
    G = Generator(mel["mels"],  melgan["channels"], melgan["bands"])
        
    return Trainer(
        ModelOptimizerWrap(G, config["iters_pretrain"], lr),
        ModelOptimizerWrap(Discriminator(), lr=lr),
        MelEncoder(mel["mels"], mel["hop"], mel["nwin"], mel["nfft"], sr),
        PQMF(pqmf["cutoff_ratio"], pqmf["beta"], pqmf["length"], melgan["bands"]))
    

if __name__ == "__main__":
    # test
    import time
    
    config_path = "config/mb_train.yaml"
    
    import yaml
    cfg = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    m = from_config(cfg)
    bs = cfg["data"]["batch_size"]
    device = cfg["device"]

    # bands = 4 #1
    # channels = 384 #512
    # bs = 128 #16
    # device = "cuda:0" #"cpu"
    
    # m = Trainer(ModelOptimizerWrap(Generator(80, channels, bands)),
    #             ModelOptimizerWrap(Discriminator()), 
    #             MelEncoder(80, 200, 800, 1024, 16000), 
    #             PQMF(0.15, 9.0, 63, bands))
    
    m.to(device)

    print("losses:")
    t0 = time.time()
    for iters in range(10):
        x = torch.randn(bs, 16000).to(device)
        # losses = (m.pretrain(x), )
        losses = m.train(x)
    t1 = time.time()
    print([l.item() for l in losses])
    print("elapsed time: {:.2f}".format(t1-t0))
        