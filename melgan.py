import torch
from pqmf import PQMF
from modules import Generator, MelEncoder


class MelGAN(torch.nn.Module):
    """Multi-band MelGAN vocoder."""
    
    def __init__(self, mels=80, channels=384, bands=4, 
                 cutoff_ratio=0.148, beta=9.0, length=63, 
                 hop=200, nwin=800, nfft=1024, sr=16000):
        """Creates multi-band MelGAN vocoder."""

        super(MelGAN, self).__init__()

        self.G = Generator(mels, channels, bands)
        self.pqmf = PQMF(cutoff_ratio, beta, length, bands)
        self.mel = MelEncoder(mels, hop, nwin, nfft, sr)
                
    def encode(self, waveform):
        """Converts the waveform into a mel-spectrogram."""
        assert waveform.ndim in [1, 2]
        return self.mel(waveform)
        
    def decode(self, melspec):
        """Converts the mel-spectrogram into a waveform."""
        assert melspec.ndim in [2,3]
        assert melspec.size(-2) == self.mel.mel.size(1)
        x = melspec.unsqueeze(0) if melspec.ndim == 2 else melspec
        
        x = self.G(x)
        x = (self.pqmf.inverse(x) if self.G.bands > 1 else x).squeeze(1)
        
        return x.squeeze(0) if melspec.ndim == 2 else x
    
    def forward(self, melspec):
        # for end-to-end TTS-models
        return self.decode(melspec)
    

def from_config(config: dict) -> MelGAN:
    """Creates multi-band MelGAN vocoder from config."""
    
    pqmf = config["pqmf"]
    mg = config["melgan"]
    mel = config["mel"]
    sr = config["data"]["sample_rate"]
    
    return MelGAN(mel["mels"], mg["channels"], mg["bands"], 
                  pqmf["cutoff_ratio"], pqmf["beta"], pqmf["length"],
                  mel["hop"], mel["nwin"], mel["nfft"], sr)
    

if __name__ == "__main__":
    # test
    from librosa.feature import melspectrogram as librosa_melspec
    
    wav = torch.randn(2,16000)
    #wav = torch.randn(16000)
    
    #wav.unfold(-1, 800, 200).transpose(-2,-1).shape
    
    mg = MelGAN(nfft=800)
    #mg, wav = mg.to("cuda:0"), wav.to("cuda:0")
    
    # encoding test:
    x1 = librosa_melspec(y=wav.cpu().numpy(), sr=16000, n_fft=800, n_mels=80, 
                         hop_length=200, win_length=800, center=False)
    x2 = mg.encode(wav).cpu()
    
    assert torch.allclose(torch.from_numpy(x1), x2[..., :x1.shape[-1]], 
                          atol=1e-3, rtol=0)
   
    # decoding test:
    import matplotlib.pyplot as plt
    import sounddevice as sd
    import librosa
    import yaml
    
    config_path = "models/melgan.yaml"
    model_path = "models/melgan.pt"
    
    cfg = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    sr = cfg["data"]["sample_rate"]
    vocoder = from_config(cfg)
    vocoder.G.load_state_dict(torch.load(model_path))
    
    # out-of-distribution sample (female)
    x = torch.from_numpy(librosa.load(librosa.example("libri3"), sr=sr)[0])
    x = x[:int(sr*5.5)]
    
    y = vocoder.encode(x)
    with torch.no_grad():
        x_hat = vocoder.decode(y)

    sd.play(x, sr, blocking=True) # original
    sd.play(x_hat, sr, blocking=True) # restored
    
    plt.plot(x    [15000: 17000], label="original")
    plt.plot(x_hat[15000: 17000], label="restored")
    plt.title("Original and restored waveforms")
    plt.legend()
    plt.show()