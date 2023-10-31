import torch
from torch import nn
from torchaudio.functional import melscale_fbanks
import torch.nn.functional as F
from torch.nn.utils import weight_norm as wn


def mel(n_fft, n_mels, sr):
    """Returns a bank of mel filters (compatible with 'librosa.filters.mel')."""
    return melscale_fbanks(sample_rate=sr, n_freqs=n_fft//2 + 1, n_mels=n_mels, 
                           norm="slaney", mel_scale="slaney", 
                           f_min=0., f_max=float(sr)/2)
# # test mel:
# import librosa
# m1 = torch.from_numpy(librosa.filters.mel(sr=16000, n_fft=1024, n_mels=80))
# m2 = mel(sr=16000, n_fft=1024, n_mels=80)
# assert torch.allclose(m1.T, m2, atol=1e-4, rtol=0)
    

def stft(x: torch.Tensor, hop: int, nfft: int, win: torch.Tensor) -> torch.Tensor:
    """Computes a scipy-compatible spectrogram of the signal."""
    # librosa.stft and torch.stft return the incorrect 
    # number of frames when n_fft != win_lenght
    assert win.ndim == 1
    assert x.size(-1) >= win.size(0)
    
    frames = x.unfold(-1, win.size(0), hop)
    X = torch.fft.fft((frames) * win, nfft)
    
    return X[..., :nfft//2+1].transpose(-2,-1)


class STFT(nn.Module):
    """Module for calculating the spectrogram of the signal."""
    
    def __init__(self, hop, nwin, nfft):
        super(STFT, self).__init__()

        self.register_buffer("win", torch.hann_window(nwin), False)
        self.forward = lambda x: stft(x, hop, nfft, self.win)


class MelEncoder(nn.Module):
    """Module for calculating the mel-spectrogram of the signal."""
    
    def __init__(self, mels, hop, nwin, nfft, sr):
        super(MelEncoder, self).__init__()
        
        self.stft = STFT(hop, nwin, nfft)
        self.delta = nwin - hop
        self.register_buffer("mel", mel(sr=sr, n_fft=nfft, n_mels=mels), False)
        
    def forward(self, x):
        x = F.pad(x, (0, self.delta), "constant", 0)# end-pad
        return (self.stft(x).pow(2).abs().transpose(-2,-1) @ self.mel).transpose(-2,-1)


class ResBlock(nn.Module):
    
    def __init__(self, channels, dilation):
        super(ResBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReplicationPad1d(dilation), # not deterministic backward for cuda
            nn.LeakyReLU(0.2),
            wn(nn.Conv1d(channels, channels, 3, dilation=dilation, bias=False)),
            nn.LeakyReLU(0.2),
            wn(nn.Conv1d(channels, channels, 1, bias=False)))
        
        # self.shortcut = wn(nn.Conv1d(channels, channels, kernel_size=1, bias=False))
        
    def forward(self, x):
        return self.block(x) + x
        #return self.block(x) + self.shortcut(x)


class ResStack(nn.Module):
    
    def __init__(self, channels: int, dilations: list=[1,3,9,27]):
        super(ResStack, self).__init__()

        blocks = [ResBlock(channels, d) for d in dilations]
        self.blocks = nn.Sequential(*blocks)
        
    def forward(self, x):
        return self.blocks(x)


class Upsample(nn.Module):
    
    def __init__(self, channels, factor):
        super(Upsample, self).__init__()
        self.factor = factor
        
        self.upsample = wn(nn.ConvTranspose1d(channels, channels//2, factor*2, 
                                              stride=factor, bias=False))
                    
        self.resstack = nn.Sequential(
            ResStack(channels // 2, dilations=[1,3,9,27]),
            nn.LeakyReLU(0.2))
        
    
    def forward(self, x):
        return self.resstack(self.upsample(x)[..., :-self.factor])


class Generator(nn.Module):
    """Multi-band generator."""
    
    def __init__(self, mels=80, channels=384, bands=4):
        
        super(Generator, self).__init__()
        assert 200 % (bands * 5 * 5) == 0
        factor0 = 200 // (bands * 5 * 5)
                
        self.flow = nn.Sequential(
            # normalization of the input mel-spectrogram
            nn.BatchNorm1d(mels), 
            
            # imput conv
            nn.ReplicationPad1d(3), # not deterministic backward for cuda
            wn(nn.Conv1d(mels, channels, 7, bias=False)),
            nn.LeakyReLU(0.2),
        
            # upsample layers
            Upsample(channels//2**0, factor=factor0),
            Upsample(channels//2**1, factor=5),
            Upsample(channels//2**2, factor=5),
            
            # output conv
            nn.LeakyReLU(0.2),
            nn.ReplicationPad1d(3), # not deterministic backward for cuda
            wn(nn.Conv1d(channels//2**3, bands, 7, bias=False)),
            nn.Tanh())
        
        self.mels = mels
        self.channels = channels
        self.bands = bands
        
    def forward(self, x):
        return self.flow(x)
    

class DiscriminatorBlock(nn.Module):
    
    def __init__(self):
        super(DiscriminatorBlock, self).__init__()
        
        self.flow = nn.Sequential(
            
            wn(nn.Conv1d(  1, 16,  15, 1, padding=7, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            
            wn(nn.Conv1d( 16, 64,  41, 4, padding=20, groups=4, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            
            wn(nn.Conv1d( 64, 256, 41, 4, padding=20, groups=16, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            
            wn(nn.Conv1d(256, 512, 41, 4, padding=20, groups=64, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            
            wn(nn.Conv1d(512, 512,  5, 1, padding=2, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            
            wn(nn.Conv1d(512,   1,  3, 1, padding=1, bias=False)))
        
    def forward(self, x):
        return self.flow(x)

    
class Discriminator(nn.Module):
    """Multi-scale discriminator."""
    
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.D1 = DiscriminatorBlock()
        self.D2 = DiscriminatorBlock()
        self.D3 = DiscriminatorBlock()
        
    def forward(self, x):
        res = [self.D1(x)]
        
        x = F.avg_pool1d(x, 4, 2, 1, count_include_pad=False)
        res.append(self.D2(x))
        
        x = F.avg_pool1d(x, 4, 2, 1, count_include_pad=False)
        res.append(self.D3(x))
        return res


# class Discriminator(nn.Module):
#     """Parallelized multi-scale discriminator."""
    
#     def __init__(self):
#         super(Discriminator, self).__init__()
        
#         self.D1 = DiscriminatorBlock()
#         self.D2 = DiscriminatorBlock()
#         self.D3 = DiscriminatorBlock()
#         self.stream1 = torch.cuda.Stream()
#         self.stream2 = torch.cuda.Stream()
        
#     def forward(self, x):
#         x2 = F.avg_pool1d(x, 4, 2, 1, count_include_pad=False)
#         x3 = F.avg_pool1d(x2, 4, 2, 1, count_include_pad=False)
        
#         with torch.cuda.stream(self.stream1):
#             y1 = self.D1(x)
        
#         with torch.cuda.stream(self.stream2):
#             y2 = self.D2(x2)
        
#         y3 = self.D3(x3)
#         torch.cuda.synchronize()
#         return [y1, y2, y3]


# abstract class
class _STFTLoss(nn.Module):
    """Basic class for Multi-resolution STFT losses."""
    
    def __init__(self, resolutions: list[STFT]):
        super(_STFTLoss, self).__init__()
        assert len(resolutions) > 0
        self.resolutions = nn.ModuleList(resolutions)
    
    def forward(self, real, pred):
        assert real.size() == pred.size()
        sm = sum([self._loss(real, pred, res) for res in self.resolutions])
        return sm / len(self.resolutions)
        
    def _loss(self, real, pred, stft: STFT):
        eps = 1e-5
        real = stft(real).abs() + eps
        pred = stft(pred).abs() + eps

        Lsc  = torch.norm(real - pred, "fro") / torch.norm(real, "fro")
        Lmag = torch.norm(real.log() - pred.log(), 1) / real.numel()
        return Lsc + Lmag

    
class FullBandLoss(_STFTLoss):
    """Multi-resolution full-band STFT loss."""
    
    def __init__(self):
        resolutions = [
            STFT(nfft=1024, nwin= 600, hop=120),
            STFT(nfft=2048, nwin=1200, hop=240),
            STFT(nfft= 512, nwin= 240, hop= 50)]
        
        super(FullBandLoss, self).__init__(resolutions)


class MultiBandLoss(_STFTLoss):
    """Multi-resolution multi-band STFT loss."""
    
    def __init__(self):
        resolutions = [
            STFT(nfft=384, nwin=150, hop=30),
            STFT(nfft=683, nwin=300, hop=60),
            STFT(nfft=171, nwin= 60, hop=10)]
            
        super(MultiBandLoss, self).__init__(resolutions)
        

class STFTTotalLoss(nn.Module): # eq.7 and eq.9
    """The final multi-resolution STFT loss."""
    
    def __init__(self, is_multi_band=True):
        
        super(STFTTotalLoss, self).__init__()
        self.is_multi_band = is_multi_band
        
        self.fb = FullBandLoss()
        self.mb = MultiBandLoss() if is_multi_band else None
        
    
    def forward(self, full_real, full_pred, band_real=None, band_pred=None):
        full_loss = self.fb(full_real, full_pred)
        if self.is_multi_band:
            return (full_loss + self.mb(band_real, band_pred)) / 2
        return full_loss


class DiscriminatorLoss(nn.Module): # eq.1
    
    def forward(self, real, fake):
        assert len(real) == len(fake)
        assert len(real) > 0
        sm =  sum([(r-1).pow(2).mean() + f.pow(2).mean() 
                   for r, f in zip(real, fake)])
        return sm / len(real)


class GeneratorLoss(nn.Module): # eq.8 (left part of the sum)

    def forward(self, fake):
        assert len(fake) > 0
        return sum([(f-1.).pow(2).mean() for f in fake])  / len(fake) 
        
    
if __name__ == "__main__":
    # test
    import scipy
    import librosa
    # import matplotlib.pyplot as plt
    
    def num_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    #--------------------------------------------------------------------------
    # stft (differences in stft implementation: librosa, scipy and torch)
    
    # nwin != nfft:
    # in this case (nwin != nfft) librosa.stft and torch.stft return the incorrect number of frames
    x = torch.randn(2,3,16000)
    X1 = scipy.signal.stft(x.numpy(), nperseg=800, noverlap=600, nfft=1024, boundary=None)[2]
    X2 = stft(x, 200, 1024, torch.hann_window(800))
    assert torch.allclose(torch.from_numpy(X1).abs(), X2.abs()*2/800, atol=1e-3, rtol=0)
    assert torch.allclose(torch.from_numpy(X1).angle(), X2.angle(), atol=1e-3, rtol=0)
    
    # nwin = nfft:
    y = torch.randn(2,16000)
    Y3 = stft(y, 200, 1024, torch.hann_window(1024))
    Y4 = torch.stft(y, 1024, 200, 1024, torch.hann_window(1024), return_complex=True, center=False)
    Y5 = librosa.stft(y=y.numpy(), n_fft=1024, hop_length=200, win_length=1024, center=False)
    assert torch.allclose(Y3, Y4)
    assert torch.allclose(Y3, torch.from_numpy(Y5), atol=1e-3, rtol=0)
    #--------------------------------------------------------------------------
        
    # generator
    x1 = torch.randn(2, 80, 80)
    g = Generator()
    #g = Generator(channels=512, bands=1)
    y = g(x1)
    assert y.size(-1)*4 == 16000
    
    # discriminator
    x2 = torch.randn(10, 1, 16000)#16000-(800-200))
    d = Discriminator()
    for i, y2 in enumerate(d(x2)):
        print("D{} shape: {}".format(i+1, y2.shape))
    
    print("Generator params number:     {}".format(num_params(g)))
    print("Discriminator params number: {}".format(num_params(d)))
     
    