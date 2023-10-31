# Copyright (C) 2023 Chudnov Roman
# MIT License (https://opensource.org/licenses/MIT)

"""
Filter design of the Pseudo-QMF (pseudo-quadrature mirror filter).
"""

import torch
from torch import pi, arange, sqrt, pow, is_tensor, tensor, cos, outer
from torch.nn.functional import conv1d, conv_transpose1d
from torch.fft import fft
from torch.special import i0, sinc


def kaiser(length, beta):
    """Creates a Kaiser window."""
    if is_tensor(beta):
        assert beta.ndim == 0, "'beta' should be a zero-dimensional tensor"
    else:
        beta = tensor(beta)
    return i0(beta * sqrt(1 - pow(2*arange(length)/length - 1, 2))) / i0(beta)


def proto(cutoff_ratio, beta, length):
    """
    Creates a prototype filter of the Pseudo-QMF as described in the paper 
    "A Kaiser Window Approach for the Design of Prototype Filters of Cosine 
    Modulated Filterbanks".
    """
    # "sinc" instead on "sin" - to avoid nans
    h = sinc(cutoff_ratio * (arange(length+1) - 0.5*length)) * cutoff_ratio
    #w = torch.kaiser_window(length+1, beta=beta, periodic=False, requires_grad=True)
    w = kaiser(length+1, beta) 
    return h * w


def pqmf(cutoff_ratio, beta, length, bands):
    """Creates a Pseudo-QMF filter bank."""
    p = proto(cutoff_ratio, beta, length-1)
    i = arange(bands)
    n = arange(length)
    # https://arxiv.org/pdf/2303.10008.pdf
    left_arg = outer((2*i+1), (n-(length-1)/2)) * pi/2/bands
    right_arg = pow(-1, i.unsqueeze(1)) * pi/4
    h = 2*p*cos(left_arg + right_arg) # Analysis filter bank
    g = 2*p*cos(left_arg - right_arg) # Synthesis filter bank
    return h, g


def design(length, bands, beta=None, epochs=10):
    """
    Returns the optimal cutoff ratio and "beta"-parameter of the Kaiser window 
    for a given filter length and number of bands.
    """
    def criterion(cutoff_ratio, beta, length, bands):
        unit_abs = lambda flt: (1. - fft(flt).sum(0).abs()).pow(2).mean()
        h, g = pqmf(cutoff_ratio, beta, length, bands)
        return (unit_abs(h) + unit_abs(g)) / 2

    cutoff_ratio = torch.tensor(1./2/bands)
    cutoff_ratio.requires_grad = True
    params = [cutoff_ratio]
    
    if beta is None:
        beta = torch.tensor(1.)
        beta.requires_grad = True
        params.append(beta)
    
    optim = torch.optim.LBFGS(params, line_search_fn="strong_wolfe")
    
    for i in range(epochs):
        
        def closure():
            optim.zero_grad()
            loss = criterion(cutoff_ratio, beta, length, bands)
            #print(i, loss.item(), cutoff_ratio.item(), beta.item())
            loss.backward()
            return loss
        
        optim.step(closure)

    return cutoff_ratio.item(), beta.item()


class PQMF(torch.nn.Module):
    """Pseudo-QMF converter."""
    
    def __init__(self, cutoff_ratio, beta, length, bands):
        super(PQMF, self).__init__()
        
        h, g = pqmf(cutoff_ratio, beta, length, bands)
        
        self.register_buffer("h", h.unsqueeze(1), False)
        self.register_buffer("g", g.unsqueeze(1), False)
        
        self.length = length
        self.bands  = bands
            
    def forward(self, x):
        """Splits the waveform into bands (analysis)."""
        return conv1d(x, self.h, stride=self.bands, padding=self.length//2)
            
    def inverse(self, x):
        """Composes the bands into a waveform (synthesis)."""
        x = conv_transpose1d(x, self.g.flip(2)*self.bands, stride=self.bands)
        return x[..., self.length//2: -(self.length-self.length//2-self.bands)]
        

if __name__ == "__main__":
    #test
    import matplotlib.pyplot as plt
    import librosa
    import sounddevice as sd
    
    cutoff_ratio = tensor(0.15)
    beta = tensor(9.0)
    length = 63
    bands = 4 #1
    
    # impulse response of the prototype filter
    plt.figure()
    plt.plot(proto(cutoff_ratio, beta, length))
    plt.title("Impulse response of the prototype filter")
    plt.show()
    
    # analysis filter bank frequency responses
    plt.figure()
    h, g = pqmf(cutoff_ratio, beta, length, bands)
    plt.plot(abs(fft(h)).T)
    plt.title("Analysis filter bank frequency responses")
    plt.show()

    # design optimal filter
    #cutoff_opt, beta_opt = design(length, bands, epochs=10)
    cutoff_opt, beta_opt = design(length, bands, beta, epochs=10)
    print("optimal filter | cutoff_ratio: {:.4f}, beta: {:.2f}".format(cutoff_opt, beta_opt))
    oh, og = pqmf(cutoff_opt, beta_opt, length, bands)
            
    plt.figure()
    plt.plot(abs(fft(oh)).T)
    plt.title("Analysis filter bank (optimal) frequency responses")
    plt.show()
    
    
    # frequency response overlap
    plt.figure()
    plt.plot(abs(fft(h ).sum(0))**2, label="given filter")
    plt.plot(abs(fft(oh).sum(0))**2, label="optimal filter")
    plt.title("Frequency response overlap")
    plt.legend()
    plt.show()
    

    # test PQMF-class:
    #pqmf_module = PQMF(cutoff_ratio, beta, length, bands)
    pqmf_module = PQMF(cutoff_opt, beta_opt, length, bands)
    s, fs = librosa.load(librosa.example("libri1"))
    x = torch.from_numpy(s[:3*fs])
    y = pqmf_module(x.unsqueeze(0).unsqueeze(0))
    x_hat = pqmf_module.inverse(y).squeeze()

    plt.figure()
    plt.plot(x    [10000: 12000], label="original")
    plt.plot(x_hat[10000: 12000], label="restored") # significantly better
    plt.title("Original and restored waveforms")
    plt.legend()
    plt.show()
    
    #sd.play(x, fs, blocking=True) # original
    sd.play(x_hat, fs, blocking=True) # restored
        