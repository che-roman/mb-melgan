# Multi-band MelGAN
Unofficial PyTorch implementation of [Multi-band MelGAN: Faster Waveform Generation for High-Quality Text-to-Speech](https://arxiv.org/abs/2005.05106).

Audio samples are available on the project [demo page](https://che-roman.github.io/mb-melgan/).

### Model
I use _Identity_ as a shortcut connection (instead of _Linear_) in residual blocks and don't use _biases_, so my implementation has slightly fewer parameters than described in the paper (_1.52_ vs _1.91_).

### PQMF
The cutoff-ratio of the pseudo quadratue nirror filter bank can be set to a specific value or to _None_. In the latter case, the optimal filter will be automatically synthesized before the start of training.

### Train
To start training for, say, 500K iterations, run the command:
```bash
train.py -l log -c config/mb_train.yaml -i 500000
```
To continue training from the last saved checkpoint for another 500K iterations, run the command:
```bash
train.py -l log -i 500000
```
The training results will be posted in the _log_ folder and available for viewing via the tensorboard.
    
### Vocoder
Pretrained multi-band vocoder (_config_ and _weights_) can be downloaded [here](https://drive.google.com/drive/folders/1Pu_1nHx2kS7ecn23vJu9AeXfwGe-btLQ?usp=sharing). This model was trained for 500K iterations on the LJSpeech dataset.

#### Example
```python
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

# wav-to-mel
y = vocoder.encode(x)
with torch.no_grad():
	# mel-to-wav
    x_hat = vocoder.decode(y)

# play restored wav
sd.play(x_hat, sr, blocking=True)
```
