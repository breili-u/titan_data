import torch
import torchaudio
import torch.nn.functional as F
import random

class RoomSimulator:
    def __init__(self, sr=16000, max_rt60=0.4):
        self.sr = sr
        self.ir_cache = self._generate_synthetic_irs(30, max_rt60)

    def _generate_synthetic_irs(self, count, max_rt60):
        cache = []
        for _ in range(count):
            rt60 = random.uniform(0.05, max_rt60)
            decay = torch.linspace(0, -60, int(rt60 * self.sr))
            amp = 10 ** (decay / 20)
            ir = torch.randn_like(amp) * amp
            ir = ir / (ir.abs().max() + 1e-9)
            cache.append(ir.unsqueeze(0))
        return cache

    def apply(self, wav, prob=0.5):
        if random.random() > prob: return wav
        ir = random.choice(self.ir_cache).to(wav.device)
        padded = F.pad(wav, (0, ir.shape[-1]))
        conv = torchaudio.functional.fftconvolve(padded, ir, mode="full")
        # Normalizar y cortar cola
        conv = conv / (conv.abs().max() + 1e-9)
        return conv[..., :wav.shape[-1]]

class SignalDegrader:
    def apply_clipping(self, wav, threshold=0.9):
        return torch.clamp(wav, -threshold, threshold)

    def apply_bandpass(self, wav, sr):
        # Efecto teléfono
        return torchaudio.functional.bandpass_biquad(wav, sr, 1500, 0.6)

    def apply_brutal(self, wav):
        """La famosa 'Brutalizer' del proyecto Titan."""
        gain = random.uniform(0.3, 2.0)
        wav = wav * gain
        if random.random() < 0.2: 
            wav = wav + random.uniform(-0.05, 0.05) # DC Offset
        if gain > 1.2 and random.random() < 0.4: 
            wav = torch.clamp(wav, -0.95, 0.95)
        return wav