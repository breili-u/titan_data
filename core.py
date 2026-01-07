import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T
import glob
import random
import numpy as np
from pathlib import Path

# Importamos los módulos internos
from .generators import NoiseSynth
from .effects import RoomSimulator, SignalDegrader
from .mixing import mix_signals

class TitanAudioDataset(Dataset):
    def __init__(self, 
                 clean_path, 
                 noise_path=None, 
                 sample_rate=16000, 
                 duration=2.0, 
                 epoch_size=10000):
        """
        Args:
            clean_path (str): Ruta a carpeta con audios limpios (wav/flac).
            noise_path (str): Ruta opcional a ruidos reales (ESC-50, etc).
            duration (float): Duración en segundos de los crops.
        """
        self.sr = sample_rate
        self.length = int(sample_rate * duration)
        self.epoch_size = epoch_size
        
        # Carga de archivos robusta
        self.clean_files = sorted(list(Path(clean_path).rglob("*.flac")) + list(Path(clean_path).rglob("*.wav")))
        self.noise_files = []
        if noise_path:
            self.noise_files = sorted(list(Path(noise_path).rglob("*.wav")))
        
        # Módulos
        self.synth = NoiseSynth(sr=sample_rate)
        self.room = RoomSimulator(sr=sample_rate)
        self.degrader = SignalDegrader()
        
        # Configuración por defecto (Se actualiza con set_curriculum)
        self.snr_range = (0, 20)
        self.prob_real_noise = 0.5
        self.prob_room = 0.5

    def set_curriculum(self, snr_range=(0, 20), prob_real_noise=0.5):
        """Permite cambiar la dificultad durante el entrenamiento."""
        self.snr_range = snr_range
        self.prob_real_noise = prob_real_noise

    def _load_random_crop(self, file_list):
        if not file_list: return None
        for _ in range(3): # 3 intentos
            try:
                path = random.choice(file_list)
                w, sr = torchaudio.load(path)
                if sr != self.sr: w = T.Resample(sr, self.sr)(w)
                if w.shape[0] > 1: w = w.mean(dim=0, keepdim=True)
                
                if w.shape[1] > self.length:
                    start = random.randint(0, w.shape[1] - self.length)
                    return w[:, start:start+self.length]
                else:
                    return F.pad(w, (0, self.length - w.shape[1]))
            except: continue
        return torch.randn(1, self.length) * 0.01

    def _get_noise(self):
        # 1. Ruido Real (si existe y toca)
        if self.noise_files and random.random() < self.prob_real_noise:
            return self._load_random_crop(self.noise_files)
        
        # 2. Ruido Procedural (Fallback o elección)
        r = random.random()
        if r < 0.4: return self.synth.colored_noise(self.length, 'pink')
        elif r < 0.7: return self.synth.colored_noise(self.length, 'brown')
        elif r < 0.9: return self.synth.mains_hum(self.length)
        else: return self.synth.transient_click(self.length)

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        # 1. Cargar Clean
        clean = self._load_random_crop(self.clean_files)
        clean = clean / (clean.abs().max() + 1e-9)
        
        # 2. Aplicar Reverb (La voz está en una sala)
        clean = self.room.apply(clean, prob=self.prob_room)

        # 3. Generar Ruido
        noise = self._get_noise()
        
        # 4. Mezclar con SNR Controlado
        target_snr = random.uniform(*self.snr_range)
        noisy = mix_signals(clean, noise, target_snr)
        
        # 5. Brutalizar entrada (simular micro malo)
        if random.random() < 0.2:
            noisy = self.degrader.apply_brutal(noisy)

        # Output estandarizado
        return torch.clamp(noisy, -1, 1), torch.clamp(clean, -1, 1)