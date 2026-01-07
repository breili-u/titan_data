import torch
import math

def calculate_rms(tensor):
    """Calcula el Root Mean Square seguro."""
    return tensor.norm(p=2) / (math.sqrt(tensor.numel()) + 1e-9)

def mix_signals(clean, noise, snr_db):
    """
    Mezcla dos señales respetando un SNR objetivo en dB.
    Args:
        clean (Tensor): Audio limpio [C, T] o [T]
        noise (Tensor): Ruido [C, T] o [T]
        snr_db (float): Relación Señal-Ruido deseada (ej: 10.0, -5.0)
    """
    clean_rms = calculate_rms(clean)
    noise_rms = calculate_rms(noise)
    
    if noise_rms < 1e-9: 
        return clean
    
    # Fórmula: target_noise = clean / 10^(snr/20)
    target_noise_rms = clean_rms / (10**(snr_db/20))
    scale = target_noise_rms / (noise_rms + 1e-9)
    
    return clean + noise * scale