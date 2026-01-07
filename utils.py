import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
from pathlib import Path
import random

def safe_load_audio(file_path, target_sr=16000, target_len=None, force_mono=True):
    """
    Carga, resamplea y ajusta un archivo de audio de forma segura.
    
    Args:
        file_path (str/Path): Ruta al archivo de audio.
        target_sr (int): Sample rate deseado.
        target_len (int, optional): Longitud fija en muestras. Si es None, devuelve el original.
        force_mono (bool): Si True, promedia canales para devolver [1, T].
        
    Returns:
        torch.Tensor: Audio procesado [Channels, Time] o None si falla.
    """
    try:
        path = str(file_path)
        # Carga rápida (backend-agnostic)
        waveform, sr = torchaudio.load(path)
        
        # 1. Resampling
        if sr != target_sr:
            resampler = T.Resample(sr, target_sr)
            waveform = resampler(waveform)
            
        # 2. Mono / Stereo Fix
        if force_mono and waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        elif not force_mono and waveform.shape[0] == 1:
            # Si pedimos estéreo pero es mono, duplicamos
            waveform = waveform.repeat(2, 1)
            
        # 3. Padding / Cropping (Si se especifica longitud)
        if target_len is not None:
            current_len = waveform.shape[1]
            
            if current_len > target_len:
                # Crop aleatorio (para entrenamiento)
                start = random.randint(0, current_len - target_len)
                waveform = waveform[:, start:start + target_len]
            elif current_len < target_len:
                # Pad con ceros (o reflect)
                padding = target_len - current_len
                # Pad a la derecha
                waveform = F.pad(waveform, (0, padding), "constant", 0)
                
        return waveform

    except Exception as e:
        print(f"Error {file_path}: {e}")
        return None

def scan_audio_files(directory, extensions=['.wav', '.flac', '.mp3']):
    """
    Escanea recursivamente un directorio buscando archivos de audio válidos.
    """
    files = []
    path = Path(directory)
    if not path.exists():
        return []
        
    for ext in extensions:
        # Case insensitive search sería ideal, pero glob es simple
        files.extend(list(path.rglob(f"*{ext}")))
        files.extend(list(path.rglob(f"*{ext.upper()}")))
        
    return sorted([str(f) for f in files])

def db_to_linear(db):
    """Convierte decibelios a escala lineal."""
    return 10 ** (db / 20.0)

def linear_to_db(scale):
    """Convierte escala lineal a decibelios."""
    return 20.0 * torch.log10(scale + 1e-9)