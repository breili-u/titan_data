__version__ = "0.1.0"
__author__ = "breili-u"

from torch.utils.data import DataLoader

from .core import TitanAudioDataset
from .generators import NoiseSynth
from .effects import RoomSimulator, SignalDegrader
from .mixing import mix_signals
from .utils import safe_load_audio, scan_audio_files
from .loss import NewtonianLoss

__all__ = [
    "DataLoader",
    "TitanAudioDataset",
    "NewtonianLoss",
    "NoiseSynth",
    "RoomSimulator",
    "SignalDegrader",
    "mix_signals",
    "safe_load_audio",
    "scan_audio_files",
]

print(f"🧬 titan_data v{__version__} initialized - Ready for Chaos.")