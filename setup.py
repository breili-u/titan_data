from setuptools import setup, find_packages

setup(
    name="titan_data",
    version="0.1.0",
    description="Advanced Audio Dataset Generator for Robust Deep Learning",
    long_description="A procedural audio augmentation library designed for training SOTA denoising models. Features real-time DSP synthesis, SNR control, and acoustic simulation.",
    author="Tu Nombre o Titan Project",
    author_email="tu@email.com",
    url="https://github.com/tu_usuario/titan_data", # Si tienes repo
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "numpy",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.8",
)