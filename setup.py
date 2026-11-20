from setuptools import setup, find_packages

setup(
    name="multimodal-peft-adapter-engine",
    version="1.0.0",
    description="Parameter-Efficient Fine-Tuning (PEFT) framework implementing LoRA, QLoRA NF4, and (IA)³ adapters for Vision-Language Models.",
    author="Siddiqui Zakir Ali",
    author_email="zak.siddiqui16@gmail.com",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.22.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "streamlit>=1.28.0",
    ],
)
