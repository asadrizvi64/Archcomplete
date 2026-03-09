from setuptools import setup, find_packages

setup(
    name="archcomplete-gs",
    version="0.1.0",
    description="ArchComplete-GS: Semantic 3DGS for Architectural Scene Completion",
    author="Syed Muhammad Asad Rizvi",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "gsplat>=1.0.0",
        "transformers>=4.40.0",
        "omegaconf>=2.3.0",
        "numpy>=1.24.0",
        "opencv-python>=4.9.0",
        "Pillow>=10.2.0",
        "tqdm>=4.66.0",
        "plyfile>=1.0.0",
        "scipy>=1.12.0",
        "networkx>=3.2",
        "wandb>=0.17.0",
    ],
)
