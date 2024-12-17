from setuptools import setup, find_packages

setup(
    name='nova_net',
    version='0.1.0',
    description='NovaNet: A Gated Multi-Scale Segmentation Architecture for Pothole600 Dataset',
    author='Your Name',
    author_email='youremail@example.com',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'albumentations',
        'tqdm',
        'Pillow',
        'numpy'
    ],
    python_requires='>=3.6',
)
