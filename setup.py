from setuptools import setup, find_packages

setup(
    name='nova_net',
    version='0.1.0',
    description='NovaNet: A Novel Method for Enhanced Pothole Detection',
    author='Soumick Sarker',
    author_email='soumicksarker9@gmail.com',
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
