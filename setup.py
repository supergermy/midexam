from setuptools import setup, find_packages

setup(
    name="midexam",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'pandas',
        'scikit-learn',
    ],
)
