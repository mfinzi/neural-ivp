from setuptools import setup, find_packages
import sys, os

setup(
    name="neural-ivp",
    description="Scalable and Stable Neural Network Based IVP Solver",
    version='0.1',
    author='Marc Finzi, Andres Potapczynski',
    author_email='maf820@nyu.edu',
    license='MIT',
    python_requires='>=3.6',
    install_requires=[
        'h5py', 'tables', 'dm-haiku', 'optax', 'pyyaml', 'fire',
        'linops @ git+https://github.com/mfinzi/linops@pde'
    ],  #jax, 
    packages=['neural_ivp'],
    long_description=open('README.md').read(),
)