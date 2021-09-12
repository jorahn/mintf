from setuptools import setup, find_packages

setup(
    name='mintf',
    version='0.1.0',
    url='https://github.com/jorahn/mintf.git',
    author='Jonathan Rahn',
    author_email='jonathan.rahn@42digital.de',
    description='Minimal Raw TensorFlow Experiments',
    packages=find_packages(),
    install_requires=['tensorflow == 2.4.3'],
)
