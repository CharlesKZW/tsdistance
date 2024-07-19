from setuptools import setup, find_packages
setup(
    name='tsdistance',
    version='0.0.1',
    packages=find_packages(),
    description='Python Library for Time Series Distance Measure',
    author='The DATUM Lab',
    author_email='your.email@example.com',
    url='https://github.com/CharlesKZW/tsdistance',
    install_requires=['numpy', 'numba'],
)