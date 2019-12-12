import setuptools
from setuptools import setup, find_packages


with open('README.md', encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name="ekmeans",
    version='0.0.3',
    author='author',
    author_email='soy.lovit@gmail.com',
    description="Constrained k-means",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/lovit/ekmeans',
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy>=1.12.1",
        "scipy>=1.1.0",
        "scikit-learn>=0.20.0",
        #'soydata @ git+ssh://git@github.com/lovit/synthetic_dataset',
    ]
)