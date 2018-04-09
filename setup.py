from setuptools import setup, find_packages  # Always prefer setuptools over distutils
from codecs import open  # To use a consistent encoding
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the relevant file
with open(path.join(here, "README.rst"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="tensorlm",
    packages=find_packages(exclude=["examples"]),
    version="0.4.1",
    description="TensorFlow wrapper for deep neural text generation on character or word level "
                "with RNNs / LSTMs",
    long_description=long_description,
    author="Kilian Batzner",
    author_email="tensorlm@kilians.net",
    license="MIT",
    url="https://github.com/batzner/tensorlm",
    download_url="https://github.com/batzner/tensorlm/archive/v0.4.1.tar.gz",
    keywords=["tensorflow", "text", "generation", "language", "model", "rnn", "lstm", "deep",
              "neural", "char", "word"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3 :: Only",
    ],
    install_requires=[
        "numpy==1.13.1",
        "nltk==3.2.4",
        "python-dateutil==2.6.1",
    ],
)
