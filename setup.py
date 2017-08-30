from distutils.core import setup

setup(
    name="tensorlm",
    packages=["tensorlm"],
    version="0.1",
    description="TensorFlow wrapper for deep neural text generation on character or word level "
                "with RNNs / LSTMs",
    author="Kilian Batzner",
    author_email="info@kilians.net",
    url="https://github.com/batzner/tensorlm",
    download_url="https://github.com/batzner/tensorlm/archive/v0.1.tar.gz",
    keywords=["tensorflow", "text", "generation", "language", "model", "rnn", "lstm", "deep",
              "neural", "char", "word"],
    classifiers=[],
)
