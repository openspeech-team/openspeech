# MIT License
#
# Copyright (c) 2021 Soohwan Kim and Sangchun Ha and Soyoung Cho
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from setuptools import find_packages, setup

setup(
    name="openspeech-core",
    version="0.4.0",
    description="Open-Source Toolkit for End-to-End Automatic Speech Recognition",
    author="Kim, Soohwan and Ha, Sangchun and Cho, Soyoung",
    author_email="sh951011@gmail.com, seomk9896@gmail.com, soyoung.cho@kaist.ac.kr",
    url="https://github.com/openspeech-team/openspeech",
    download_url="https://github.com/openspeech-team/openspeech/releases/tag/v0.4.zip",
    install_requires=[
        "torch>=1.6.0",
        "levenshtein",
        "numpy",
        "pandas",
        "astropy",
        "sentencepiece",
        "pytorch-lightning",
        "hydra-core",
        "wget",
        "wandb",
    ],
    packages=find_packages(exclude=[]),
    keywords=["openspeech", "asr", "speech_recognition", "pytorch-lightning", "hydra"],
    python_requires=">=3.7",
    package_data={},
    zip_safe=False,
)
