## Introduction
  
[***KoSpeech***](https://github.com/sooftware/KoSpeech), an open-source software, is modular and extensible end-to-end Korean automatic speech recognition (ASR) toolkit based on the deep learning library PyTorch. 
KoSpeech provides a variety of features, but we created KoSpeech2 for more features and better code quality. 
To do so, we combined [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) and KoSpeech.  
  
PyTorch Lightning is the lightweight PyTorch wrapper for high-performance AI research. PyTorch is extremely easy to use to build complex AI models. But once the research gets complicated and things like multi-GPU training, 16-bit precision and TPU training get mixed in, users are likely to introduce bugs. PyTorch Lightning solves exactly this problem. Lightning structures your PyTorch code so it can abstract the details of training. This makes AI research scalable and fast to iterate on.
  
### Installation
  
This project recommends Python 3.7 or higher.  
I recommend creating a new virtual environment for this project (using virtual env or conda).
  

#### Prerequisites
  
* numpy: `pip install numpy` (Refer [here](https://github.com/numpy/numpy) for problem installing Numpy).
* pytorch: Refer to [PyTorch website](http://pytorch.org/) to install the version w.r.t. your environment.   
* librosa: `conda install -c conda-forge librosa` (Refer [here](https://github.com/librosa/librosa) for problem installing librosa)
* torchaudio: `pip install torchaudio==0.6.0` (Refer [here](https://github.com/pytorch/pytorch) for problem installing torchaudio)
* sentencepiece: `pip install sentencepiece` (Refer [here](https://github.com/google/sentencepiece) for problem installing sentencepiece)
* pytorch-lightning: `pip install pytorch-lightning` (Refer [here](https://github.com/PyTorchLightning/pytorch-lightning) for problem installing pytorch-lightning)
* hydra: `pip install hydra-core --upgrade` (Refer [here](https://github.com/facebookresearch/hydra) for problem installing hydra)
  
#### Install from source
Currently I only support installation from source code using setuptools. Checkout the source code and run the   
following commands:  
```
$ pip install -e .
$ ./setup.sh
```
  
#### Install Apex (for 16-bit training) 
  
For faster training install NVIDIA's apex library:
  
```
$ git clone https://github.com/NVIDIA/apex
$ cd apex

# ------------------------
# OPTIONAL: on your cluster you might need to load CUDA 10 or 9
# depending on how you installed PyTorch

# see available modules
module avail

# load correct CUDA before install
module load cuda-10.0
# ------------------------

# make sure you've loaded a cuda version > 4.0 and < 7.0
module load gcc-6.1.0

$ pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```  
### Get Started
  
We use [Hydra](https://github.com/facebookresearch/hydra) to control all the training configurations. If you are not familiar with Hydra we recommend visiting the [Hydra website](https://hydra.cc/). Generally, Hydra is an open-source framework that simplifies the development of research applications by providing the ability to create a hierarchical configuration dynamically.
  
#### Prepare KsponSpeech dataset
  
To operate KoSpeech2, the [KsponSpeech](https://aihub.or.kr/aidata/105) dataset must be prepared in advance. To download [KsponSpeech](https://aihub.or.kr/aidata/105), you needs permission from [AI Hub](https://aihub.or.kr/).
  
#### Train Speech Recognizer
  
You can simply train with KsponSpeech dataset like below:  
  
- Example1: Train the `conformer-lstm` model with `filter-bank` features on GPU.
  
```
$ python ./kospeech2_cli/hydra_train.py \
    common.dataset_path=$DATASET_PATH \
    common.manifest_file_path=$MANIFEST_FILE_PATH \  
    vocab=character \
    model=conformer_lstm \
    audio=fbank \
    lr_scheduler=reduce_lr_on_plateau \
    training=gpu \
    criterion=joint_ctc_cross_entropy
```

- Example2: Train the `listen_attend_spell` model with `mel-spectrogram` features On TPU:
  
```
$ python ./kospeech2_cli/hydra_train.py \
    common.dataset_path=$DATASET_PATH \
    common.manifest_file_path=$MANIFEST_FILE_PATH \  
    vocab=character \
    model=listen_attend_spell \
    audio=melspectrogram \
    lr_scheduler=reduce_lr_on_plateau \
    training=tpu \
    criterion=joint_ctc_cross_entropy
```
  
### Troubleshoots and Contributing
If you have any questions, bug reports, and feature requests, please [open an issue](https://github.com/sooftware/KoSpeech2/issues) on Github.   
  
We appreciate any kind of feedback or contribution.  Feel free to proceed with small issues like bug fixes, documentation improvement.  For major contributions and new features, please discuss with the collaborators in corresponding issues.
  
#### Code Style
We follow [PEP-8](https://www.python.org/dev/peps/pep-0008/) for code style. Especially the style of docstrings is important to generate documentation. 
  
#### License
This project is licensed under the MIT LICENSE - see the [LICENSE.md](https://github.com/sooftware/KoSpeech2/blob/master/LICENSE) file for details
  
### Citation
  
If you use the system for academic work, please cite:
  
```
@GITHUB{2021-kospeech2,
  author = {Kim, Soohwan and Ha, Sangchun},
  title  = {KoSpeech2: Open-Source Toolkit for End-to-End Korean Speech Recognition},
  month  = {May},
  year   = {2021}
}
```