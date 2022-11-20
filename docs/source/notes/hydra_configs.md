# Openspeech's Hydra configuration

This page describes how openspeech uses [Hydra](https://github.com/facebookresearch/hydra) to manage configuration.

## What is Hydra?

[Hydra](https://github.com/facebookresearch/hydra) is an open-source Python framework that simplifies the development of research and other complex applications. The key feature is the ability to dynamically create a hierarchical configuration by composition and override it through config files and the command line. The name Hydra comes from its ability to run multiple similar jobs - much like a Hydra with multiple heads.

## Motivation

Openspeech aims to provide as many options as possible. However, too many options cause a lot of confusion to users.
. To address this problem, we needed a hierarchical configuration management toolkit.
Hydra was the best choice for us in that respect. We have referred to the structure of [Fairseq](https://github.com/pytorch/fairseq) that successfully applied Hydra.
Thank you for fairseq team.

We thought a lot about which method is better, using the `YAML` file or using `@dataclass`.
After much consideration, we decided to use `@dataclass`, which is easy to understand each module's configuration.
`@dataclass` has default values for that module and has been configured to be stored in the same Python file as each module.

Additionally, Hydra has a rich and growing library of plugins that provide functionality such as hyperparameter sweeping (including using bayesian optimization through the Ax library), job launching across various platforms, and more.

## Creating or migrating components

In general, each new (or updated) component should provide a companion [dataclass](https://www.python.org/dev/peps/pep-0557/).
These dataclass are typically located in the same file as the component and are passed as arguments to the `register_*()` functions.
These classes are decorated with a `@dataclass` decorator, and typically inherit from `OpenspeechDataclass`.


#### Example:

```python
from dataclasses import dataclass, field
from openspeech.dataclass.configurations import OpenspeechDataclass


@dataclass
class ConformerLSTMConfigs(OpenspeechDataclass):
    model_name: str = field(
        default="conformer_lstm", metadata={"help": "Model name"}
    )
    encoder_dim: int = field(
        default=256, metadata={"help": "Dimension of encoder."}
    )
```

### `@register_*()` function

We actively utilized the `@register_*()` function inspired by [Fairseq](https://github.com/pytorch/fairseq).
The function `@register_*()` automatically registers classes and associated data classes.
This method is very effective when adding new modules.
Below is an example of how `register_*()` functions and data classes are utilized in `Openspeech`.

#### Model example:
```python
@dataclass
class TransformerConfigs(ModelConfigs):
    model_name: str = field(
        default="transformer", metadata={"help": "Model name"}
    )
    extractor: str = field(
        default="vgg", metadata={"help": "The CNN feature extractor."}
    )
    ...


@register_model('transformer', dataclass=TransformerConfigs)
class SpeechTransformerModel(OpenspeechEncoderDecoderModel):
    ...
    def build_model(self):
        ...
```

#### Dataset example:
```python
@dataclass
class MelSpectrogramConfigs(AudioConfigs):
    name: str = field(
        default="melspectrogram", metadata={"help": "Name of dataset."}
    )
    num_mels: int = field(
        default=80, metadata={"help": "The number of mfc coefficients to retain."}
    )
    ...


@register_dataset("melspectrogram", dataclass=MelSpectrogramConfigs)
class MelSpectrogramDataset(AudioDataset):
    ...
    def __init__(self):
        ...
```

## Openspeech's configuration structure

Below are the configuration dataclasses that you can select from `Openspeech`.

```
defaults:
  - audio:
    - fbank
    - melspectrogram
    - mfcc
    - spectrogram
  - common:
    - kspon
    - libri
    - aishell
  - criterion:
    - cross_entropy
    - ctc
    - joint_ctc_cross_entropy
    - label_smoothed_cross_entropy
    - transducer
  - lr_scheduler:
    - reduce_lr_on_plateau
    - transformer
    - tri_stage
    - warmup_reduce_lr_on_plateau
    - warmup
  - model:
    - conformer_encoder_only
    - conformer_lstm
    - conformer_transducer
    - deepspeech2
    - jasper
    - listen_attend_spell
    - rnn_transducer
    - transformer
    - transformer_transducer
  - trainer:
    - cpu
    - gpu
    - tpu
    - cpu-fp64
    - gpu-fp16
    - tpu-fp16
  - vocab:
    - aishell_character
    - kspon_character
    - kspon_subword
    - kspon_grapheme
    - libri_character
    - libri_subword
```

## Training with `hydra_train.py`

On startup, Hydra will create a configuration object that contains a hierarchy of all the necessary dataclasses populated with their default values in the code.

Some of the most common use cases are shown below:

### 1. Override default values through command line:

```diff
$ python ./openspeech_cli/hydra_train.py \
    common=libri \
+   common.dataset_path=$DATASET_PATH \
+   common.dataset_download=True \
+   common.manifest_file_path=$MANIFEST_FILE_PATH \
    vocab=libri_subword \
+   vocab.vocab_size=10000 \
    model=conformer_lstm \
+   model.encoder_dim=320 \
    audio=mfcc \
    lr_scheduler=warmup_reduce_lr_on_plateau \
    trainer=gpu-fp16 \
    criterion=ctc
```

Note that along with explicitly providing values for parameters such as `common.dataset_path`, this also tells Hydra to overlay configuration found in dataclass.
If you want to train a model without specifying a particular architecture you can simply specify `model=conformer_lstm`.

### 2. Add new configuration through command line:

```diff
$ python ./openspeech_cli/hydra_train.py \
    common=libri \
    vocab=libri_subword \
    model=conformer_lstm \
    audio=mfcc \
    lr_scheduler=warmup_reduce_lr_on_plateau \
    trainer=gpu-fp16 \
+   +trainer.is_gpu=True \
+   +trainer.is_tpu=False \
    criterion=ctc \
```

More detailed methods of using hydra can be found [Hydra website](https://hydra.cc/). If you have any questions, feel free to send me an email or create an issue.