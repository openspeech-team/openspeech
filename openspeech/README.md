## OpenSpeech Structure.

```
.
├── README.md
├── __init__.py
├── configs
│   ├── README.md
│   ├── eval.yaml
│   ├── lm_train.yaml
│   └── train.yaml
├── criterion
│   ├── __init__.py
│   ├── cross_entropy
│   │   ├── __init__.py
│   │   ├── configuration.py
│   │   └── cross_entropy.py
│   ├── ctc
│   │   ├── __init__.py
│   │   ├── configuration.py
│   │   └── ctc.py
│   ├── joint_ctc_cross_entropy
│   │   ├── __init__.py
│   │   ├── configuration.py
│   │   └── joint_ctc_cross_entropy.py
│   ├── label_smoothed_cross_entropy
│   │   ├── __init__.py
│   │   ├── configuration.py
│   │   └── label_smoothed_cross_entropy.py
│   ├── perplexity
│   │   ├── __init__.py
│   │   ├── configuration.py
│   │   └── perplexity.py
│   └── transducer
│       ├── __init__.py
│       ├── configuration.py
│       └── transducer.py
├── data
│   ├── __init__.py
│   ├── audio
│   │   ├── __init__.py
│   │   ├── augment.py
│   │   ├── data_loader.py
│   │   ├── dataset.py
│   │   ├── filter_bank
│   │   │   ├── __init__.py
│   │   │   ├── configuration.py
│   │   │   └── filter_bank.py
│   │   ├── load.py
│   │   ├── melspectrogram
│   │   │   ├── __init__.py
│   │   │   ├── configuration.py
│   │   │   └── melspectrogram.py
│   │   ├── mfcc
│   │   │   ├── __init__.py
│   │   │   ├── configuration.py
│   │   │   └── mfcc.py
│   │   └── spectrogram
│   │       ├── __init__.py
│   │       ├── configuration.py
│   │       └── spectrogram.py
│   ├── sampler.py
│   └── text
│       ├── data_loader.py
│       └── dataset.py
├── dataclass
│   ├── __init__.py
│   ├── configurations.py
│   └── initialize.py
├── datasets
│   ├── README.md
│   ├── __init__.py
│   ├── aishell
│   │   ├── __init__.py
│   │   ├── lit_data_module.py
│   │   └── preprocess.py
│   ├── ksponspeech
│   │   ├── __init__.py
│   │   ├── lit_data_module.py
│   │   └── preprocess
│   │       ├── __init__.py
│   │       ├── character.py
│   │       ├── grapheme.py
│   │       ├── preprocess.py
│   │       └── subword.py
│   ├── language_model
│   │   ├── __init__.py
│   │   └── lit_data_module.py
│   └── librispeech
│       ├── __init__.py
│       ├── lit_data_module.py
│       └── preprocess
│           ├── __init__.py
│           ├── character.py
│           ├── preprocess.py
│           └── subword.py
├── decoders
│   ├── __init__.py
│   ├── lstm_attention_decoder.py
│   ├── openspeech_decoder.py
│   ├── rnn_transducer_decoder.py
│   ├── transformer_decoder.py
│   └── transformer_transducer_decoder.py
├── encoders
│   ├── __init__.py
│   ├── conformer_encoder.py
│   ├── contextnet_encoder.py
│   ├── convolutional_lstm_encoder.py
│   ├── convolutional_transformer_encoder.py
│   ├── deepspeech2.py
│   ├── jasper.py
│   ├── lstm_encoder.py
│   ├── openspeech_encoder.py
│   ├── quartznet.py
│   ├── rnn_transducer_encoder.py
│   ├── squeezeformer_encoder.py
│   ├── transformer_encoder.py
│   └── transformer_transducer_encoder.py
├── lm
│   ├── __init__.py
│   ├── lstm_lm.py
│   ├── openspeech_lm.py
│   └── transformer_lm.py
├── metrics.py
├── models
│   ├── README.md
│   ├── __init__.py
│   ├── conformer
│   │   ├── __init__.py
│   │   ├── configurations.py
│   │   └── model.py
│   ├── contextnet
│   │   ├── __init__.py
│   │   ├── configurations.py
│   │   └── model.py
│   ├── deepspeech2
│   │   ├── __init__.py
│   │   ├── configurations.py
│   │   └── model.py
│   ├── jasper
│   │   ├── __init__.py
│   │   ├── configurations.py
│   │   └── model.py
│   ├── listen_attend_spell
│   │   ├── __init__.py
│   │   ├── configurations.py
│   │   └── model.py
│   ├── lstm_lm
│   │   ├── __init__.py
│   │   ├── configurations.py
│   │   └── model.py
│   ├── openspeech_ctc_model.py
│   ├── openspeech_encoder_decoder_model.py
│   ├── openspeech_language_model.py
│   ├── openspeech_model.py
│   ├── openspeech_transducer_model.py
│   ├── quartznet
│   │   ├── __init__.py
│   │   ├── configurations.py
│   │   └── model.py
│   ├── rnn_transducer
│   │   ├── __init__.py
│   │   ├── configurations.py
│   │   └── model.py
│   ├── transformer
│   │   ├── __init__.py
│   │   ├── configurations.py
│   │   └── model.py
│   ├── transformer_lm
│   │   ├── __init__.py
│   │   ├── configurations.py
│   │   └── model.py
│   ├── squeezeformer
│   │   ├── __init__.py
│   │   ├── configurations.py
│   │   └── model.py
│   └── transformer_transducer
│       ├── __init__.py
│       ├── configurations.py
│       └── model.py
├── modules
│   ├── __init__.py
│   ├── add_normalization.py
│   ├── additive_attention.py
│   ├── batchnorm_relu_rnn.py
│   ├── conformer_attention_module.py
│   ├── conformer_block.py
│   ├── conformer_convolution_module.py
│   ├── conformer_feed_forward_module.py
│   ├── contextnet_block.py
│   ├── contextnet_module.py
│   ├── conv2d_extractor.py
│   ├── conv2d_subsampling.py
│   ├── conv_base.py
│   ├── conv_group_shuffle.py
│   ├── deepspeech2_extractor.py
│   ├── depthwise_conv1d.py
│   ├── depthwise_conv2d.py
│   ├── dot_product_attention.py
│   ├── glu.py
│   ├── jasper_block.py
│   ├── jasper_subblock.py
│   ├── location_aware_attention.py
│   ├── mask.py
│   ├── mask_conv1d.py
│   ├── mask_conv2d.py
│   ├── multi_head_attention.py
│   ├── pointwise_conv1d.py
│   ├── positional_encoding.py
│   ├── positionwise_feed_forward.py
│   ├── quartznet_block.py
│   ├── quartznet_subblock.py
│   ├── relative_multi_head_attention.py
│   ├── residual_connection_module.py
│   ├── squeezeformer_attention_module.py
│   ├── squeezeformer_block.py
│   ├── squeezeformer_module.py
│   ├── swish.py
│   ├── time_channel_separable_conv1d.py
│   ├── transformer_embedding.py
│   ├── vgg_extractor.py
│   └── wrapper.py
├── optim
│   ├── __init__.py
│   ├── adamp.py
│   ├── novograd.py
│   ├── optimizer.py
│   ├── radam.py
│   └── scheduler
│       ├── __init__.py
│       ├── lr_scheduler.py
│       ├── reduce_lr_on_plateau_scheduler.py
│       ├── transformer_lr_scheduler.py
│       ├── tri_stage_lr_scheduler.py
│       ├── warmup_reduce_lr_on_plateau_scheduler.py
│       └── warmup_scheduler.py
├── search
│   ├── __init__.py
│   ├── beam_search_base.py
│   ├── beam_search_ctc.py
│   ├── beam_search_lstm.py
│   ├── beam_search_rnn_transducer.py
│   ├── beam_search_transformer.py
│   ├── beam_search_transformer_transducer.py
│   └── ensemble_search.py
├── tokenizers
│   ├── __init__.py
│   ├── aishell
│   │   ├── __init__.py
│   │   └── character.py
│   ├── ksponspeech
│   │   ├── __init__.py
│   │   ├── character.py
│   │   ├── grapheme.py
│   │   └── subword.py
│   ├── librispeech
│   │   ├── __init__.py
│   │   ├── character.py
│   │   └── subword.py
│   └── tokenizer.py
└── utils.py
```
