# Openspeech's configurations
  
This page describes all configurations in `Openspeech`.
  
## `common`
  
### `kspon`  
- `dataset` : Select dataset for training (librispeech, ksponspeech, aishell)
- `dataset_path` : Path of dataset
- `manifest_file_path` : Path of manifest file
- `preprocess_mode` : KsponSpeech preprocess mode
  
### `libri`  
- `dataset` : Select dataset for training (librispeech, ksponspeech, aishell)
- `dataset_path` : Path of dataset
- `dataset_download` : Flag indication whether to download dataset or not.
- `manifest_file_path` : Path of manifest file
- `preprocess_mode` : KsponSpeech preprocess mode
  
### `aishell`  
- `dataset` : Select dataset for training (librispeech, ksponspeech, aishell)
- `dataset_path` : Path of dataset
- `dataset_download` : Flag indication whether to download dataset or not.
- `manifest_file_path` : Path of manifest file
  
## `audio`
  
### `fbank`  
- `sample_rate` : Sampling rate of audio
- `frame_length` : Frame length for spectrogram
- `frame_shift` : Length of hop between STFT
- `apply_spec_augment` : Flag indication whether to apply spec augment or not
- `freq_mask_para` : Hyper Parameter for freq masking to limit freq masking length
- `freq_mask_num` : How many freq-masked area to make
- `time_mask_num` : How many time-masked area to make
- `del_silence` : Flag indication whether to apply delete silence or not
- `name` : Name of dataset.
- `num_mels` : The number of mfc coefficients to retain.
  
### `melspectrogram`  
- `sample_rate` : Sampling rate of audio
- `frame_length` : Frame length for spectrogram
- `frame_shift` : Length of hop between STFT
- `apply_spec_augment` : Flag indication whether to apply spec augment or not
- `freq_mask_para` : Hyper Parameter for freq masking to limit freq masking length
- `freq_mask_num` : How many freq-masked area to make
- `time_mask_num` : How many time-masked area to make
- `del_silence` : Flag indication whether to apply delete silence or not
- `name` : Name of dataset.
- `num_mels` : The number of mfc coefficients to retain.
  
### `spectrogram`  
- `sample_rate` : Sampling rate of audio
- `frame_length` : Frame length for spectrogram
- `frame_shift` : Length of hop between STFT
- `apply_spec_augment` : Flag indication whether to apply spec augment or not
- `freq_mask_para` : Hyper Parameter for freq masking to limit freq masking length
- `freq_mask_num` : How many freq-masked area to make
- `time_mask_num` : How many time-masked area to make
- `del_silence` : Flag indication whether to apply delete silence or not
- `name` : Name of dataset.
- `num_mels` : The number of mfc coefficients to retain. Spectrogram is independent of mel, but uses the 'num_mels' variable to unify feature size variables
  
### `mfcc`  
- `sample_rate` : Sampling rate of audio
- `frame_length` : Frame length for spectrogram
- `frame_shift` : Length of hop between STFT
- `apply_spec_augment` : Flag indication whether to apply spec augment or not
- `freq_mask_para` : Hyper Parameter for freq masking to limit freq masking length
- `freq_mask_num` : How many freq-masked area to make
- `time_mask_num` : How many time-masked area to make
- `del_silence` : Flag indication whether to apply delete silence or not
- `name` : Name of dataset.
- `num_mels` : The number of mfc coefficients to retain.
  
## `model`
  
### `listen_attend_spell`  
- `optimizer` : Optimizer for training.
- `model_name` : Model name
- `num_encoder_layers` : The number of encoder layers.
- `num_decoder_layers` : The number of decoder layers.
- `hidden_state_dim` : The hidden state dimension of encoder.
- `encoder_dropout_p` : The dropout probability of encoder.
- `encoder_bidirectional` : If True, becomes a bidirectional encoders
- `rnn_type` : Type of rnn cell (rnn, lstm, gru)
- `extractor` : The CNN feature extractor.
- `activation` : Type of activation function
- `joint_ctc_attention` : Flag indication joint ctc attention or not
- `max_length` : Max decoding length.
- `num_attention_heads` : The number of attention heads.
- `decoder_dropout_p` : The dropout probability of decoder.
- `decoder_attn_mechanism` : The attention mechanism for decoder.
- `teacher_forcing_ratio` : The ratio of teacher forcing. 
  
### `conformer_encoder_only`  
- `optimizer` : Optimizer for training.
- `model_name` : Model name
- `encoder_dim` : Dimension of encoder.
- `num_encoder_layers` : The number of encoder layers.
- `num_attention_heads` : The number of attention heads.
- `feed_forward_expansion_factor` : The expansion factor of feed forward module.
- `conv_expansion_factor` : The expansion factor of convolution module.
- `input_dropout_p` : The dropout probability of inputs.
- `feed_forward_dropout_p` : The dropout probability of feed forward module.
- `attention_dropout_p` : The dropout probability of attention module.
- `conv_dropout_p` : The dropout probability of convolution module.
- `conv_kernel_size` : The kernel size of convolution.
- `half_step_residual` : Flag indication whether to use half step residual or not
- `joint_ctc_attention` : Flag indication joint ctc attention or not
  
### `deepspeech2`  
- `optimizer` : Optimizer for training.
- `model_name` : Model name
- `rnn_type` : Type of rnn cell (rnn, lstm, gru)
- `num_rnn_layers` : The number of rnn layers
- `rnn_hidden_dim` : Hidden state dimenstion of RNN.
- `dropout_p` : The dropout probability of model.
- `bidirectional` : If True, becomes a bidirectional encoders
- `activation` : Type of activation function
  
### `jasper`  
- `optimizer` : Optimizer for training.
- `model_name` : Model name
- `version` : Jasper's version. Supports `10x5`, `5x3`
  
### `transformer`  
- `optimizer` : Optimizer for training.
- `model_name` : Model name
- `extractor` : The CNN feature extractor.
- `d_model` : Dimension of model.
- `d_ff` : Dimenstion of feed forward network.
- `num_attention_heads` : The number of attention heads.
- `num_encoder_layers` : The number of encoder layers.
- `num_decoder_layers` : The number of decoder layers.
- `encoder_dropout_p` : The dropout probability of encoder.
- `decoder_dropout_p` : The dropout probability of decoder.
- `ffnet_style` : Style of feed forward network. (ff, conv)
- `max_length` : Max decoding length.
- `teacher_forcing_ratio` : The ratio of teacher forcing. 
- `joint_ctc_attention` : Flag indication joint ctc attention or not
  
### `conformer_transducer`  
- `optimizer` : Optimizer for training.
- `model_name` : Model name
- `encoder_dim` : Dimension of encoder.
- `num_encoder_layers` : The number of encoder layers.
- `num_attention_heads` : The number of attention heads.
- `feed_forward_expansion_factor` : The expansion factor of feed forward module.
- `conv_expansion_factor` : The expansion factor of convolution module.
- `input_dropout_p` : The dropout probability of inputs.
- `feed_forward_dropout_p` : The dropout probability of feed forward module.
- `attention_dropout_p` : The dropout probability of attention module.
- `conv_dropout_p` : The dropout probability of convolution module.
- `conv_kernel_size` : The kernel size of convolution.
- `half_step_residual` : Flag indication whether to use half step residual or not
- `num_decoder_layers` : The number of decoder layers.
- `decoder_dropout_p` : The dropout probability of decoder.
- `max_length` : Max decoding length.
- `teacher_forcing_ratio` :  The ratio of teacher forcing. 
- `joint_ctc_attention` : Flag indication joint ctc attention or not
- `rnn_type` : Type of rnn cell (rnn, lstm, gru)
- `decoder_hidden_state_dim` : Hidden state dimension of decoder.
- `decoder_output_dim` : Output dimension of decoder.
  
### `rnn_transducer`  
- `optimizer` : Optimizer for training.
- `model_name` : Model name
- `encoder_hidden_state_dim` : Dimension of encoder.
- `decoder_hidden_state_dim` : Dimension of decoder.
- `num_encoder_layers` : The number of encoder layers.
- `num_decoder_layers` : The number of decoder layers.
- `encoder_dropout_p` : The dropout probability of encoder.
- `decoder_dropout_p` : The dropout probability of decoder.
- `bidirectional` : If True, becomes a bidirectional encoders
- `rnn_type` : Type of rnn cell (rnn, lstm, gru)
- `output_dim` : Dimension of outputs
  
### `transformer_transducer`  
- `optimizer` : Optimizer for training.
- `model_name` : Model name
- `encoder_dim` : Dimension of encoder name
- `d_ff` : Dimension of feed forward network
- `num_audio_layers` : Number of audio layers
- `num_label_layers` : Number of label layers
- `num_attention_heads` : Number of attention heads
- `audio_dropout_p` : Dropout probability of audio layer
- `label_dropout_p` : Dropout probability of label layer
- `decoder_hidden_state_dim` : Hidden state dimension of decoder
- `decoder_output_dim` : Dimension of model output.
- `conv_kernel_size` : Kernel size of convolution layer.
- `max_positional_length` : Max length of positional encoding.
  
### `conformer_lstm`  
- `optimizer` : Optimizer for training.
- `model_name` : Model name
- `encoder_dim` : Dimension of encoder.
- `num_encoder_layers` : The number of encoder layers.
- `num_attention_heads` : The number of attention heads.
- `feed_forward_expansion_factor` : The expansion factor of feed forward module.
- `conv_expansion_factor` : The expansion factor of convolution module.
- `input_dropout_p` : The dropout probability of inputs.
- `feed_forward_dropout_p` : The dropout probability of feed forward module.
- `attention_dropout_p` : The dropout probability of attention module.
- `conv_dropout_p` : The dropout probability of convolution module.
- `conv_kernel_size` : The kernel size of convolution.
- `half_step_residual` : Flag indication whether to use half step residual or not
- `num_decoder_layers` : The number of decoder layers.
- `decoder_dropout_p` : The dropout probability of decoder.
- `max_length` : Max decoding length.
- `teacher_forcing_ratio` :  The ratio of teacher forcing. 
- `joint_ctc_attention` : Flag indication joint ctc attention or not
- `rnn_type` : Type of rnn cell (rnn, lstm, gru)
- `decoder_attn_mechanism` : The attention mechanism for decoder.
  
## `criterion`
  
### `label_smoothed_cross_entropy`  
- `reduction` : Reduction method of criterion
- `criterion_name` : Criterion name for training.
- `smoothing` : Ratio of smoothing loss (confidence = 1.0 - smoothing)
  
### `joint_ctc_cross_entropy`  
- `reduction` : Reduction method of criterion
- `criterion_name` : Criterion name for training.
- `ctc_weight` : Weight of ctc loss for training.
- `cross_entropy_weight` : Weight of cross entropy loss for training.
- `smoothing` : Ratio of smoothing loss (confidence = 1.0 - smoothing)
- `zero_infinity` : Whether to zero infinite losses and the associated gradients.
  
### `cross_entropy`  
- `reduction` : Reduction method of criterion
- `criterion_name` : Criterion name for training
  
### `transducer`  
- `reduction` : Reduction method of criterion
- `criterion_name` : Criterion name for training.
  
### `ctc`  
- `reduction` : Reduction method of criterion
- `criterion_name` : Criterion name for training
- `zero_infinity` : Whether to zero infinite losses and the associated gradients.
  
## `lr_scheduler`
  
### `reduce_lr_on_plateau`  
- `lr` : Learning rate
- `scheduler_name` : Name of learning rate scheduler.
- `lr_patience` : Number of epochs with no improvement after which learning rate will be reduced.
- `lr_factor` : Factor by which the learning rate will be reduced. new_lr = lr * factor.
  
### `warmup`  
- `lr` : Learning rate
- `scheduler_name` : Name of learning rate scheduler.
- `peak_lr` : Maximum learning rate.
- `init_lr` : Initial learning rate.
- `warmup_steps` : Warmup the learning rate linearly for the first N updates
- `total_steps` : Total training steps.
  
### `warmup_reduce_lr_on_plateau`  
- `lr` : Learning rate
- `scheduler_name` : Name of learning rate scheduler.
- `lr_patience` : Number of epochs with no improvement after which learning rate will be reduced.
- `lr_factor` : Factor by which the learning rate will be reduced. new_lr = lr * factor.
- `peak_lr` : Maximum learning rate.
- `init_lr` : Initial learning rate.
- `warmup_steps` : Warmup the learning rate linearly for the first N updates
  
### `tri_stage`  
- `lr` : Learning rate
- `scheduler_name` : Name of learning rate scheduler.
- `init_lr` : Initial learning rate.
- `peak_lr` : Maximum learning rate.
- `final_lr` : Final learning rate.
- `init_lr_scale` : Initial learning rate scale.
- `final_lr_scale` : Final learning rate scale
- `warmup_steps` : Warmup the learning rate linearly for the first N updates
- `hold_steps` : Hold the learning rate for the N updates
- `decay_steps` : Decay the learning rate linearly for the N updates
- `total_steps` : Total training steps.
  
### `transformer`  
- `lr` : Learning rate
- `scheduler_name` : Name of learning rate scheduler.
- `peak_lr` : Maximum learning rate.
- `final_lr` : Final learning rate.
- `final_lr_scale` : Final learning rate scale
- `warmup_steps` : Warmup the learning rate linearly for the first N updates
- `decay_steps` : Steps in decay stages
  
## `trainer`
  
### `cpu`  
- `seed` : Seed for training.
- `accelerator` : Previously known as distributed_backend (dp, ddp, ddp2, etc…).
- `accumulate_grad_batches` : Accumulates grads every k batches or as set up in the dict.
- `num_workers` : The number of cpu cores
- `batch_size` : Size of batch
- `check_val_every_n_epoch` : Check val every n train epochs.
- `gradient_clip_val` : 0 means don’t clip.
- `use_tensorboard` : If set to True, will use tensorboard log.
- `max_epochs` : Stop training once this number of epochs is reached.
- `auto_scale_batch_size` : If set to True, will initially run a batch size finder trying to find the largest batch size that fits into memory.
- `name` : Trainer name
- `device` : Training device.
- `use_cuda` : If set True, will train with GPU
  
### `gpu`  
- `seed` : Seed for training.
- `accelerator` : Previously known as distributed_backend (dp, ddp, ddp2, etc…).
- `accumulate_grad_batches` : Accumulates grads every k batches or as set up in the dict.
- `num_workers` : The number of cpu cores
- `batch_size` : Size of batch
- `check_val_every_n_epoch` : Check val every n train epochs.
- `gradient_clip_val` : 0 means don’t clip.
- `use_tensorboard` : If set to True, will use tensorboard log.
- `max_epochs` : Stop training once this number of epochs is reached.
- `auto_scale_batch_size` : If set to True, will initially run a batch size finder trying to find the largest batch size that fits into memory.
- `name` : Trainer name
- `device` : Training device.
- `use_cuda` : If set True, will train with GPU
- `auto_select_gpus` : If enabled and gpus is an integer, pick available gpus automatically.
  
### `tpu`  
- `seed` : Seed for training.
- `accelerator` : Previously known as distributed_backend (dp, ddp, ddp2, etc…).
- `accumulate_grad_batches` : Accumulates grads every k batches or as set up in the dict.
- `num_workers` : The number of cpu cores
- `batch_size` : Size of batch
- `check_val_every_n_epoch` : Check val every n train epochs.
- `gradient_clip_val` : 0 means don’t clip.
- `use_tensorboard` : If set to True, will use tensorboard log.
- `max_epochs` : Stop training once this number of epochs is reached.
- `auto_scale_batch_size` : If set to True, will initially run a batch size finder trying to find the largest batch size that fits into memory.
- `name` : Trainer name
- `device` : Training device.
- `use_cuda` : If set True, will train with GPU
- `use_tpu` : If set True, will train with GPU
- `tpu_cores` : Number of TPU cores
  
### `gpu-fp16`  
- `seed` : Seed for training.
- `accelerator` : Previously known as distributed_backend (dp, ddp, ddp2, etc…).
- `accumulate_grad_batches` : Accumulates grads every k batches or as set up in the dict.
- `num_workers` : The number of cpu cores
- `batch_size` : Size of batch
- `check_val_every_n_epoch` : Check val every n train epochs.
- `gradient_clip_val` : 0 means don’t clip.
- `use_tensorboard` : If set to True, will use tensorboard log.
- `max_epochs` : Stop training once this number of epochs is reached.
- `auto_scale_batch_size` : If set to True, will initially run a batch size finder trying to find the largest batch size that fits into memory.
- `name` : Trainer name
- `device` : Training device.
- `use_cuda` : If set True, will train with GPU
- `auto_select_gpus` : If enabled and gpus is an integer, pick available gpus automatically.
- `precision` : Double precision (64), full precision (32) or half precision (16). Can be used on CPU, GPU or TPUs.
- `amp_backend` : The mixed precision backend to use (“native” or “apex”)
  
### `tpu-fp16`  
- `seed` : Seed for training.
- `accelerator` : Previously known as distributed_backend (dp, ddp, ddp2, etc…).
- `accumulate_grad_batches` : Accumulates grads every k batches or as set up in the dict.
- `num_workers` : The number of cpu cores
- `batch_size` : Size of batch
- `check_val_every_n_epoch` : Check val every n train epochs.
- `gradient_clip_val` : 0 means don’t clip.
- `use_tensorboard` : If set to True, will use tensorboard log.
- `max_epochs` : Stop training once this number of epochs is reached.
- `auto_scale_batch_size` : If set to True, will initially run a batch size finder trying to find the largest batch size that fits into memory.
- `name` : Trainer name
- `device` : Training device.
- `use_cuda` : If set True, will train with GPU
- `use_tpu` : If set True, will train with GPU
- `tpu_cores` : Number of TPU cores
- `precision` : Double precision (64), full precision (32) or half precision (16). Can be used on CPU, GPU or TPUs.
- `amp_backend` : The mixed precision backend to use (“native” or “apex”)
  
### `cpu-fp64`  
- `seed` : Seed for training.
- `accelerator` : Previously known as distributed_backend (dp, ddp, ddp2, etc…).
- `accumulate_grad_batches` : Accumulates grads every k batches or as set up in the dict.
- `num_workers` : The number of cpu cores
- `batch_size` : Size of batch
- `check_val_every_n_epoch` : Check val every n train epochs.
- `gradient_clip_val` : 0 means don’t clip.
- `use_tensorboard` : If set to True, will use tensorboard log.
- `max_epochs` : Stop training once this number of epochs is reached.
- `auto_scale_batch_size` : If set to True, will initially run a batch size finder trying to find the largest batch size that fits into memory.
- `name` : Trainer name
- `device` : Training device.
- `use_cuda` : If set True, will train with GPU
- `precision` : Double precision (64), full precision (32) or half precision (16). Can be used on CPU, GPU or TPUs.
- `amp_backend` : The mixed precision backend to use (“native” or “apex”)
  
## `vocab`
  
### `libri_subword`  
- `sos_token` : Start of sentence token
- `eos_token` : End of sentence token
- `pad_token` : Pad token
- `blank_token` : Blank token (for CTC training)
- `encoding` : Encoding of vocab
- `unit` : Unit of vocabulary.
- `sp_model_path` : Path of sentencepiece model.
- `vocab_size` : Size of vocabulary.
- `vocab_path` : Path of vocabulary file.
  
### `libri_character`  
- `sos_token` : Start of sentence token
- `eos_token` : End of sentence token
- `pad_token` : Pad token
- `blank_token` : Blank token (for CTC training)
- `encoding` : Encoding of vocab
- `unit` : Unit of vocabulary.
- `vocab_path` : Path of vocabulary file.
  
### `aishell_character`  
- `sos_token` : Start of sentence token
- `eos_token` : End of sentence token
- `pad_token` : Pad token
- `blank_token` : Blank token (for CTC training)
- `encoding` : Encoding of vocab
- `unit` : Unit of vocabulary.
- `vocab_path` : Path of vocabulary file.
  
### `kspon_subword`  
- `sos_token` : Start of sentence token
- `eos_token` : End of sentence token
- `pad_token` : Pad token
- `blank_token` : Blank token (for CTC training)
- `encoding` : Encoding of vocab
- `unit` : Unit of vocabulary.
- `sp_model_path` : Path of sentencepiece model.
- `vocab_size` : Size of vocabulary.
  
### `kspon_grapheme`  
- `sos_token` : Start of sentence token
- `eos_token` : End of sentence token
- `pad_token` : Pad token
- `blank_token` : Blank token (for CTC training)
- `encoding` : Encoding of vocab
- `unit` : Unit of vocabulary.
- `vocab_path` : Path of vocabulary file.
  
### `kspon_character`  
- `sos_token` : Start of sentence token
- `eos_token` : End of sentence token
- `pad_token` : Pad token
- `blank_token` : Blank token (for CTC training)
- `encoding` : Encoding of vocab
- `unit` : Unit of vocabulary.
- `vocab_path` : Path of vocabulary file.
  