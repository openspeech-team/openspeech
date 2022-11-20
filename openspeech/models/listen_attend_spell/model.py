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

from omegaconf import DictConfig

from openspeech.decoders import LSTMAttentionDecoder
from openspeech.encoders import ConvolutionalLSTMEncoder, LSTMEncoder
from openspeech.models import OpenspeechEncoderDecoderModel, register_model
from openspeech.models.listen_attend_spell.configurations import (
    DeepCNNWithJointCTCListenAttendSpellConfigs,
    JointCTCListenAttendSpellConfigs,
    ListenAttendSpellConfigs,
    ListenAttendSpellWithLocationAwareConfigs,
    ListenAttendSpellWithMultiHeadConfigs,
)
from openspeech.tokenizers.tokenizer import Tokenizer


@register_model("listen_attend_spell", dataclass=ListenAttendSpellConfigs)
class ListenAttendSpellModel(OpenspeechEncoderDecoderModel):
    r"""
    Listen, Attend and Spell model with configurable encoder and decoder.
    Paper: https://arxiv.org/abs/1508.01211

    Args:
        configs (DictConfig): configuration set.
        tokenizer (Tokeizer): tokenizer is in charge of preparing the inputs for a model.

    Inputs:
        - **inputs** (torch.FloatTensor): A input sequence passed to encoders. Typically for inputs this will be a padded `FloatTensor` of size ``(batch, seq_length, dimension)``.
        - **input_lengths** (torch.LongTensor): The length of input tensor. ``(batch)``

    Returns:
        outputs (dict): Result of model predictions.
    """

    def __init__(self, configs: DictConfig, tokenizer: Tokenizer) -> None:
        super(ListenAttendSpellModel, self).__init__(configs, tokenizer)

        self.encoder = LSTMEncoder(
            input_dim=self.configs.audio.num_mels,
            num_layers=self.configs.model.num_encoder_layers,
            num_classes=self.num_classes,
            hidden_state_dim=self.configs.model.hidden_state_dim,
            dropout_p=self.configs.model.encoder_dropout_p,
            bidirectional=self.configs.model.encoder_bidirectional,
            rnn_type=self.configs.model.rnn_type,
            joint_ctc_attention=self.configs.model.joint_ctc_attention,
        )
        decoder_hidden_state_dim = (
            self.configs.model.hidden_state_dim << 1
            if self.configs.model.encoder_bidirectional
            else self.configs.model.hidden_state_dim
        )
        self.decoder = LSTMAttentionDecoder(
            num_classes=self.num_classes,
            max_length=self.configs.model.max_length,
            hidden_state_dim=decoder_hidden_state_dim,
            pad_id=self.tokenizer.pad_id,
            sos_id=self.tokenizer.sos_id,
            eos_id=self.tokenizer.eos_id,
            num_heads=self.configs.model.num_attention_heads,
            dropout_p=self.configs.model.decoder_dropout_p,
            num_layers=self.configs.model.num_decoder_layers,
            attn_mechanism=self.configs.model.decoder_attn_mechanism,
            rnn_type=self.configs.model.rnn_type,
        )

    def set_beam_decoder(self, beam_size: int = 3):
        """Setting beam search decoder"""
        from openspeech.search import BeamSearchLSTM

        self.decoder = BeamSearchLSTM(
            decoder=self.decoder,
            beam_size=beam_size,
        )


@register_model("listen_attend_spell_with_location_aware", dataclass=ListenAttendSpellWithLocationAwareConfigs)
class ListenAttendSpellWithLocationAwareModel(OpenspeechEncoderDecoderModel):
    r"""
    Listen, Attend and Spell model with configurable encoder and decoder.
    Paper: https://arxiv.org/abs/1508.01211

    Args:
        configs (DictConfig): configuration set.
        tokenizer (Tokeizer): tokenizer is in charge of preparing the inputs for a model.

    Inputs:
        - **inputs** (torch.FloatTensor): A input sequence passed to encoders. Typically for inputs this will be a padded `FloatTensor` of size ``(batch, seq_length, dimension)``.
        - **input_lengths** (torch.LongTensor): The length of input tensor. ``(batch)``

    Returns:
        outputs (dict): Result of model predictions.
    """

    def __init__(self, configs: DictConfig, tokenizer: Tokenizer) -> None:
        super(ListenAttendSpellWithLocationAwareModel, self).__init__(configs, tokenizer)

        self.encoder = LSTMEncoder(
            input_dim=self.configs.audio.num_mels,
            num_layers=self.configs.model.num_encoder_layers,
            num_classes=self.num_classes,
            hidden_state_dim=self.configs.model.hidden_state_dim,
            dropout_p=self.configs.model.encoder_dropout_p,
            bidirectional=self.configs.model.encoder_bidirectional,
            rnn_type=self.configs.model.rnn_type,
            joint_ctc_attention=self.configs.model.joint_ctc_attention,
        )
        decoder_hidden_state_dim = (
            self.configs.model.hidden_state_dim << 1
            if self.configs.model.encoder_bidirectional
            else self.configs.model.hidden_state_dim
        )
        self.decoder = LSTMAttentionDecoder(
            num_classes=self.num_classes,
            max_length=self.configs.model.max_length,
            hidden_state_dim=decoder_hidden_state_dim,
            pad_id=self.tokenizer.pad_id,
            sos_id=self.tokenizer.sos_id,
            eos_id=self.tokenizer.eos_id,
            num_heads=self.configs.model.num_attention_heads,
            dropout_p=self.configs.model.decoder_dropout_p,
            num_layers=self.configs.model.num_decoder_layers,
            attn_mechanism=self.configs.model.decoder_attn_mechanism,
            rnn_type=self.configs.model.rnn_type,
        )

    def set_beam_decoder(self, beam_size: int = 3):
        """Setting beam search decoder"""
        from openspeech.search import BeamSearchLSTM

        self.decoder = BeamSearchLSTM(
            decoder=self.decoder,
            beam_size=beam_size,
        )


@register_model("listen_attend_spell_with_multi_head", dataclass=ListenAttendSpellWithMultiHeadConfigs)
class ListenAttendSpellWithMultiHeadModel(OpenspeechEncoderDecoderModel):
    r"""
    Listen, Attend and Spell model with configurable encoder and decoder.
    Paper: https://arxiv.org/abs/1508.01211

    Args:
        configs (DictConfig): configuration set.
        tokenizer (Tokeizer): tokenizer is in charge of preparing the inputs for a model.

    Inputs:
        - **inputs** (torch.FloatTensor): A input sequence passed to encoders. Typically for inputs this will be a padded `FloatTensor` of size ``(batch, seq_length, dimension)``.
        - **input_lengths** (torch.LongTensor): The length of input tensor. ``(batch)``

    Returns:
        outputs (dict): Result of model predictions.
    """

    def __init__(self, configs: DictConfig, tokenizer: Tokenizer) -> None:
        super(ListenAttendSpellWithMultiHeadModel, self).__init__(configs, tokenizer)

        self.encoder = LSTMEncoder(
            input_dim=self.configs.audio.num_mels,
            num_layers=self.configs.model.num_encoder_layers,
            num_classes=self.num_classes,
            hidden_state_dim=self.configs.model.hidden_state_dim,
            dropout_p=self.configs.model.encoder_dropout_p,
            bidirectional=self.configs.model.encoder_bidirectional,
            rnn_type=self.configs.model.rnn_type,
            joint_ctc_attention=self.configs.model.joint_ctc_attention,
        )
        decoder_hidden_state_dim = (
            self.configs.model.hidden_state_dim << 1
            if self.configs.model.encoder_bidirectional
            else self.configs.model.hidden_state_dim
        )
        self.decoder = LSTMAttentionDecoder(
            num_classes=self.num_classes,
            max_length=self.configs.model.max_length,
            hidden_state_dim=decoder_hidden_state_dim,
            pad_id=self.tokenizer.pad_id,
            sos_id=self.tokenizer.sos_id,
            eos_id=self.tokenizer.eos_id,
            num_heads=self.configs.model.num_attention_heads,
            dropout_p=self.configs.model.decoder_dropout_p,
            num_layers=self.configs.model.num_decoder_layers,
            attn_mechanism=self.configs.model.decoder_attn_mechanism,
            rnn_type=self.configs.model.rnn_type,
        )

    def set_beam_decoder(self, beam_size: int = 3):
        """Setting beam search decoder"""
        from openspeech.search import BeamSearchLSTM

        self.decoder = BeamSearchLSTM(
            decoder=self.decoder,
            beam_size=beam_size,
        )


@register_model("joint_ctc_listen_attend_spell", dataclass=JointCTCListenAttendSpellConfigs)
class JointCTCListenAttendSpellModel(OpenspeechEncoderDecoderModel):
    r"""
    Joint CTC-Attention Listen, Attend and Spell model with configurable encoder and decoder.
    Paper: https://arxiv.org/abs/1609.06773

    Args:
        configs (DictConfig): configuration set.
        tokenizer (Tokeizer): tokenizer is in charge of preparing the inputs for a model.

    Inputs:
        - **inputs** (torch.FloatTensor): A input sequence passed to encoders. Typically for inputs this will be a padded `FloatTensor` of size ``(batch, seq_length, dimension)``.
        - **input_lengths** (torch.LongTensor): The length of input tensor. ``(batch)``

    Returns:
        outputs (dict): Result of model predictions.
    """

    def __init__(self, configs: DictConfig, tokenizer: Tokenizer) -> None:
        super(JointCTCListenAttendSpellModel, self).__init__(configs, tokenizer)

        self.encoder = LSTMEncoder(
            input_dim=self.configs.audio.num_mels,
            num_layers=self.configs.model.num_encoder_layers,
            num_classes=self.num_classes,
            hidden_state_dim=self.configs.model.hidden_state_dim,
            dropout_p=self.configs.model.encoder_dropout_p,
            bidirectional=self.configs.model.encoder_bidirectional,
            rnn_type=self.configs.model.rnn_type,
            joint_ctc_attention=self.configs.model.joint_ctc_attention,
        )
        decoder_hidden_state_dim = (
            self.configs.model.hidden_state_dim << 1
            if self.configs.model.encoder_bidirectional
            else self.configs.model.hidden_state_dim
        )
        self.decoder = LSTMAttentionDecoder(
            num_classes=self.num_classes,
            max_length=self.configs.model.max_length,
            hidden_state_dim=decoder_hidden_state_dim,
            pad_id=self.tokenizer.pad_id,
            sos_id=self.tokenizer.sos_id,
            eos_id=self.tokenizer.eos_id,
            num_heads=self.configs.model.num_attention_heads,
            dropout_p=self.configs.model.decoder_dropout_p,
            num_layers=self.configs.model.num_decoder_layers,
            attn_mechanism=self.configs.model.decoder_attn_mechanism,
            rnn_type=self.configs.model.rnn_type,
        )

    def set_beam_decoder(self, beam_size: int = 3):
        """Setting beam search decoder"""
        from openspeech.search.beam_search_lstm import BeamSearchLSTM

        self.decoder = BeamSearchLSTM(
            decoder=self.decoder,
            beam_size=beam_size,
        )


@register_model("deep_cnn_with_joint_ctc_listen_attend_spell", dataclass=DeepCNNWithJointCTCListenAttendSpellConfigs)
class DeepCNNWithJointCTCListenAttendSpellModel(OpenspeechEncoderDecoderModel):
    r"""
    Listen, Attend and Spell model with configurable encoder and decoder.
    Paper: https://arxiv.org/abs/1508.01211

    Args:
        configs (DictConfig): configuration set.
        tokenizer (Tokenizer): tokenizer is in charge of preparing the inputs for a model.

    Inputs:
        - **inputs** (torch.FloatTensor): A input sequence passed to encoders. Typically for inputs this will be a padded `FloatTensor` of size ``(batch, seq_length, dimension)``.
        - **input_lengths** (torch.LongTensor): The length of input tensor. ``(batch)``

    Returns:
        outputs (dict): Result of model predictions.
    """

    def __init__(self, configs: DictConfig, tokenizer: Tokenizer) -> None:
        super(DeepCNNWithJointCTCListenAttendSpellModel, self).__init__(configs, tokenizer)

        self.encoder = ConvolutionalLSTMEncoder(
            input_dim=self.configs.audio.num_mels,
            num_layers=self.configs.model.num_encoder_layers,
            num_classes=self.num_classes,
            hidden_state_dim=self.configs.model.hidden_state_dim,
            dropout_p=self.configs.model.encoder_dropout_p,
            bidirectional=self.configs.model.encoder_bidirectional,
            rnn_type=self.configs.model.rnn_type,
            joint_ctc_attention=self.configs.model.joint_ctc_attention,
        )
        decoder_hidden_state_dim = (
            self.configs.model.hidden_state_dim << 1
            if self.configs.model.encoder_bidirectional
            else self.configs.model.hidden_state_dim
        )
        self.decoder = LSTMAttentionDecoder(
            num_classes=self.num_classes,
            max_length=self.configs.model.max_length,
            hidden_state_dim=decoder_hidden_state_dim,
            pad_id=self.tokenizer.pad_id,
            sos_id=self.tokenizer.sos_id,
            eos_id=self.tokenizer.eos_id,
            num_heads=self.configs.model.num_attention_heads,
            dropout_p=self.configs.model.decoder_dropout_p,
            num_layers=self.configs.model.num_decoder_layers,
            attn_mechanism=self.configs.model.decoder_attn_mechanism,
            rnn_type=self.configs.model.rnn_type,
        )

    def set_beam_decoder(self, beam_size: int = 3):
        """Setting beam search decoder"""
        from openspeech.search import BeamSearchLSTM

        self.decoder = BeamSearchLSTM(
            decoder=self.decoder,
            beam_size=beam_size,
        )
