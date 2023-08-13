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

import unicodedata

import pandas as pd


def load_label(filepath):
    grpm2id = dict()
    id2grpm = dict()

    vocab_data_frame = pd.read_csv(filepath, encoding="utf-8")

    id_list = vocab_data_frame["id"]
    grpm_list = vocab_data_frame["grpm"]

    for _id, grpm in zip(id_list, grpm_list):
        grpm2id[grpm] = _id
        id2grpm[_id] = grpm
    return grpm2id, id2grpm


def sentence_to_target(transcript, grpm2id):
    target = str()

    for grpm in transcript:
        target += str(grpm2id[grpm]) + " "

    return target[:-1]


def sentence_to_grapheme(audio_paths, transcripts, manifest_file_path: str, vocab_path: str):
    grapheme_transcripts = list()

    for transcript in transcripts:
        grapheme_transcripts.append(" ".join(unicodedata.normalize("NFKD", transcript).replace(" ", "|")).upper())

    generate_grapheme_labels(grapheme_transcripts, vocab_path)

    print("create_script started..")
    grpm2id, id2grpm = load_label(vocab_path)

    with open(manifest_file_path, "w") as f:
        for audio_path, transcript, grapheme_transcript in zip(audio_paths, transcripts, grapheme_transcripts):
            audio_path = audio_path.replace("txt", "pcm")
            grpm_id_transcript = sentence_to_target(grapheme_transcript.split(), grpm2id)
            f.write(f"{audio_path}\t{transcript}\t{grpm_id_transcript}\n")


def generate_grapheme_labels(grapheme_transcripts, vocab_path: str):
    vocab_list = list()
    vocab_freq = list()

    for grapheme_transcript in grapheme_transcripts:
        graphemes = grapheme_transcript.split()
        for grapheme in graphemes:
            if grapheme not in vocab_list:
                vocab_list.append(grapheme)
                vocab_freq.append(1)
            else:
                vocab_freq[vocab_list.index(grapheme)] += 1

    vocab_freq, vocab_list = zip(*sorted(zip(vocab_freq, vocab_list), reverse=True))
    vocab_dict = {"id": [0, 1, 2, 3], "grpm": ["<pad>", "<sos>", "<eos>", "<blank>"], "freq": [0, 0, 0, 0]}

    for idx, (grpm, freq) in enumerate(zip(vocab_list, vocab_freq)):
        vocab_dict["id"].append(idx + 4)
        vocab_dict["grpm"].append(grpm)
        vocab_dict["freq"].append(freq)

    label_df = pd.DataFrame(vocab_dict)
    label_df.to_csv(vocab_path, encoding="utf-8", index=False)
