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

import os
import re
from joblib import Parallel, delayed, cpu_count
from tqdm import tqdm


def bracket_filter(sentence, mode='phonetic'):
    new_sentence = str()

    if mode == 'phonetic':
        flag = False

        for ch in sentence:
            if ch == '(' and flag is False:
                flag = True
                continue
            if ch == '(' and flag is True:
                flag = False
                continue
            if ch != ')' and flag is False:
                new_sentence += ch

    elif mode == 'spelling':
        flag = True

        for ch in sentence:
            if ch == '(':
                continue
            if ch == ')':
                if flag is True:
                    flag = False
                    continue
                else:
                    flag = True
                    continue
            if ch != ')' and flag is True:
                new_sentence += ch

    else:
        raise ValueError("Unsupported mode : {0}".format(mode))

    return new_sentence


def special_filter(sentence, mode='phonetic', replace=None):
    SENTENCE_MARK = ['?', '!', '.']
    NOISE = ['o', 'n', 'u', 'b', 'l']
    EXCEPT = ['/', '+', '*', '-', '@', '$', '^', '&', '[', ']', '=', ':', ';', ',']

    new_sentence = str()
    for idx, ch in enumerate(sentence):
        if ch not in SENTENCE_MARK:
            if idx + 1 < len(sentence) and ch in NOISE and sentence[idx + 1] == '/':
                continue

        if ch == '#':
            new_sentence += '샾'

        elif ch == '%':
            if mode == 'phonetic':
                new_sentence += replace
            elif mode == 'spelling':
                new_sentence += '%'

        elif ch not in EXCEPT:
            new_sentence += ch

    pattern = re.compile(r'\s\s+')
    new_sentence = re.sub(pattern, ' ', new_sentence.strip())
    return new_sentence


def sentence_filter(raw_sentence, mode, replace=None):
    return special_filter(bracket_filter(raw_sentence, mode), mode, replace)


PERCENT_FILES = {
    '087797': '퍼센트',
    '215401': '퍼센트',
    '284574': '퍼센트',
    '397184': '퍼센트',
    '501006': '프로',
    '502173': '프로',
    '542363': '프로',
    '581483': '퍼센트'
}


def read_preprocess_text_file(file_path, mode):
    with open(file_path, 'r', encoding='cp949') as f:
        raw_sentence = f.read()
        file_name = os.path.basename(file_path)
        if file_name[12:18] in PERCENT_FILES.keys():
            replace = PERCENT_FILES[file_name[12:18]]
        else:
            replace = None
        return sentence_filter(raw_sentence, mode=mode, replace=replace)


def preprocess(dataset_path, mode='phonetic'):
    print('preprocess started..')

    audio_paths = list()
    transcripts = list()

    with Parallel(n_jobs=cpu_count() - 1) as parallel:

        for folder in os.listdir(dataset_path):
            # folder : {KsponSpeech_01, ..., KsponSpeech_05}
            if not folder.startswith('KsponSpeech'):
                continue
            path = os.path.join(dataset_path, folder)
            for idx, subfolder in tqdm(list(enumerate(os.listdir(path))), desc=f'Preprocess text files on {path}'):
                path = os.path.join(dataset_path, folder, subfolder)

                # list-up files
                sub_file_list = [
                    os.path.join(path, file_name) for file_name in os.listdir(path) if file_name.endswith('.txt')
                ]
                audio_sub_file_list = [
                    os.path.join(folder, subfolder, file_name)
                    for file_name in os.listdir(path) if file_name.endswith('.txt')
                ]

                # do parallel and get results
                new_sentences = parallel(
                    delayed(read_preprocess_text_file)(p, mode) for p in sub_file_list
                )

                audio_paths.extend(audio_sub_file_list)
                transcripts.extend(new_sentences)

    return audio_paths, transcripts


def preprocess_test_data(manifest_file_dir: str, mode='phonetic'):
    audio_paths = list()
    transcripts = list()

    for split in ("eval_clean.trn", "eval_other.trn"):
        with open(os.path.join(manifest_file_dir, split), encoding='utf-8') as f:
            for line in f.readlines():
                audio_path, raw_transcript = line.split(" :: ")
                transcript = sentence_filter(raw_transcript, mode=mode)

                audio_paths.append(audio_path)
                transcripts.append(transcript)

    return audio_paths, transcripts
