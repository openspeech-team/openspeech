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

import importlib
import os

from openspeech.optim.adamp import AdamP
from openspeech.optim.radam import RAdam
from openspeech.optim.novograd import Novograd

# automatically import any Python files in the models/ directory
scheduler_dir = os.path.dirname(__file__)
for file in os.listdir(scheduler_dir):
    if os.path.isdir(os.path.join(scheduler_dir, file)) and file != "__pycache__":
        for subfile in os.listdir(os.path.join(scheduler_dir, file)):
            path = os.path.join(scheduler_dir, file, subfile)
            if subfile.endswith(".py"):
                scheduler_name = subfile[: subfile.find(".py")] if subfile.endswith(".py") else subfile
                module = importlib.import_module(f"openspeech.optim.scheduler.{scheduler_name}")
