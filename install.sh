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

# Recommended: install with uv (https://github.com/astral-sh/uv)
# Creates a .venv and installs everything pinned by uv.lock.
uv sync                 # runtime dependencies only
# uv sync --extra dev   # + dev tools (pytest, black, ruff, pre-commit)
# uv sync --extra wandb  # + Weights & Biases logger

# Alternative (pip):
# pip install -e .          # runtime
# pip install -e ".[dev]"   # + dev tools

# ---------------------------------------------------------------------------
# OPTIONAL native libraries (NOT required for the default install).
# Only needed for warp-rnnt loss / ctcdecode beam search on supported platforms.
#
# warp-rnnt — https://github.com/1ytic/warp-rnnt
# git clone https://github.com/1ytic/warp-rnnt
# cd warp-rnnt/pytorch_binding && python setup.py install && cd ../..
#
# ctcdecode — https://github.com/parlance/ctcdecode
# git clone --recursive https://github.com/parlance/ctcdecode.git
# cd ctcdecode && pip install .
