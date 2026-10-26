# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Added tokens past the model vocabulary tokenize as plain text (CPU, no checkpoint).

The Kimi ``TikTokenTokenizer`` registers ChatML-style strings such as ``<|im_end|>`` as added special tokens at ids
past the model's embedding rows when Transformers constructs it without ``additional_special_tokens``. A prompt
quoting one of them then tokenizes to an id the model cannot embed. The Kimi K2.5 / K3 input processor unregisters
those entries (``drop_out_of_vocab_added_tokens``), so the string becomes the tokenizer's native plain-text ids; text
without it, and added tokens inside the vocabulary, tokenize as before.
"""

import tiktoken
from transformers import PreTrainedTokenizer

from tensorrt_llm.tokenizer.tokenizer import TransformersTokenizer, drop_out_of_vocab_added_tokens

TEXT = 'The template ends each turn with "<|im_end|>".'
OTHER = "no marker here, only [BOS] and text"


class TikTokenTokenizer(PreTrainedTokenizer):
    """A byte-level tiktoken vocabulary with ``[BOS]`` and an out-of-vocabulary ``<|im_end|>``.

    256 byte ids plus ``[BOS]`` (id 256) inside the vocabulary, like the checkpoint's tiktoken specials;
    ``<|im_end|>`` is added past it, as Transformers 5.x does for the checkpoint.
    """

    def __init__(self) -> None:
        ranks = {bytes([i]): i for i in range(256)}
        self.model = tiktoken.Encoding(
            name="kimi-test",
            pat_str=r"\s+|\S+",
            mergeable_ranks=ranks,
            special_tokens={"[BOS]": 256},
        )
        self.decoder = {i: piece.hex() for piece, i in ranks.items()}
        self.decoder[256] = "[BOS]"
        self.encoder = {piece: i for i, piece in self.decoder.items()}
        super().__init__(model_max_length=100000)
        self.add_tokens(["[BOS]"], special_tokens=True)  # in the vocabulary: id 256
        self.add_tokens(["<|im_end|>"], special_tokens=True)  # past it: id 257

    @property
    def vocab_size(self) -> int:
        return len(self.encoder)

    def get_vocab(self) -> dict[str, int]:
        return dict(self.encoder, **self.added_tokens_encoder)

    def _tokenize(self, text: str, **kwargs) -> list[str]:
        return [self.decoder[i] for i in self.model.encode(text, allowed_special="all")]

    def _convert_token_to_id(self, token: str) -> int:
        return self.encoder[token]

    def _convert_id_to_token(self, index: int) -> str:
        return self.decoder[index]

    def encode(self, text, *args, **kwargs) -> list[int]:
        if args or kwargs:
            return super().encode(text, *args, **kwargs)
        return self.model.encode(text, allowed_special="all")


# Match the checkpoint class identity checked by the production guard.
TikTokenTokenizer.__module__ = "tests.tokenization_kimi"


def test_repro_without_the_fix() -> None:
    tok = TikTokenTokenizer()
    assert tok.vocab_size == 257
    assert tok.added_tokens_encoder["<|im_end|>"] == 257
    assert 257 in tok(TEXT).input_ids  # the id no embedding row has


def test_out_of_vocab_added_token_is_plain_text() -> None:
    reference = TikTokenTokenizer()
    plain = reference.encode(TEXT)
    other = reference(OTHER).input_ids
    tok = TikTokenTokenizer()
    assert drop_out_of_vocab_added_tokens(tok, 257) == ["<|im_end|>"]
    ids = tok(TEXT).input_ids
    assert ids == plain and max(ids) < 257
    assert tok.encode(TEXT, add_special_tokens=False) == plain
    assert "<|im_end|>" not in tok.added_tokens_encoder
    assert "<|im_end|>" not in tok.all_special_tokens
    # Text without the string, including an in-vocabulary added token, is unchanged.
    assert tok(OTHER).input_ids == other
    assert tok.added_tokens_encoder["[BOS]"] == 256
    assert 256 in tok(OTHER).input_ids
    assert drop_out_of_vocab_added_tokens(tok, 257) == []


def test_wrapped_tokenizer() -> None:
    wrapper = TransformersTokenizer(TikTokenTokenizer())
    assert (
        "<|im_end|>" in wrapper.tokenizer.added_tokens_encoder
    )  # the wrapper alone changes nothing
    assert drop_out_of_vocab_added_tokens(wrapper, 257) == ["<|im_end|>"]
    assert wrapper(TEXT).input_ids == TikTokenTokenizer().encode(TEXT)
    assert "<|im_end|>" not in wrapper._all_special_tokens_set
    assert 257 not in wrapper._all_special_ids_set


def test_tokens_inside_a_larger_model_vocabulary_are_kept() -> None:
    tok = TikTokenTokenizer()
    assert drop_out_of_vocab_added_tokens(tok, 258) == []
    assert tok.added_tokens_encoder["<|im_end|>"] == 257
    assert 257 in tok(TEXT).input_ids
