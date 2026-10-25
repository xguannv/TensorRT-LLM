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
"""Response-path stamps (TRTLLM_K3LL_RESP_STAMPS): off by default, one JSON line per request and hop when on."""

import glob
import importlib
import json

from tensorrt_llm._torch.pyexecutor import resp_stamps


def _reload(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(resp_stamps.ENV, raising=False)
    else:
        monkeypatch.setenv(resp_stamps.ENV, value)
    return importlib.reload(resp_stamps)


def test_off_by_default(monkeypatch, tmp_path):
    for value in (None, "", "0"):
        mod = _reload(monkeypatch, value)
        assert mod.ENABLED is False
        mod.mark("E", 1, False)
        mod.mark("E", 1, True)
        assert not mod._open and not mod._files
    _reload(monkeypatch, None)


def test_first_and_final_per_request(monkeypatch, tmp_path):
    mod = _reload(monkeypatch, str(tmp_path))
    try:
        assert mod.ENABLED
        mod.mark("E", 7, False, cid=70)
        mod.mark("E", 8, False)
        mod.mark("E", 7, False)
        mod.mark("E", 7, True)
        mod.mark("S", 7, True, xrid="abc")
        mod.mark("E", None, True)
        files = sorted(glob.glob(str(tmp_path / "*.jsonl")))
        assert [f.split("/")[-1][0] for f in files] == ["E", "S"]
        e = [json.loads(x) for x in open(files[0])]
        assert len(e) == 1 and e[0]["key"] == 7 and e[0]["n"] == 3 and e[0]["cid"] == 70
        assert e[0]["final_ns"] >= e[0]["first_ns"]
        s = [json.loads(x) for x in open(files[1])]
        assert s[0]["xrid"] == "abc" and s[0]["first_ns"] == s[0]["final_ns"]
        assert (("E", 8) in mod._open) and len(mod._open) == 1
    finally:
        for f in mod._files.values():
            f.close()
        _reload(monkeypatch, None)
