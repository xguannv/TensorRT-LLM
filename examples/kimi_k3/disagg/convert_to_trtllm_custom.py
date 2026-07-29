# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Convert a ``benchmarks/cpp/prepare_dataset.py`` workload into the
``trtllm_custom`` JSONL format consumed by the disaggregated benchmark client.

``prepare_dataset.py`` writes a single JSON object::

    {"metadata": {...}, "samples": [{"input_len": N, "input_ids": [...],
                                     "output_len": M, "task_id": ...}, ...]}

while ``CustomDataset`` (``tensorrt_llm/serve/scripts/benchmark_dataset.py``)
expects one minimal OpenAI-style request per line::

    {"input": {"messages": [{"role": "system", "content": ""},
                            {"role": "user", "content": [token ids]}],
               "max_tokens": M, "num_tokens": N}}

``num_tokens`` is mandatory on **every** line, not optional. ``CustomDataset``
only skips batch tokenization when it collects one positive ``num_tokens`` per
prompt; otherwise it falls back to tokenizing the prompt, which is a list of
token IDs here and would yield a wrong length or raise. Emitting the raw token
IDs as the user message keeps the request length exact end to end -- the client
never re-tokenizes it.

Typical Kimi K3 8K/1K use (from the repository root)::

    python benchmarks/cpp/prepare_dataset.py \
        --tokenizer "$MODEL" \
        --output kimi_k3_8192_1024_128req.json \
        --random-seed 420 \
        --trust-remote-code \
        token-unif-dist \
        --num-requests 128 \
        --input-min 8192 --input-max 8192 \
        --output-min 1024 --output-max 1024

    python examples/kimi_k3/disagg/convert_to_trtllm_custom.py \
        kimi_k3_8192_1024_128req.json \
        kimi_k3_8192_1024_128req_trtllm_custom.jsonl \
        --expect-num-requests 128 \
        --expect-input-len 8192 \
        --expect-output-len 1024

The ``--expect-*`` checks are advisory guards: they fail loudly when the
generated workload does not match the intended fixed-length shape.
"""

import argparse
import json
import sys


def convert(samples):
    """Yield one ``trtllm_custom`` record per ``prepare_dataset.py`` sample."""
    for i, sample in enumerate(samples):
        try:
            input_ids = sample["input_ids"]
            output_len = sample["output_len"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"sample {i} is missing {exc}; expected the "
                             "prepare_dataset.py 'samples' schema") from exc

        # input_len is informational in the source workload; the authoritative
        # length is len(input_ids), which is what the server actually receives.
        num_tokens = len(input_ids)
        declared = sample.get("input_len")
        if declared is not None and declared != num_tokens:
            raise ValueError(
                f"sample {i}: input_len={declared} disagrees with "
                f"len(input_ids)={num_tokens}")

        yield {
            "input": {
                "messages": [
                    {
                        "role": "system",
                        "content": ""
                    },
                    {
                        "role": "user",
                        "content": input_ids
                    },
                ],
                "max_tokens": output_len,
                # Mandatory: keeps CustomDataset from re-tokenizing the IDs.
                "num_tokens": num_tokens,
            }
        }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source",
                        help="JSON workload written by prepare_dataset.py")
    parser.add_argument("destination", help="trtllm_custom JSONL to write")
    parser.add_argument("--expect-num-requests",
                        type=int,
                        help="fail unless exactly this many records are written")
    parser.add_argument("--expect-input-len",
                        type=int,
                        help="fail unless every record has this input length")
    parser.add_argument("--expect-output-len",
                        type=int,
                        help="fail unless every record requests this many "
                        "output tokens")
    args = parser.parse_args(argv)

    with open(args.source, encoding="utf-8") as src:
        workload = json.load(src)

    if "samples" not in workload:
        parser.error(f"{args.source} has no 'samples' key; it does not look "
                     "like prepare_dataset.py output")

    records = list(convert(workload["samples"]))

    for i, record in enumerate(records):
        payload = record["input"]
        if (args.expect_input_len is not None
                and payload["num_tokens"] != args.expect_input_len):
            parser.error(f"record {i}: input length {payload['num_tokens']} "
                         f"!= --expect-input-len {args.expect_input_len}")
        if (args.expect_output_len is not None
                and payload["max_tokens"] != args.expect_output_len):
            parser.error(f"record {i}: max_tokens {payload['max_tokens']} "
                         f"!= --expect-output-len {args.expect_output_len}")

    if (args.expect_num_requests is not None
            and len(records) != args.expect_num_requests):
        parser.error(f"wrote {len(records)} records != "
                     f"--expect-num-requests {args.expect_num_requests}")

    with open(args.destination, "w", encoding="utf-8") as dst:
        for record in records:
            dst.write(json.dumps(record) + "\n")

    metadata = workload.get("metadata", {})
    print(f"{args.destination}: {len(records)} records")
    if metadata:
        print(f"  source workload: {metadata.get('workload_type')} "
              f"input {metadata.get('input_min')}-{metadata.get('input_max')} "
              f"output {metadata.get('output_min')}-{metadata.get('output_max')}"
              )
    return 0


if __name__ == "__main__":
    sys.exit(main())
