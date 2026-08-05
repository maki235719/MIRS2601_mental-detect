"""collect モードの STAI-S full/short6 切替の自己チェック。pytest不要、`python test_stai_s_form.py` で実行。"""
import builtins
import os
import tempfile

import peroson_detect as pd

SUMMARIES = {1: {"samples": 5, "sum": 250.0, "first_t": 0.0, "last_t": 10.0,
                  "z_sum": {k: 0.0 for k in pd._FEATURES}}}


def _run(inputs):
    it = iter(inputs)
    builtins_input = builtins.input
    builtins.input = lambda *_a, **_kw: next(it, "")
    try:
        return pd.collect_stai_labels(SUMMARIES)
    finally:
        builtins.input = builtins_input


def main():
    with tempfile.TemporaryDirectory() as tmp:
        pd.STAI_DATASET_PATH = os.path.join(tmp, "stai_dataset.jsonl")
        pd.STAI_CSV_PATH = os.path.join(tmp, "stai_dataset.csv")

        pd.STAI_S_FORM = "short6"
        result = _run(["12", ""])
        assert result[1]["stai_state"] == 12 * (20.0 / 6.0), result

        pd.STAI_S_FORM = "short6"
        result = _run(["30", ""])  # out of range for short6 (6-24) -> skipped
        assert 1 not in result, result

        pd.STAI_S_FORM = "full"
        result = _run(["55", ""])
        assert result[1]["stai_state"] == 55.0, result

    print("OK: STAI_S_FORM full/short6 の切替と正規化を確認しました。")


if __name__ == "__main__":
    main()
