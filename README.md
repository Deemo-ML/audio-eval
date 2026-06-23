# audio-eval

Audio-only TRIDENT-style evaluation toolkit adapted from [`j1anglin/TRIDENT_MM26MGC`](https://github.com/j1anglin/TRIDENT_MM26MGC) for evaluating local result folders with a Qwen3.5-4B text evaluator.

This repository focuses on the **audio track** and reproduces the core scoring flow used by the TRIDENT starter kit:

1. Read predictions from a folder.
2. Score structured tasks:
   - TFQ: accuracy.
   - MCQ: penalty-aware multi-select score.
   - Type-B OEQ: detection accuracy from `Likely Authentic` / `Likely Manipulated`.
3. Map Type-A and Type-B OEQ free-form explanations into a fixed audio artifact checklist using Qwen3.5-4B.
4. Compute artifact metrics: Cover, CHAIR, hallucination rate, and F0.5.
5. Compute the audio TCS:

```text
tcs = 0.4 * (100 * acc_det)
    + 0.3 * (100 * ((typea_f_0_5 + typeb_f_0_5) / 2))
    + 0.3 * (100 * (0.5 * acc_tfq + 0.5 * score_mcq))
```

## Install

```bash
pip install -r requirements.txt
```

For GPU inference, install the PyTorch build that matches your CUDA version before installing `transformers` dependencies.

## Expected data layout

The evaluator follows the TRIDENT starter-kit style layout:

```text
DATA_ROOT/
  TFQ/public_val/answers.jsonl
  MCQ/public_val/answers.jsonl
  OEQ/public_val/answers_audio.csv

PREDICTIONS_ROOT/
  tfq/MODEL_NAME/public_val/*.jsonl|*.json
  mcq/MODEL_NAME/public_val/*.jsonl|*.json
  typea_oeq/MODEL_NAME/public_val/*.jsonl|*.json
  typeb_oeq/MODEL_NAME/public_val/*.jsonl|*.json
```

`typea_oeq` also accepts `perception_oeq`, and `typeb_oeq` also accepts `detection_oeq` as directory aliases.

## Prediction record format

Recommended fields are:

```json
{"question_id": "...", "parsed_answer": "True", "response": "True"}
{"question_id": "...", "parsed_choices": ["A", "C"], "response": "A, C"}
{"sample_id": "...", "response": "Likely Manipulated. Observable artifacts include Hiss and Unnatural Prosody."}
```

For Type-B OEQ, detection is parsed from either `parsed_label` (`real` or `fake`) or from the first non-empty line of `response`, which should contain `Likely Authentic` or `Likely Manipulated`.

## Run evaluation with Qwen3.5-4B

```bash
python scripts/evaluate_audio_predictions.py \
  --task all \
  --split public_val \
  --data-root /path/to/TRIDENT/data \
  --predictions-root /path/to/outputs \
  --model MODEL_NAME \
  --qwen-model /path/to/Qwen3.5-4B \
  --device-map auto \
  --torch-dtype auto
```

To reuse previous OEQ mappings and avoid re-running Qwen:

```bash
python scripts/evaluate_audio_predictions.py ... --skip-existing-mapping
```

To force regeneration:

```bash
python scripts/evaluate_audio_predictions.py ... --overwrite-mapping
```

## Outputs

Evaluation summaries are written to:

```text
PREDICTIONS_ROOT/evaluation_results/audio__SPLIT__MODEL_NAME.json
```

OEQ artifact mapping files are written to:

```text
PREDICTIONS_ROOT/evaluation_results/oeq_mappings/TASK/MODEL_NAME/SPLIT/qwen35-4b-response/*.json
```

## Audio artifact taxonomy

The audio evaluator maps explanations to five TRIDENT audio artifacts:

- `Clipping`: harsh, fuzzy, or crackling sound when audio is too loud.
- `Hiss`: high-frequency static noise, like a "shhhh" sound.
- `Buzz`: low-frequency tone, often caused by electrical interference.
- `Pops`: abrupt, short, sharp sounds interrupting audio.
- `Unnatural Prosody`: robotic, monotonous, or flat speech lacking natural intonation.

This code is intended for local validation and ablation studies before official submission, not as a replacement for the official private leaderboard evaluator.
