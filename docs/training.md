# Reproducible model experiments

## Data

Use `naver-clova-ix/cord-v2`, pinned to the revision in each model manifest.
CORD publishes Indonesian receipt images and semantic labels under CC BY 4.0:
https://github.com/clovaai/cord . Its released sample has 800 train, 100 validation,
and 100 test receipts. Some merchant/payment labels were removed from the release;
this project's financial extraction scope follows the available annotations.

Never merge validation/test examples into training. Keep official splits, record
receipt IDs and split hashes, and store private uploads separately from the benchmark.
Human corrections are review records, not automatically labelled training data.

## Lightweight learned model

The core classifier uses lexical, layout, numeric and neighboring-line features.
It runs without a GPU. Read its measured model card and evaluation report before
interpreting scores; micro averages can hide rare-class errors.

```powershell
python -m pip install -r requirements-train.txt
python -m training.train
```

Use the training CLI's `--help` for cache and report paths. The threshold is chosen
on validation data. A test report is for a frozen model version, not a tuning target.

## Transformer experiment

The optional multimodal experiment fine-tunes Microsoft's `layoutlmv3-base` with
receipt images, line text and 2-D bounding boxes. The backbone is pretrained; this
project does not train a foundation model from scratch. Its CC BY-NC-SA 4.0 model
license is separate from this repository's MIT source license:
https://huggingface.co/microsoft/layoutlmv3-base .

An RTX 4060 laptop with approximately 8 GB VRAM is the development hardware. Start
with batch size one, gradient accumulation, and mixed precision. LayoutLMv3's
installed implementation does not support gradient checkpointing; the training
code enables it only for models that support it. Record peak memory and actual
duration. No rented GPU is needed.

```powershell
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install transformers==4.57.6 accelerate
python -m training.transformer_train --help
```

The transformer CLI publishes its own report and keeps the fast model available for
the application. A higher-cost model is not automatically a better serving default.

## What evaluation measures

- **Supplied-text extraction:** labels predicted from CORD annotated text/boxes.
  It isolates extraction; it is not OCR accuracy.
- **Image-to-extraction subset:** real OCR followed by extraction on a declared
  held-out subset. Report coverage, field matches and latency separately.
- **Synthetic demo fixtures:** original English receipts for UI interaction.
  They are outside the benchmark distribution and do not establish generalization.

Publish baseline and learned per-class precision, recall and F1, numeric field
exact match, validation-selected review thresholds, acceptance precision/coverage,
and latency. A raw model confidence score is not a guaranteed probability of a
correct financial record. Missing/ambiguous values and arithmetic failures still
require human review.
