# Data and model provenance

## Benchmark

CORD: Park et al., *CORD: A Consolidated Receipt Dataset for Post-OCR Parsing*,
Document Intelligence Workshop at NeurIPS, 2019.
https://github.com/clovaai/cord

Released sample: `naver-clova-ix/cord-v2`; CC BY 4.0. Attribution applies to
benchmark-derived artifacts. Benchmark images and local caches are not committed.
Each experiment manifest records the exact revision and split hashes.

## OCR

RapidOCR-ONNXRuntime performs local image inference using packaged OCR weights.
Upstream source and model provenance: https://github.com/RapidAI/RapidOCR .
Installed versions are recorded in the environment report. OCR outputs may contain
recognition and grouping errors; the downstream review policy cannot repair every error.

## Transformer

LayoutLMv3: Huang et al., *LayoutLMv3: Pre-training for Document AI with Unified
Text and Image Masking*, ACM Multimedia 2022.
https://huggingface.co/microsoft/layoutlmv3-base

The base model carries CC BY-NC-SA 4.0 terms. Fine-tuned weights retain their
upstream licensing restrictions. Repository MIT licensing applies to original code,
not to every dependency, dataset or upstream model.

## Public samples

`scripts/make_samples.py` creates artificial receipts from invented merchants and
amounts. These images contain no real customer data and are original project fixtures.
Precomputed sample predictions are labelled as sample data in the public showcase.

## Local uploads

Local uploaded files, OCR text, SQLite records, and correction histories are ignored
by Git. No uploaded content is sent to a managed inference provider. Do not publish
your data directory or use real receipts with personal/payment details as screenshots.
