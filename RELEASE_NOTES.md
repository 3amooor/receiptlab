# ReceiptLab v1.0.0

Receipt OCR, trained classical extraction, optional fine-tuned LayoutLMv3 research weights, deterministic amount recovery and validation, and a React review workspace.

Includes measured CORD-v2 benchmark reports, real OCR failure analysis, model/source provenance, backend and browser tests, container verification, and a public sample workspace.

The optional archive contains the epoch-2 checkpoint selected on validation, processor/tokenizer files, evaluation manifest, attribution, and individual checksums. Verify the archive SHA256 against reports/transformer-artifact.json. Model weights inherit **CC-BY-NC-SA-4.0** and are for noncommercial research. Code is MIT.

Classifier metrics exclude OCR errors. Actual OCR performance is reported separately, and every assisted receipt requires human approval. Public samples are original synthetic fixtures with precomputed outputs; live uploads run locally.
