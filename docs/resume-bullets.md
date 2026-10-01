# Resume entry

**ReceiptLab — Multimodal Document AI & Human Review Platform**  
Python · PyTorch · Hugging Face Transformers · scikit-learn · ONNX · FastAPI · React/TypeScript · SQLite · Docker · GitHub Actions

- Fine-tuned LayoutLMv3 on 800 CORD receipt documents; selected the checkpoint on 100 validation documents and achieved **0.957 macro F1 / 97.9% micro F1** on 100 held-out documents using supplied text, layout, and image inputs.
- Built and calibrated a TF-IDF/geometry classifier reaching **0.939 test macro F1**, compared with a **0.624** rules baseline; separately measured OCR-driven extraction and documented failure modes.
- Delivered a receipt review application with source-box evidence, deterministic financial validation, durable jobs, stale-edit protection, correction audits, and JSON/CSV exports; verified backend, browser, real inference, and CI/container flows.

Keep “supplied text” or “annotated input” attached to the transformer metric. It is line classification, not end-to-end OCR accuracy. Do not claim autonomous financial processing, production deployment, invoice accuracy, or invented business impact.

Before interviews, complete a local upload/review/export and reproduce training. Be ready to explain the tokenizer-state bug, checkpoint selection, calibration, OCR/domain shift, rare-class support, monetary ambiguity, and why anchors always require review.
