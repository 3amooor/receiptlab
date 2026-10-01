# ReceiptLab engineering guidance

This is a separate portfolio project, unrelated to the real-estate buyer system.
The source code is MIT licensed. CORD dataset attribution and model-specific licenses
remain separate; do not redistribute real uploads or benchmark receipt images.

Keep raw OCR, learned predictions, validated fields, human corrections, and benchmark
labels distinct. Never report supplied benchmark text as end-to-end OCR accuracy.
Select hyperparameters and review thresholds on training/validation only. Freeze test
data and measure it once for a model version; publish unsuccessful results too.

The local API is single-user and loopback-bound. Do not expose it publicly without
authentication, authorization, rate limits, retention controls, and deployment review.
Uploaded content is untrusted. Bound file size, pages, pixels, numeric values, and
storage paths. Model artifacts must come from trusted local training or verified releases.

Use document revision checks for review writes. Preserve extraction before corrections.
Do not claim a review correction is a model prediction or automatically retrain from it.
All displayed benchmark numbers must be read from measured report artifacts.

Run pytest, dependency consistency, Ruff, frontend type/build checks, and browser
acceptance tests before publishing a changed release. Keep generated runtime data,
credentials, environments, dependencies, and large datasets out of Git.
