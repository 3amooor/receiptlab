# ReceiptLab line classifier model card

The frozen model classifies receipt lines into nine roles using word/character TF-IDF,
normalized geometry, adjacent line text, and multiclass logistic regression.
Amount parsing and review decisions remain deterministic application code.

## Data and experiment protocol

CORD-v2, official document splits: 800 train, 100 validation, 100 test.
Scalar probability temperature and review threshold were selected on validation only.
Test was evaluated after the artifact was frozen. Training uses annotated text and boxes,
not OCR-generated input. See models/manifest.json for the pinned dataset revision.

## Measured results

- Rules baseline test macro F1: 0.6239
- Learned model test macro F1: 0.9385
- Learned target-role macro F1: 0.9403
- Annotated-text total exact-value accuracy: 0.9578947368421052
- Actual OCR total exact-value accuracy (20 seeded test documents): 0.05555555555555555
- Validation-selected review threshold: 0.3

See evaluation.json for class support, precision/recall, absent/present field handling,
OCR matching assumptions, and latency. These results do not establish invoice, Arabic,
or operational financial accuracy.

## Intended use and limitations

Portfolio research and assistive receipt review. Trained on Indonesian receipts.
Currency is not inferred. Ambiguous amounts, missing totals, low confidence and failed
arithmetic require human review. Line probabilities do not establish document confidence.
Uploaded documents cannot override model artifacts or load pickled objects.
