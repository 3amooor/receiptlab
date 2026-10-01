# Failure analysis

The frozen classical model reached 0.939 macro F1 on official test annotations but only recovered 1/18 present unambiguous totals from the 20-document OCR sample.

CORD supplied lines and OCR detections have different coverage and ordering. Extra headers alter neighboring text. RapidOCR separates TOTAL from its amount; small vertical offsets change line order. Numeric lines can become item amounts, while the printed key becomes an item description. Dates and times also resemble amounts when parsing accepts a trailing number.

The operational assistant recognizes bounded explicit labels. It reads a complete inline monetary string or an unambiguous numeric box on the same row to the right. It withholds differing candidates or disagreement with the model, excludes unsupported descriptions, retains raw predictions, assigns no learned confidence to anchors, and always requires review.

Rules were fixed before separate validation and additional test evaluation. The original v1 report remains published. Assistance recovered 10/20 validation totals and 10/18 present totals in the additional test sample.

LayoutLMv3 encountered two integration failures before the completed experiment: an unsupported checkpointing option, and retained fast-tokenizer padding state inflating chunk costs after a padded encode. The final implementation explicitly specifies padding/truncation, checks coverage, and asserts 800 training windows per epoch. Its saved checkpoint came from epoch 2, selected on validation; epoch 3 was weaker. Test evaluation uses that saved checkpoint.

Future research could use training-only OCR augmentation, word/box alignment, more varied layouts, and document-level calibration. These are not claimed as completed.
