import type { Report } from "./types";
const obj = (v: unknown): Record<string, unknown> =>
  v && typeof v === "object" && !Array.isArray(v)
    ? (v as Record<string, unknown>)
    : {};
const pct = (v: unknown) =>
  typeof v === "number" ? (v * 100).toFixed(1) + "%" : "—";
const f1 = (v: unknown) => (typeof v === "number" ? v.toFixed(3) : "—");
export default function AdditionalBenchmarks({ report }: { report: Report }) {
  const transformer = obj(report.transformer),
    test = obj(transformer.test),
    assisted = obj(report.assisted);
  return (
    <>
      {Object.keys(test).length > 0 && (
        <section className="benchmark field-benchmark">
          <div className="section-label">
            MULTIMODAL TRANSFORMER · ANNOTATED INPUT · EXCLUDES OCR
          </div>
          <h3>A document transformer, fine-tuned.</h3>
          <p className="muted">
            Receipt pixels, supplied text, and bounding boxes feed a separately
            trained model. These scores classify lines using annotated text and
            exclude OCR errors.
          </p>
          <div className="metrics">
            <div>
              <span>TEST MACRO F1</span>
              <strong>{f1(test.macro_f1)}</strong>
              <small>
                {String(test.documents)} documents · {String(test.lines)} lines
              </small>
            </div>
            <div>
              <span>TEST MICRO F1</span>
              <strong>{f1(test.micro_f1)}</strong>
              <small>Annotated input</small>
            </div>
            <div>
              <span>TRAINING DOCUMENTS</span>
              <strong>{String(transformer.training_documents)}</strong>
              <small>
                {String(transformer.epochs)} epochs · {String(transformer.gpu)}
              </small>
            </div>
            <div>
              <span>INFERENCE LATENCY</span>
              <strong>
                {typeof test.latency_ms_p50 === "number"
                  ? Math.round(test.latency_ms_p50) + " ms"
                  : "—"}
              </strong>
              <small>Median · excludes OCR</small>
            </div>
          </div>
          <p className="muted">
            Validation-selected acceptance: {pct(test.accepted_line_precision)}{" "}
            precision at {pct(test.accepted_line_coverage)} line coverage. This
            does not establish financial extraction accuracy. Base model
            license: {String(transformer.license)}.
          </p>
        </section>
      )}
      {Object.keys(assisted).length > 0 && (
        <section className="benchmark field-benchmark">
          <div className="section-label">
            OPERATIONAL ASSISTANCE · IMAGE → OCR → RULES + MODEL
          </div>
          <h3>Text anchors assist the review pipeline.</h3>
          <p className="muted">
            Explicit printed labels assist extraction. Every assisted result
            requires review; recovered values carry no learned confidence.
          </p>
          <div className="metrics">
            <div>
              <span>EVALUATED DOCUMENTS</span>
              <strong>{String(assisted.documents)}</strong>
              <small>Additional operational evaluation</small>
            </div>
            <div>
              <span>FLAGGED FOR REVIEW</span>
              <strong>{String(assisted.documents_requiring_review)}</strong>
              <small>Human approval required</small>
            </div>
            <div>
              <span>ASSISTED TOTAL ACCURACY</span>
              <strong>
                {pct(
                  obj(obj(assisted.assisted_fields).total)
                    .present_value_accuracy,
                )}
              </strong>
              <small>Present, unambiguous totals</small>
            </div>
            <div>
              <span>RAW TOTAL ACCURACY</span>
              <strong>
                {pct(
                  obj(obj(assisted.raw_learned_fields).total)
                    .present_value_accuracy,
                )}
              </strong>
              <small>Same sample · without assistance</small>
            </div>
          </div>
          <div className="warnings">
            <strong>Development note</strong>
            <p>{String(assisted.development_note)}</p>
          </div>
        </section>
      )}
    </>
  );
}
