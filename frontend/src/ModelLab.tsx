import AdditionalBenchmarks from "./AdditionalBenchmarks";
import {
  ArrowDownToLine,
  FlaskConical,
  ShieldCheck,
  ScanLine,
} from "lucide-react";
import type { Report } from "./types";
import { download } from "./api";
const obj = (v: unknown): Record<string, unknown> =>
  v && typeof v === "object" && !Array.isArray(v)
    ? (v as Record<string, unknown>)
    : {};
const pct = (v: unknown) =>
  typeof v === "number" ? (v * 100).toFixed(1) + "%" : "—";
const f1 = (v: unknown) => (typeof v === "number" ? v.toFixed(3) : "—");
const name = (s: string) => s.replaceAll("_", " ").replaceAll(".", " / ");
export default function ModelLab({ report }: { report: Report | null }) {
  const b = obj(report?.baseline),
    l = obj(report?.learned),
    o = obj(report?.actual_ocr_pipeline),
    bp = obj(b.per_class),
    lp = obj(l.per_class);
  if (!report || !Object.keys(l).length)
    return (
      <div className="empty model-empty">
        <FlaskConical size={38} />
        <h2>
          The report goes here.
          <br />
          <em>After the experiment.</em>
        </h2>
        <p>
          No evaluation report has been loaded. Train and evaluate the local
          pipeline to publish actual held-out results.
        </p>
      </div>
    );
  const dataset =
    typeof report.dataset === "string"
      ? report.dataset
      : textDataset(report.dataset);
  return (
    <div className="model-lab">
      <div className="report-heading">
        <div>
          <div className="eyebrow">EVALUATION REPORT AVAILABLE</div>
          <h2>{String(report.model_version || "Receipt extraction model")}</h2>
          <p>
            {dataset} · {String(report.split || "Held-out test split")}
          </p>
        </div>
        <button
          className="button secondary"
          onClick={() =>
            download(
              new Blob([JSON.stringify(report, null, 2)], {
                type: "application/json",
              }),
              "receiptlab-evaluation.json",
            )
          }
        >
          <ArrowDownToLine size={15} />
          Download report
        </button>
      </div>
      <div className="method">
        <ShieldCheck size={23} />
        <div>
          <strong>Ground-truth text + bounding boxes</strong>
          <p>
            Baseline and learned results measure extraction on annotated input
            and exclude OCR errors. The complete image-to-data pipeline is
            evaluated separately below.
          </p>
        </div>
      </div>
      <div className="metrics">
        {[
          [
            "HELD-OUT DOCUMENTS",
            String(report.test_documents ?? "—"),
            String(report.split || "Recorded test split"),
          ],
          [
            "LEARNED MACRO F1",
            f1(l.macro_f1),
            "Rules baseline: " + f1(b.macro_f1),
          ],
          ["LINE ACCURACY", pct(l.accuracy), "Annotated input · excludes OCR"],
          [
            "EXTRACTION LATENCY",
            typeof obj(l.latency_ms).median === "number"
              ? Math.round(Number(obj(l.latency_ms).median)) + " ms"
              : "—",
            "Median · excludes OCR",
          ],
        ].map(([label, value, note]) => (
          <div key={label}>
            <span>{label}</span>
            <strong>{value}</strong>
            <small>{note}</small>
          </div>
        ))}
      </div>
      <div className="benchmark-layout">
        <section className="benchmark">
          <div className="section-label">
            PER-CLASS F1 SCORE <span>□ Rules baseline · ■ Learned</span>
          </div>
          <h3>Rules meet a learned model.</h3>
          <div className="chart">
            {Object.keys(lp)
              .filter(
                (k) => !["accuracy", "macro avg", "weighted avg"].includes(k),
              )
              .map((k) => (
                <div className="chart-row" key={k}>
                  <div>
                    <span>{name(k)}</span>
                    <small>n = {String(obj(lp[k]).support ?? "—")}</small>
                  </div>
                  <div>
                    {[
                      [obj(bp[k])["f1-score"], "baseline"],
                      [obj(lp[k])["f1-score"], "learned"],
                    ].map(([v, c]) => (
                      <div key={String(c)}>
                        <i
                          className={String(c)}
                          style={{
                            width:
                              (typeof v === "number"
                                ? Math.max(0, Math.min(1, v)) * 100
                                : 0) + "%",
                          }}
                        />
                        <b>{f1(v)}</b>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
          </div>
          <p className="muted">
            F1 balances precision and recall. Support is the number of annotated
            lines per class.
          </p>
        </section>
        <aside className="experiment">
          <FlaskConical size={30} />
          <h2>
            Reproducible.
            <br />
            <em>Inspectable.</em>
          </h2>
          <p>
            A deterministic rules baseline sets the reference. Validation data
            controls review thresholds.
          </p>
          <dl>
            <dt>Target macro F1</dt>
            <dd>{f1(l.target_macro_f1)}</dd>
            <dt>Review threshold</dt>
            <dd>{pct(report.confidence_threshold)}</dd>
            <dt>Flagged for review</dt>
            <dd>
              {String(l.documents_requiring_review ?? "—")} /{" "}
              {String(l.documents ?? report.test_documents ?? "—")}
            </dd>
            <dt>Test used for tuning</dt>
            <dd>
              {report.test_used_for_parameter_selection === false
                ? "No"
                : report.test_used_for_parameter_selection === true
                  ? "Yes"
                  : "Not reported"}
            </dd>
          </dl>
        </aside>
      </div>
      <section className="benchmark field-benchmark">
        <div className="section-label">
          MONETARY FIELD RECOVERY · ANNOTATED INPUT · EXCLUDES OCR
        </div>
        <h3>The numbers that matter.</h3>
        <FieldTable base={obj(b.fields)} learned={obj(l.fields)} />
        <p className="muted">
          Value accuracy evaluates present, unambiguous truth. Presence + value
          also evaluates correct absence. Ambiguous exclusions are recorded in
          the JSON report.
        </p>
      </section>
      <section className="ocr-benchmark">
        <div className="section-label">
          <ScanLine size={17} /> THE COMPLETE PIPELINE
        </div>
        <h2>And when OCR has to read it?</h2>
        <p className="muted">
          Image → OCR → extraction → validation. An independent sample; do not
          compare directly with the full test benchmark.
        </p>
        {Object.keys(o).length ? (
          <>
            <div className="metrics">
              {[
                ["SAMPLED DOCUMENTS", String(o.documents ?? "—")],
                [
                  "LINE DETECTION RECALL",
                  pct(o.annotated_line_detection_recall_iou_025),
                ],
                [
                  "EXACT TEXT RECALL",
                  pct(o.annotated_line_exact_text_recall_iou_025),
                ],
                [
                  "FLAGGED FOR REVIEW",
                  String(o.documents_requiring_review ?? "—"),
                ],
              ].map(([label, v]) => (
                <div key={label}>
                  <span>{label}</span>
                  <strong>{v}</strong>
                  <small>
                    {label.includes("RECALL")
                      ? "Annotated line matches at IoU ≥ 0.25"
                      : String(o.sampling || "Recorded sample")}
                  </small>
                </div>
              ))}
            </div>
            <FieldTable learned={obj(o.fields)} />
          </>
        ) : (
          <p className="muted">
            No complete OCR evaluation is included. Extraction results do not
            establish image-to-data performance.
          </p>
        )}
      </section>
      <AdditionalBenchmarks report={report} />
      <div className="limitations">
        <h3>Know the boundaries.</h3>
        <ul>
          {(Array.isArray(report.limitations)
            ? report.limitations
            : [
                "The benchmark covers its recorded dataset and split. New layouts require review.",
              ]
          ).map((v, i) => (
            <li key={i}>{String(v)}</li>
          ))}
        </ul>
      </div>
      <details className="raw">
        <summary>Inspect complete evaluation artifact</summary>
        <pre>{JSON.stringify(report, null, 2)}</pre>
      </details>
    </div>
  );
}
function textDataset(v: unknown) {
  const d = obj(v);
  return String(d.name || d.id || "CORD receipt benchmark");
}
function FieldTable({
  base,
  learned,
}: {
  base?: Record<string, unknown>;
  learned: Record<string, unknown>;
}) {
  return (
    <div className="table-scroll">
      <table>
        <thead>
          <tr>
            <th>Field</th>
            {base && <th>Baseline value accuracy</th>}
            <th>Value accuracy</th>
            <th>Present documents</th>
            <th>Presence + value accuracy</th>
          </tr>
        </thead>
        <tbody>
          {["subtotal", "tax", "discount", "total"].map((k) => {
            const v = obj(learned[k]);
            return (
              <tr key={k}>
                <td>{name(k)}</td>
                {base && <td>{pct(obj(base[k]).present_value_accuracy)}</td>}
                <td>
                  <strong>{pct(v.present_value_accuracy)}</strong>
                </td>
                <td>{String(v.present_documents ?? "—")}</td>
                <td>{pct(v.presence_and_value_accuracy)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
