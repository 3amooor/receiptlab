import { useEffect, useRef, useState } from "react";
import {
  Activity,
  ArrowDownToLine,
  ArrowLeft,
  ArrowRight,
  Check,
  CheckCheck,
  ChevronDown,
  Code2,
  FileText,
  FlaskConical,
  Layers3,
  LoaderCircle,
  Menu,
  Plus,
  RefreshCw,
  Search,
  ShieldCheck,
  Trash2,
  Upload,
  X,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import * as api from "./api";
import type { Audit, Doc, EditItem, Key, Report } from "./types";
import ModelLab from "./ModelLab";
const keys: Key[] = ["subtotal", "tax", "discount", "total"];
const labels: Record<Key, string> = {
  subtotal: "Subtotal",
  tax: "Tax",
  discount: "Discount",
  total: "Grand total",
};
const statuses: Record<string, string> = {
  needs_review: "Needs review",
  approved: "Approved",
  ready: "Ready to review",
  queued: "Queued",
  processing: "Processing",
  failed: "Failed",
};
const titles: Record<string, string> = {
  "sample-a": "Paper & Bean",
  "sample-b": "Green Market",
  "sample-c": "Studio Supply",
};
const text = (v: unknown) => (v == null ? "" : String(v));
const empty = () => ({ subtotal: "", tax: "", discount: "", total: "" });
const date = (v?: string) =>
  v
    ? new Date(v).toLocaleString(undefined, {
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      })
    : "Recorded";
const error = (e: unknown) =>
  e instanceof Error ? e.message : "Something went wrong.";
function Status({ value }: { value: string }) {
  return (
    <span className={"status " + value}>
      <i />
      {statuses[value] || value}
    </span>
  );
}
function Confidence({ value, origin }: { value?: number; origin?: string }) {
  return (
    <span
      className={
        "confidence " + (value !== undefined && value < 0.75 ? "low" : "")
      }
      title={
        origin === "text_anchor"
          ? "Printed-label source evidence. No learned confidence is claimed."
          : "Original model confidence; edits do not change this score"
      }
    >
      {origin === "text_anchor"
        ? "Anchor"
        : value === undefined
          ? "—"
          : Math.round(value * 100) + "%"}
    </span>
  );
}
export default function App() {
  const [view, setView] = useState("workspace"),
    [docs, setDocs] = useState<Doc[]>([]),
    [doc, setDoc] = useState<Doc | null>(null),
    [fields, setFields] = useState<Record<Key, string>>(empty),
    [items, setItems] = useState<EditItem[]>([]),
    [events, setEvents] = useState<Audit[]>([]),
    [report, setReport] = useState<Report | null>(null),
    [loading, setLoading] = useState(true),
    [busy, setBusy] = useState(false),
    [online, setOnline] = useState(false),
    [problem, setProblem] = useState(""),
    [dirty, setDirty] = useState(false),
    [toast, setToast] = useState(""),
    [active, setActive] = useState<Key | null>("total"),
    [tab, setTab] = useState("review"),
    [zoom, setZoom] = useState(100),
    [boxes, setBoxes] = useState(true),
    [search, setSearch] = useState(""),
    [filter, setFilter] = useState("all"),
    [menu, setMenu] = useState(false),
    [drag, setDrag] = useState(false),
    [imageFailed, setImageFailed] = useState(false);
  const input = useRef<HTMLInputElement>(null),
    canvas = useRef<HTMLDivElement>(null),
    selection = useRef(""),
    isDirty = useRef(false);
  isDirty.current = dirty;
  function hydrate(d: Doc) {
    setDoc(d);
    setFields(
      Object.fromEntries(
        keys.map((k) => [k, text(d.extraction?.fields[k]?.value)]),
      ) as Record<Key, string>,
    );
    setItems(
      (d.extraction?.items || []).map((i) => ({
        description: text(i.description),
        quantity: text(i.quantity),
        unit_price: text(i.unit_price),
        amount: text(i.amount),
      })),
    );
    setDirty(false);
    setImageFailed(false);
  }
  async function pick(d: Doc) {
    if (isDirty.current && d.id !== doc?.id) {
      setToast("Save or discard edits before switching documents.");
      return;
    }
    selection.current = d.id;
    try {
      const [full, a] = await Promise.all([
        api.document(d.id),
        api.audit(d.id),
      ]);
      if (selection.current !== d.id) return;
      hydrate(full);
      setEvents(a);
      setZoom(100);
    } catch (e) {
      setToast(error(e));
    }
  }
  async function refresh() {
    if (isDirty.current) {
      setToast("Save or discard edits before refreshing.");
      return;
    }
    setLoading(true);
    try {
      await api.health();
      setOnline(true);
      setProblem("");
      const data = await api.documents();
      setDocs(data);
      if (data.length) await pick(data[0]);
      try {
        setReport(await api.metrics());
      } catch {
        setReport(null);
      }
    } catch (e) {
      setOnline(false);
      setProblem(error(e));
    } finally {
      setLoading(false);
    }
  }
  useEffect(() => {
    void refresh();
  }, []);
  useEffect(() => {
    if (!toast) return;
    const t = setTimeout(() => setToast(""), 5500);
    return () => clearTimeout(t);
  }, [toast]);
  useEffect(() => {
    if (api.DEMO || !doc || !["queued", "processing"].includes(doc.status))
      return;
    const id = doc.id,
      t = setInterval(async () => {
        try {
          const d = await api.document(id);
          setDocs(await api.documents());
          if (selection.current === id && !isDirty.current) {
            hydrate(d);
            setEvents(await api.audit(id));
          }
        } catch (e) {
          setToast(error(e));
        }
      }, 1800);
    return () => clearInterval(t);
  }, [doc?.id, doc?.status]);
  const box = active ? doc?.extraction?.fields[active]?.bbox : null;
  useEffect(() => {
    const t = setTimeout(() => {
      const c = canvas.current,
        p = c?.querySelector<HTMLElement>(".receipt-paper");
      if (!c || !p || !box) return;
      const y = p.offsetTop + ((box[1] + box[3]) / 2) * p.clientHeight;
      if (y < c.scrollTop + 60 || y > c.scrollTop + c.clientHeight - 60)
        c.scrollTo({
          top: Math.max(0, y - c.clientHeight / 2),
          behavior: "smooth",
        });
    }, 120);
    return () => clearTimeout(t);
  }, [active, box, zoom, doc?.id]);
  function navigate(v: string) {
    setView(v);
    setMenu(false);
  }
  async function upload(file: File) {
    if (api.DEMO) {
      setToast(
        "Uploads run in the local app. Review and export the saved examples here.",
      );
      return;
    }
    if (dirty) {
      setToast("Save or discard your edits first.");
      return;
    }
    if (
      !["image/png", "image/jpeg", "application/pdf"].includes(file.type) ||
      file.size > 10 * 1024 * 1024
    ) {
      setToast("Choose a PNG, JPG, or PDF smaller than 10 MB.");
      return;
    }
    setBusy(true);
    try {
      const d = await api.upload(file);
      setDocs(await api.documents());
      await pick(d);
      setView("workspace");
      setToast("Receipt added. Extraction updates automatically.");
    } catch (e) {
      setToast(error(e));
    } finally {
      setBusy(false);
    }
  }
  async function sample() {
    if (dirty) {
      setToast("Save or discard your edits first.");
      return;
    }
    setBusy(true);
    try {
      if (api.DEMO && docs.length)
        await pick(
          docs[(docs.findIndex((d) => d.id === doc?.id) + 1) % docs.length],
        );
      else {
        const d = await api.sample(
          ["sample-a", "sample-b", "sample-c"][docs.length % 3],
        );
        setDocs(await api.documents());
        await pick(d);
      }
      setView("workspace");
    } catch (e) {
      setToast(error(e));
    } finally {
      setBusy(false);
    }
  }
  async function save(approve: boolean) {
    if (!doc) return;
    if (keys.some((k) => fields[k] && !/^\d+(?:\.\d{1,2})?$/.test(fields[k]))) {
      setToast(
        "Amounts must be non-negative numbers with up to two decimal places.",
      );
      return;
    }
    if (
      items.some(
        (i) =>
          !i.description.trim() ||
          [i.quantity, i.unit_price, i.amount].some(
            (v) => v && !/^\d+(?:\.\d+)?$/.test(v),
          ),
      )
    ) {
      setToast(
        "Every item needs a description and valid non-negative numbers.",
      );
      return;
    }
    setBusy(true);
    try {
      const d = await api.review(doc, fields, items, approve);
      hydrate(d);
      setDocs(await api.documents());
      setEvents(await api.audit(d.id));
      setToast(
        approve
          ? "Receipt approved. Review and audit record saved."
          : "Corrections saved.",
      );
    } catch (e) {
      setToast(error(e));
    } finally {
      setBusy(false);
    }
  }
  async function exportData(format: "json" | "csv") {
    if (!doc) return;
    if (dirty) {
      setToast("Save edits before exporting.");
      return;
    }
    try {
      await api.exportFile(doc, format);
      setToast(format.toUpperCase() + " downloaded.");
    } catch (e) {
      setToast(error(e));
    }
  }
  const queue = docs.filter(
    (d) =>
      d.filename.toLowerCase().includes(search.toLowerCase()) &&
      (filter === "all" || d.status === filter),
  );
  const count = docs.filter((d) =>
    ["needs_review", "ready"].includes(d.status),
  ).length;
  const approved = docs.filter((d) => d.status === "approved").length;
  const balanced =
    !!fields.total &&
    !!fields.subtotal &&
    Math.abs(
      Number(fields.subtotal) +
        Number(fields.tax || 0) -
        Number(fields.discount || 0) -
        Number(fields.total),
    ) < 0.021;
  const nav = [
    { id: "workspace", label: "Workspace", Icon: Layers3 },
    { id: "model", label: "Model lab", Icon: FlaskConical },
    { id: "activity", label: "Activity", Icon: Activity },
  ];
  return (
    <div
      className="shell"
      onDragOver={(e) => {
        e.preventDefault();
        if (!api.DEMO) setDrag(true);
      }}
      onDragLeave={(e) => {
        if (!e.currentTarget.contains(e.relatedTarget as Node)) setDrag(false);
      }}
      onDrop={(e) => {
        e.preventDefault();
        setDrag(false);
        if (e.dataTransfer.files[0]) void upload(e.dataTransfer.files[0]);
      }}
    >
      <aside className={"sidebar " + (menu ? "open" : "")}>
        <a
          className="brand"
          href="#"
          onClick={(e) => {
            e.preventDefault();
            navigate("workspace");
          }}
          aria-label="ReceiptLab workspace"
        >
          <svg viewBox="0 0 36 40" fill="none" aria-hidden="true">
            <path
              d="M8 3h20v33l-5-3-5 3-5-3-5 3V3Z"
              stroke="currentColor"
              strokeWidth="2.3"
            />
            <path
              d="M13 12h10M13 18h10M13 24h6"
              stroke="currentColor"
              strokeWidth="2.3"
            />
            <circle cx="28" cy="28" r="6" fill="#d2e784" />
            <path d="m25 28 2 2 4-4" stroke="#192b25" strokeWidth="1.7" />
          </svg>
          <span>
            Receipt<b>Lab</b>
            <small>DOCUMENT INTELLIGENCE</small>
          </span>
        </a>
        <div className="nav-label">YOUR WORKBENCH</div>
        <nav aria-label="Main navigation">
          {nav.map(({ id, label, Icon }) => (
            <button
              key={id}
              aria-label={label}
              className={view === id ? "active" : ""}
              onClick={() => navigate(id)}
            >
              <Icon size={19} />
              <span>{label}</span>
              {id === "workspace" && <b>{docs.length}</b>}
            </button>
          ))}
        </nav>
        <div className="sidebar-note">
          <small>01 / THE APPROACH</small>
          <p>
            Good AI shows
            <br />
            <em>its working.</em>
          </p>
          <span>
            Every prediction has evidence.
            <br />
            Every correction leaves a trail.
          </span>
          <div className="note-grid">
            <i />
            <i />
            <i />
            <i />
            <i />
            <i />
            <i />
            <i />
            <i />
          </div>
        </div>
        <div className="connection">
          <i className={online ? "on" : ""} />
          <div>
            <strong>
              {api.DEMO
                ? "Sample workspace"
                : online
                  ? "Local engine connected"
                  : "Engine offline"}
            </strong>
            <small>
              {api.DEMO
                ? "Saved examples · browser review"
                : "Private files · local inference"}
            </small>
          </div>
          <ShieldCheck size={17} />
        </div>
      </aside>
      {menu && (
        <button
          className="scrim"
          onClick={() => setMenu(false)}
          aria-label="Close navigation"
        />
      )}
      <main>
        <header className="topbar">
          <div>
            <button
              className="icon mobile-menu"
              onClick={() => setMenu(!menu)}
              aria-label="Open navigation"
            >
              <Menu size={20} />
            </button>
            <span>ReceiptLab</span>
            <i>/</i>
            <strong>
              {view === "workspace"
                ? "Document workspace"
                : view === "model"
                  ? "Model lab"
                  : "Activity log"}
            </strong>
          </div>
          <div>
            <small>{api.DEMO ? "INTERACTIVE SAMPLE" : "LOCAL WORKSPACE"}</small>
            <b className="avatar">OT</b>
          </div>
        </header>
        <div className="content">
          {api.DEMO && (
            <div className="banner">
              <FlaskConical size={16} />
              <span>
                <strong>Sample workspace.</strong> Original synthetic receipts
                with saved outputs. Edits stay in your browser. Uploads run in
                the local app.
              </span>
            </div>
          )}
          {!online && problem && (
            <div className="banner offline" role="alert">
              <span>{problem}</span>
              <button onClick={() => void refresh()}>
                <RefreshCw size={14} />
                Reconnect
              </button>
            </div>
          )}
          <div className="page-heading">
            <div>
              <div className="eyebrow">
                {view === "workspace"
                  ? "PAPER. PARSED. VERIFIED."
                  : view === "model"
                    ? "MEASURE BEFORE YOU TRUST."
                    : "A RECORD OF EVERY DECISION."}
              </div>
              <h1>
                {view === "workspace" ? (
                  <>
                    Document <em>workspace.</em>
                  </>
                ) : view === "model" ? (
                  <>
                    Inside the <em>model.</em>
                  </>
                ) : (
                  <>
                    Nothing <em>unrecorded.</em>
                  </>
                )}
              </h1>
              <p>
                {view === "workspace"
                  ? "Turn everyday receipts into data you can actually trust."
                  : view === "model"
                    ? "Real experiments, honest benchmarks, and visible limitations."
                    : "Trace extraction, corrections, and approvals for the selected receipt."}
              </p>
            </div>
            <div className="heading-actions">
              {view === "workspace" ? (
                <>
                  <button
                    className="button secondary"
                    disabled={busy || loading}
                    onClick={() => void sample()}
                  >
                    <FlaskConical size={16} />
                    {api.DEMO ? "Next sample" : "Try a sample"}
                  </button>
                  <button
                    className="button primary"
                    disabled={busy || api.DEMO || !online}
                    onClick={() => input.current?.click()}
                    title={
                      api.DEMO
                        ? "Run the local app to upload your receipts"
                        : undefined
                    }
                  >
                    <Upload size={16} />
                    Upload receipt
                  </button>
                </>
              ) : (
                <button
                  className="button secondary"
                  onClick={() => navigate("workspace")}
                >
                  <ArrowLeft size={16} />
                  Back to workspace
                </button>
              )}
            </div>
          </div>
          <input
            ref={input}
            type="file"
            accept="image/png,image/jpeg,application/pdf"
            className="hidden"
            aria-label="Upload receipt"
            onChange={(e) => {
              if (e.target.files?.[0]) void upload(e.target.files[0]);
              e.target.value = "";
            }}
          />
          {view === "workspace" && (
            <>
              <div className="summary">
                {[
                  [docs.length, "Receipts in workspace"],
                  [count, "Awaiting human review"],
                  [approved, "Approved & exportable"],
                ].map(([n, l], i) => (
                  <div key={String(l)}>
                    <b className={i === 1 ? "orange" : ""}>
                      {String(n).padStart(2, "0")}
                    </b>
                    <span>{l}</span>
                  </div>
                ))}
                <small>
                  Extract <ArrowRight size={12} /> Review{" "}
                  <ArrowRight size={12} /> Export
                </small>
              </div>
              {loading ? (
                <div className="empty loading">
                  <LoaderCircle className="spin" size={28} />
                  Opening your workspace…
                </div>
              ) : (
                <div className="workbench">
                  <section className="queue">
                    <div className="panel-heading">
                      <h2>
                        Documents <b>{docs.length}</b>
                      </h2>
                      <button
                        className="icon"
                        onClick={() => void refresh()}
                        aria-label="Refresh documents"
                      >
                        <RefreshCw size={14} />
                      </button>
                    </div>
                    <label className="search">
                      <Search size={14} />
                      <input
                        value={search}
                        onChange={(e) => setSearch(e.target.value)}
                        placeholder="Find a receipt"
                        aria-label="Find a receipt"
                      />
                    </label>
                    <select
                      value={filter}
                      onChange={(e) => setFilter(e.target.value)}
                      aria-label="Filter documents"
                    >
                      <option value="all">All documents</option>
                      {Object.entries(statuses).map(([k, v]) => (
                        <option key={k} value={k}>
                          {v}
                        </option>
                      ))}
                    </select>
                    <div className="queue-list">
                      {queue.map((d, i) => (
                        <button
                          key={d.id}
                          className={
                            "doc-card " + (d.id === doc?.id ? "selected" : "")
                          }
                          onClick={() => {
                            if (d.id !== doc?.id) void pick(d);
                          }}
                        >
                          <div>
                            <span className="thumbnail">
                              <FileText size={23} />
                            </span>
                            <small>{String(i + 1).padStart(2, "0")}</small>
                          </div>
                          <strong>
                            {titles[d.filename.replace(/\.[^.]+$/, "")] ||
                              d.filename
                                .replace(/\.[^.]+$/, "")
                                .replaceAll("_", " ")}
                          </strong>
                          <span className="doc-date">
                            {d.filename.split(".").pop()?.toUpperCase()} ·{" "}
                            {date(d.created_at)}
                          </span>
                          <Status value={d.status} />
                        </button>
                      ))}
                      {!queue.length && (
                        <p className="queue-empty">
                          {docs.length
                            ? "No matching receipts."
                            : "Your first receipt belongs here."}
                        </p>
                      )}
                    </div>
                    <button
                      className="queue-add"
                      disabled={busy || !online}
                      onClick={() =>
                        api.DEMO ? void sample() : input.current?.click()
                      }
                    >
                      <Plus size={15} />
                      {api.DEMO ? "Explore another sample" : "Add a receipt"}
                      <small>
                        {api.DEMO
                          ? "Saved original examples"
                          : "PNG, JPG, PDF · up to 10 MB"}
                      </small>
                    </button>
                  </section>
                  {!doc ? (
                    <div className="empty">
                      <FileText size={45} />
                      <h2>
                        A little paper.
                        <br />
                        <em>A lot of possibility.</em>
                      </h2>
                      <p>
                        Upload a receipt or explore an original example.
                        <br />
                        Extracted data appears beside the source for review.
                      </p>
                      <button
                        className="button primary"
                        disabled={busy || !online}
                        onClick={() => void sample()}
                      >
                        <FlaskConical size={16} />
                        Try a sample receipt
                      </button>
                    </div>
                  ) : (
                    <>
                      <section className="viewer">
                        <div className="viewer-tools">
                          <span>
                            <FileText size={15} />
                            Source document
                          </span>
                          <div>
                            <button
                              className={
                                "icon " + (boxes ? "selected-tool" : "")
                              }
                              onClick={() => setBoxes(!boxes)}
                              aria-label="Toggle evidence highlights"
                            >
                              <Layers3 size={15} />
                            </button>
                            <button
                              className="icon"
                              disabled={zoom <= 50}
                              onClick={() => setZoom((z) => z - 25)}
                              aria-label="Zoom out"
                            >
                              <ZoomOut size={15} />
                            </button>
                            <small>{zoom}%</small>
                            <button
                              className="icon"
                              disabled={zoom >= 200}
                              onClick={() => setZoom((z) => z + 25)}
                              aria-label="Zoom in"
                            >
                              <ZoomIn size={15} />
                            </button>
                          </div>
                        </div>
                        <div className="receipt-canvas" ref={canvas}>
                          <div className="canvas-label">
                            ORIGINAL / {doc.width || "—"} × {doc.height || "—"}
                          </div>
                          {api.image(doc) && !imageFailed ? (
                            <div
                              className="receipt-paper"
                              style={{
                                width: (84 * zoom) / 100 + "%",
                                minWidth:
                                  zoom > 100
                                    ? (84 * zoom) / 100 + "%"
                                    : undefined,
                              }}
                            >
                              <img
                                src={api.image(doc)}
                                alt={"Original receipt: " + doc.filename}
                                onError={() => setImageFailed(true)}
                              />
                              {boxes && box && (
                                <div
                                  className="evidence-box"
                                  style={{
                                    left: box[0] * 100 + "%",
                                    top: box[1] * 100 + "%",
                                    width: (box[2] - box[0]) * 100 + "%",
                                    height: (box[3] - box[1]) * 100 + "%",
                                  }}
                                >
                                  <span>
                                    {active ? labels[active] : "Evidence"}
                                  </span>
                                </div>
                              )}
                            </div>
                          ) : (
                            <div className="image-empty">
                              <FileText size={35} />
                              <p>
                                {imageFailed
                                  ? "Could not load the image."
                                  : "Preparing the preview…"}
                              </p>
                              <button onClick={() => setImageFailed(false)}>
                                Retry preview
                              </button>
                            </div>
                          )}
                          <div className="canvas-caption">
                            <span>
                              <ShieldCheck size={12} />
                              Source stays beside every field
                            </span>
                            <span>Page 1 / {doc.page_count || 1}</span>
                          </div>
                        </div>
                        <div className="viewer-foot">
                          <span>
                            <i />
                            Selected field evidence
                          </span>
                          <span>{doc.source || "Uploaded receipt"}</span>
                        </div>
                      </section>
                      <section className="review">
                        <div className="review-tabs">
                          <div>
                            {["review", "json"].map((t) => (
                              <button
                                key={t}
                                className={tab === t ? "active" : ""}
                                onClick={() => setTab(t)}
                              >
                                {t === "review" ? (
                                  "Extraction"
                                ) : (
                                  <>
                                    <Code2 size={13} />
                                    JSON
                                  </>
                                )}
                              </button>
                            ))}
                          </div>
                          <details className="export">
                            <summary aria-label="Export document">
                              <ArrowDownToLine size={17} />
                              <ChevronDown size={11} />
                            </summary>
                            <div>
                              <button
                                disabled={!doc.extraction}
                                onClick={() => void exportData("json")}
                              >
                                Download JSON
                              </button>
                              <button
                                disabled={!doc.extraction}
                                onClick={() => void exportData("csv")}
                              >
                                Download CSV
                              </button>
                            </div>
                          </details>
                        </div>
                        {!doc.extraction ? (
                          <div className="processing">
                            <LoaderCircle
                              className={doc.status === "failed" ? "" : "spin"}
                              size={30}
                            />
                            <h3>
                              {doc.status === "failed"
                                ? "Extraction needs attention."
                                : "Reading the receipt."}
                            </h3>
                            <p>
                              {doc.error ||
                                "OCR and the extraction model are working. This view updates automatically."}
                            </p>
                          </div>
                        ) : tab === "json" ? (
                          <div className="json">
                            <span>
                              Structured output · revision {doc.revision}
                            </span>
                            <pre>{JSON.stringify(doc.extraction, null, 2)}</pre>
                            <button
                              className="button secondary"
                              onClick={() => void exportData("json")}
                            >
                              <ArrowDownToLine size={15} />
                              Download JSON
                            </button>
                          </div>
                        ) : (
                          <>
                            <div className="review-scroll">
                              <div className="review-state">
                                <Status value={doc.status} />
                                <small>
                                  REV {String(doc.revision).padStart(2, "0")}
                                </small>
                              </div>
                              <div className="review-intro">
                                <h2>Check the details.</h2>
                                <p>
                                  Click a field to inspect its source.
                                  <br />
                                  Amounts use the currency printed on the
                                  receipt.
                                </p>
                              </div>
                              <div className="section-label">
                                <span>MONETARY FIELDS</span>
                                <span>MODEL / SOURCE</span>
                              </div>
                              <div className="fields">
                                {keys.map((k) => (
                                  <div
                                    key={k}
                                    className={
                                      "field " +
                                      (active === k ? "active " : "") +
                                      (k === "total" ? "total" : "")
                                    }
                                  >
                                    <div>
                                      <label htmlFor={"field-" + k}>
                                        {labels[k]}
                                      </label>
                                      <Confidence
                                        value={
                                          doc.extraction?.fields[k]?.confidence
                                        }
                                        origin={
                                          doc.extraction?.fields[k]?.origin
                                        }
                                      />
                                    </div>
                                    <div className="money">
                                      <span aria-hidden="true">¤</span>
                                      <input
                                        id={"field-" + k}
                                        inputMode="decimal"
                                        value={fields[k]}
                                        placeholder="Not extracted"
                                        onFocus={() => setActive(k)}
                                        onChange={(e) => {
                                          setFields((f) => ({
                                            ...f,
                                            [k]: e.target.value,
                                          }));
                                          setDirty(true);
                                        }}
                                      />
                                      <button
                                        className="icon"
                                        onClick={() => setActive(k)}
                                        aria-label={
                                          "Show " + labels[k] + " evidence"
                                        }
                                      >
                                        <Search size={13} />
                                      </button>
                                    </div>
                                  </div>
                                ))}
                              </div>
                              <div
                                className={
                                  "balance " + (balanced ? "balanced" : "")
                                }
                              >
                                <ShieldCheck size={14} />
                                {balanced
                                  ? "Amounts balance: subtotal + tax − discount = total."
                                  : "Check the amounts against the receipt before approval."}
                              </div>
                              <div className="section-label">
                                <span>LINE ITEMS · {items.length}</span>
                                <button
                                  onClick={() => {
                                    setItems((a) => [
                                      ...a,
                                      {
                                        description: "",
                                        quantity: "1",
                                        unit_price: "",
                                        amount: "",
                                      },
                                    ]);
                                    setDirty(true);
                                  }}
                                >
                                  <Plus size={13} />
                                  Add
                                </button>
                              </div>
                              <div className="items">
                                {items.map((item, i) => (
                                  <div className="item" key={i}>
                                    <div>
                                      <small>
                                        {String(i + 1).padStart(2, "0")}
                                      </small>
                                      <input
                                        value={item.description}
                                        placeholder="Item description"
                                        aria-label={
                                          "Item " + (i + 1) + " description"
                                        }
                                        onFocus={() => setActive(null)}
                                        onChange={(e) => {
                                          setItems((a) =>
                                            a.map((v, j) =>
                                              j === i
                                                ? {
                                                    ...v,
                                                    description: e.target.value,
                                                  }
                                                : v,
                                            ),
                                          );
                                          setDirty(true);
                                        }}
                                      />
                                      <button
                                        className="icon"
                                        onClick={() => {
                                          setItems((a) =>
                                            a.filter((_, j) => j !== i),
                                          );
                                          setDirty(true);
                                        }}
                                        aria-label={"Remove item " + (i + 1)}
                                      >
                                        <Trash2 size={13} />
                                      </button>
                                    </div>
                                    <div className="item-values">
                                      {(
                                        [
                                          "quantity",
                                          "unit_price",
                                          "amount",
                                        ] as const
                                      ).map((k) => (
                                        <label key={k}>
                                          {k === "quantity"
                                            ? "QTY"
                                            : k === "unit_price"
                                              ? "UNIT PRICE"
                                              : "AMOUNT"}
                                          <input
                                            inputMode="decimal"
                                            value={item[k]}
                                            placeholder="—"
                                            aria-label={
                                              "Item " +
                                              (i + 1) +
                                              " " +
                                              k.replace("_", " ")
                                            }
                                            onChange={(e) => {
                                              setItems((a) =>
                                                a.map((v, j) =>
                                                  j === i
                                                    ? {
                                                        ...v,
                                                        [k]: e.target.value,
                                                      }
                                                    : v,
                                                ),
                                              );
                                              setDirty(true);
                                            }}
                                          />
                                        </label>
                                      ))}
                                    </div>
                                  </div>
                                ))}
                                {!items.length && (
                                  <p className="muted">
                                    No line items extracted. Add them from the
                                    source.
                                  </p>
                                )}
                              </div>
                              {!!doc.extraction.warnings?.length && (
                                <div className="warnings">
                                  <strong>Review notes</strong>
                                  <ul>
                                    {doc.extraction.warnings.map((w, i) => (
                                      <li key={i}>{w.replaceAll("_", " ")}</li>
                                    ))}
                                  </ul>
                                </div>
                              )}
                              <details className="ocr">
                                <summary>
                                  OCR evidence{" "}
                                  <span>
                                    {doc.extraction.lines?.length || 0} lines
                                  </span>
                                </summary>
                                {doc.extraction.lines?.map((l, i) => (
                                  <div key={i}>
                                    <span>{l.text}</span>
                                    <small>{l.label}</small>
                                  </div>
                                ))}
                              </details>
                              <div className="model-note">
                                <FlaskConical size={13} />
                                <span>
                                  {doc.extraction.model_version ||
                                    "Extraction model"}
                                  <br />
                                  Model scores and source labels refer to
                                  original extraction.
                                </span>
                              </div>
                            </div>
                            <div className="review-actions">
                              <div>
                                {dirty ? (
                                  <>
                                    <i />
                                    Unsaved changes
                                    <button onClick={() => hydrate(doc)}>
                                      Discard
                                    </button>
                                  </>
                                ) : (
                                  <>
                                    <Check size={13} />
                                    All changes saved
                                  </>
                                )}
                              </div>
                              <div>
                                <button
                                  className="button secondary"
                                  disabled={busy || !dirty}
                                  onClick={() => void save(false)}
                                >
                                  Save edits
                                </button>
                                <button
                                  className="button approve"
                                  disabled={
                                    busy ||
                                    (doc.status === "approved" && !dirty)
                                  }
                                  onClick={() => void save(true)}
                                >
                                  {busy ? (
                                    <LoaderCircle className="spin" size={15} />
                                  ) : (
                                    <CheckCheck size={16} />
                                  )}{" "}
                                  {doc.status === "approved" && !dirty
                                    ? "Approved"
                                    : "Approve receipt"}
                                </button>
                              </div>
                            </div>
                          </>
                        )}
                      </section>
                    </>
                  )}
                </div>
              )}
              <div className="workspace-note">
                <span>
                  Human review is part of the pipeline. Confidence is a signal,
                  not a guarantee.
                </span>
                <button onClick={() => navigate("model")}>
                  How the model was evaluated <ArrowRight size={13} />
                </button>
              </div>
            </>
          )}
          {view === "model" && <ModelLab report={report} />}
          {view === "activity" && (
            <div className="activity-layout">
              <section>
                <div className="section-label">DOCUMENT AUDIT TRAIL</div>
                <h2>{doc?.filename || "No receipt selected"}</h2>
                <p className="muted">
                  Corrections increment the document revision and preserve
                  original confidence.
                </p>
                {events.length ? (
                  <ol className="audit-list">
                    {[...events].reverse().map((e, i) => (
                      <li key={e.id || i}>
                        <span>
                          <Activity size={17} />
                        </span>
                        <div>
                          <h3>
                            {text(
                              e.action ||
                                e.event ||
                                e.event_type ||
                                "Document event",
                            ).replaceAll("_", " ")}
                          </h3>
                          <small>
                            {date(e.created_at || e.timestamp)}
                            {e.revision !== undefined
                              ? " · Revision " + e.revision
                              : ""}
                          </small>
                          {(e.details ?? e.detail) !== undefined && (
                            <pre>
                              {typeof (e.details ?? e.detail) === "string"
                                ? text(e.details ?? e.detail)
                                : JSON.stringify(
                                    e.details ?? e.detail,
                                    null,
                                    2,
                                  )}
                            </pre>
                          )}
                          {e.after !== undefined && (
                            <details>
                              <summary>Inspect saved snapshot</summary>
                              <pre>{JSON.stringify(e.after, null, 2)}</pre>
                            </details>
                          )}
                        </div>
                      </li>
                    ))}
                  </ol>
                ) : (
                  <div className="empty">
                    <Activity size={30} />
                    <h3>The trail starts with a receipt.</h3>
                    <p>Select a document to see its history.</p>
                  </div>
                )}
              </section>
              <aside>
                <ShieldCheck size={30} />
                <h2>
                  Accountable
                  <br />
                  <em>by design.</em>
                </h2>
                <p>
                  The source image, original confidence, correction revision,
                  and review event remain connected.
                </p>
                <dl>
                  <dt>Document ID</dt>
                  <dd>{doc?.id || "—"}</dd>
                  <dt>Current revision</dt>
                  <dd>{doc?.revision || "—"}</dd>
                  <dt>Storage</dt>
                  <dd>
                    {api.DEMO
                      ? "Your browser · sample review"
                      : "Local API · SQLite"}
                  </dd>
                </dl>
              </aside>
            </div>
          )}
          <footer>
            <span>RECEIPTLAB / BUILT FOR EVIDENCE.</span>
            <span>OCR + machine learning + human judgment</span>
          </footer>
        </div>
      </main>
      {drag && (
        <div className="drop">
          <Upload size={44} />
          <h2>Drop your receipt here.</h2>
          <p>PNG, JPG, or PDF · up to 10 MB</p>
        </div>
      )}
      {toast && (
        <div className="toast" role="status">
          <Check size={18} />
          <span>{toast}</span>
          <button
            className="icon"
            onClick={() => setToast("")}
            aria-label="Dismiss notification"
          >
            <X size={15} />
          </button>
        </div>
      )}
    </div>
  );
}
