import type { Audit, Demo, Doc, EditItem, Key, Report } from "./types";
export const DEMO = import.meta.env.VITE_DEMO_MODE === "1";
const BASE = (import.meta.env.VITE_API_BASE || "http://127.0.0.1:8010").replace(
  /\/$/,
  "",
);
const asset = (p: string) => import.meta.env.BASE_URL + p.replace(/^\//, "");
const storage = "receiptlab.review.v1";
let cache: Demo | undefined;
export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let r: Response;
  try {
    r = await fetch(BASE + path, init);
  } catch {
    throw Error(
      "Cannot reach the API. Start the local service on port 8010, then reconnect.",
    );
  }
  if (!r.ok) {
    let message = "Request failed (" + r.status + ").";
    try {
      const b = await r.json(),
        d = b.detail ?? b.error;
      message = typeof d === "string" ? d : d ? JSON.stringify(d) : message;
    } catch {
      /* preserve HTTP status */
    }
    throw Error(message);
  }
  return r.json() as Promise<T>;
}
async function samples() {
  if (cache) return cache;
  const r = await fetch(asset("demo-data.json"));
  if (!r.ok)
    throw Error("Sample data unavailable. Refresh or run the local app.");
  const data = (await r.json()) as Demo;
  try {
    const edits = JSON.parse(localStorage.getItem(storage) || "{}") as Record<
      string,
      { doc: Doc; audit: Audit[] }
    >;
    data.documents = data.documents.map((d) => edits[d.id]?.doc || d);
    for (const id of Object.keys(edits)) data.audit[id] = edits[id].audit;
  } catch {
    /* browser persistence optional */
  }
  cache = data;
  return data;
}
export const health = () =>
  DEMO ? Promise.resolve({ status: "sample" }) : request("/api/health");
export const documents = async () =>
  DEMO ? (await samples()).documents : request<Doc[]>("/api/documents");
export const document = async (id: string) => {
  if (!DEMO) return request<Doc>("/api/documents/" + encodeURIComponent(id));
  const d = (await samples()).documents.find((v) => v.id === id);
  if (!d) throw Error("Document not found.");
  return d;
};
export const audit = async (id: string) =>
  DEMO
    ? (await samples()).audit[id] || []
    : request<Audit[]>("/api/documents/" + encodeURIComponent(id) + "/audit");
export const metrics = async () =>
  DEMO ? (await samples()).metrics : request<Report>("/api/metrics");
export const sample = (name: string) =>
  request<Doc>("/api/samples/" + name, { method: "POST" });
export function image(d: Doc) {
  const p = d.image_url || "";
  return !p
    ? ""
    : /^https?:\/\//.test(p)
      ? p
      : DEMO
        ? asset(p)
        : BASE + (p.startsWith("/") ? p : "/" + p);
}
export async function upload(file: File) {
  const data = new FormData();
  data.append("file", file);
  return request<Doc>("/api/documents", { method: "POST", body: data });
}
export async function review(
  doc: Doc,
  fields: Record<Key, string>,
  items: EditItem[],
  approve: boolean,
) {
  const normalized = Object.fromEntries(
    Object.entries(fields).map(([k, v]) => [k, v.trim() || null]),
  );
  const clean = items.map((i) => ({
    description: i.description.trim(),
    quantity: i.quantity.trim() || null,
    unit_price: i.unit_price.trim() || null,
    amount: i.amount.trim() || null,
  }));
  if (!DEMO)
    return request<Doc>(
      "/api/documents/" + encodeURIComponent(doc.id) + "/review",
      {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          revision: doc.revision,
          fields: normalized,
          items: clean,
          approve,
        }),
      },
    );
  const money = (v: string | null) =>
    v === null || /^\d{1,8}(?:\.\d{1,2})?$/.test(v);
  if (Object.values(normalized).some((v) => !money(v)))
    throw Error(
      "Amounts must be non-negative numbers with up to two decimal places.",
    );
  if (
    clean.some(
      (i) =>
        !i.description ||
        i.description.length > 240 ||
        !money(i.unit_price) ||
        !money(i.amount) ||
        (i.quantity !== null &&
          (!/^\d{1,6}(?:\.\d{1,3})?$/.test(i.quantity) ||
            Number(i.quantity) <= 0)),
    )
  )
    throw Error("Check descriptions, quantities and line-item amounts.");
  const checks: string[] = [];
  if (!normalized.total) checks.push("Enter a grand total before approving.");
  if (
    normalized.subtotal !== null &&
    normalized.total !== null &&
    Math.abs(
      Number(normalized.subtotal) +
        Number(normalized.tax || 0) -
        Number(normalized.discount || 0) -
        Number(normalized.total),
    ) > 0.021
  )
    checks.push(
      "Amounts do not balance. Correct subtotal + tax − discount = total.",
    );
  if (
    clean.length &&
    clean.every((i) => i.amount !== null) &&
    normalized.subtotal !== null &&
    Math.abs(
      clean.reduce((sum, i) => sum + Number(i.amount), 0) -
        Number(normalized.subtotal),
    ) > 0.021
  )
    checks.push("Line-item amounts do not add up to the subtotal.");
  if (clean.some((i) => i.amount === null))
    checks.push("Each line item needs an amount.");
  if (
    clean.some(
      (i) =>
        i.quantity !== null &&
        i.unit_price !== null &&
        i.amount !== null &&
        Math.abs(Number(i.quantity) * Number(i.unit_price) - Number(i.amount)) >
          0.021,
    )
  )
    checks.push("Quantity × unit price must equal item amount.");
  if (approve && checks.length) throw Error(checks.join(" "));
  const data = await samples(),
    current = data.documents.find((d) => d.id === doc.id);
  if (!current || current.revision !== doc.revision)
    throw Error("Stale review. Reload this document before saving.");
  const updated: Doc = {
    ...current,
    status: approve ? "approved" : "needs_review",
    revision: current.revision + 1,
    extraction: {
      ...current.extraction!,
      fields: Object.fromEntries(
        Object.entries(normalized).map(([k, value]) => [
          k,
          { ...current.extraction?.fields[k as Key], value },
        ]),
      ),
      items: clean,
      warnings: checks,
      needs_review: !approve,
    },
  };
  data.documents = data.documents.map((d) => (d.id === doc.id ? updated : d));
  data.audit[doc.id] = [
    ...(data.audit[doc.id] || []),
    {
      action: approve ? "sample_review_approved" : "sample_review_saved",
      created_at: new Date().toISOString(),
      revision: updated.revision,
      before: current.extraction,
      after: updated.extraction,
      details: {
        mode: "Local browser correction. Source confidence refers to original extraction.",
      },
    },
  ];
  try {
    const edits = JSON.parse(localStorage.getItem(storage) || "{}");
    edits[doc.id] = { doc: updated, audit: data.audit[doc.id] };
    localStorage.setItem(storage, JSON.stringify(edits));
  } catch {
    /* retained in memory */
  }
  return updated;
}
export function download(blob: Blob, name: string) {
  const url = URL.createObjectURL(blob),
    a = window.document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
export async function exportFile(doc: Doc, format: "json" | "csv") {
  if (!DEMO) {
    const r = await fetch(
      BASE +
        "/api/documents/" +
        encodeURIComponent(doc.id) +
        "/export?format=" +
        format,
    );
    if (!r.ok) throw Error("Export failed (" + r.status + ").");
    download(
      await r.blob(),
      doc.filename.replace(/\.[^.]+$/, "") + "." + format,
    );
    return;
  }
  const esc = (v: unknown) => {
    const s = String(v ?? ""),
      safe = /^\s*[=+\-@]|^[\t\r\n]/.test(s) ? "'" + s : s;
    return '"' + safe.replaceAll('"', '""') + '"';
  };
  const rows = [
    ["record_type", "description", "quantity", "unit_price", "amount"],
    ...Object.entries(doc.extraction?.fields || {}).map(([k, v]) => [
      "field",
      k,
      "",
      "",
      v?.value,
    ]),
    ...(doc.extraction?.items || []).map((i) => [
      "item",
      i.description,
      i.quantity,
      i.unit_price,
      i.amount,
    ]),
  ];
  const text =
    format === "json"
      ? JSON.stringify({ mode: "sample_workspace", document: doc }, null, 2)
      : rows.map((r) => r.map(esc).join(",")).join("\r\n");
  download(
    new Blob([text], {
      type: format === "json" ? "application/json" : "text/csv;charset=utf-8",
    }),
    "sample-" + doc.filename.replace(/\.[^.]+$/, "") + "." + format,
  );
}
