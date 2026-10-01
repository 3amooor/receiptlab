export type Key = "subtotal" | "tax" | "discount" | "total";
export type Field = {
  value: string | number | null;
  confidence?: number;
  bbox?: [number, number, number, number] | null;
  origin?: string;
};
export type Item = {
  description: string;
  quantity: string | number | null;
  unit_price: string | number | null;
  amount: string | number | null;
  confidence?: number;
};
export type EditItem = {
  description: string;
  quantity: string;
  unit_price: string;
  amount: string;
};
export type Doc = {
  id: string;
  filename: string;
  created_at: string;
  status: string;
  source?: string;
  page_count?: number;
  width?: number;
  height?: number;
  image_url?: string;
  revision: number;
  error?: string | null;
  extraction?: {
    fields: Partial<Record<Key, Field>>;
    items: Item[];
    lines?: {
      text: string;
      bbox: number[];
      label?: string;
      confidence?: number;
    }[];
    warnings?: string[];
    confidence?: number;
    model_version?: string;
    needs_review?: boolean;
  } | null;
};
export type Audit = {
  id?: string | number;
  action?: string;
  event?: string;
  event_type?: string;
  created_at?: string;
  timestamp?: string;
  revision?: number;
  details?: unknown;
  detail?: unknown;
  before?: unknown;
  after?: unknown;
};
export type Report = Record<string, unknown>;
export type Demo = {
  documents: Doc[];
  audit: Record<string, Audit[]>;
  metrics: Report;
};
