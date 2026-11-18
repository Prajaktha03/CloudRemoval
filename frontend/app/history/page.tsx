"use client";

import { Suspense, useMemo, useState } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { ArrowRight, ChevronLeft, ChevronRight, ExternalLink, RefreshCcw, Search, TimerReset } from "lucide-react";
import { ErrorAlert, LoadingState, Modal, Notice, Panel, SectionTitle, StatusBadge } from "@/components/ui";
import { useResource } from "@/hooks/use-resource";
import { formatDate } from "@/lib/api";
import type { Experiment } from "@/lib/api";

const PAGE_SIZE = 10;
const EMPTY_RECORDS: Experiment[] = [];

export default function HistoryPage() {
  return <Suspense fallback={<LoadingState label="Opening processing history…" />}><HistoryContent /></Suspense>;
}

function HistoryContent() {
  const { data, error, loading, refresh } = useResource<{ experiments: Experiment[] }>("/experiments?limit=200");
  const searchParams = useSearchParams();
  const [search, setSearch] = useState(searchParams.get("search") ?? "");
  const [status, setStatus] = useState("all");
  const [page, setPage] = useState(1);
  const [selected, setSelected] = useState<Experiment | null>(null);

  const records = data?.experiments ?? EMPTY_RECORDS;
  const filtered = useMemo(() => records.filter((record) => {
    if (status === "prediction" && record.kind !== "prediction") return false;
    if (status === "evaluation" && record.kind !== "evaluation") return false;
    const haystack = [record.id, record.kind, record.created_at, ...Object.values(record.metrics).map(String)].join(" ").toLowerCase();
    return haystack.includes(search.toLowerCase());
  }), [records, search, status]);
  const pageCount = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const current = filtered.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);

  return (
    <>
      <SectionTitle
        eyebrow="EXPERIMENT RECORDS"
        title="Processing history"
        detail="Review saved successful API operations. Failed jobs, source files, and per-job duration are not persisted by the current service."
        action={<button className="button button-secondary" onClick={() => void refresh()} type="button"><RefreshCcw size={14} /> Refresh</button>}
      />
      {error && <ErrorAlert message={error} onRetry={() => void refresh()} />}
      <Notice tone="warning">
        This is the backend experiment ledger, not a complete job queue: only successful prediction/evaluation summaries are stored. Inference is synchronous, so there is no live job timeline, failure count, retry record, or processing duration.
      </Notice>

      <Panel className="panel-pad history-panel">
        <div className="panel-heading panel-heading-wrap">
          <div><h2>Saved operations</h2><p>{loading ? "Loading from the experiment database…" : `${filtered.length} of ${records.length} saved records`}</p></div>
          <div className="toolbar">
            <label className="search-inline"><Search size={14} /><input aria-label="Search history" onChange={(event) => { setSearch(event.target.value); setPage(1); }} placeholder="Search ID, type, or saved metrics…" value={search} /></label>
            <label className="sort-control"><span className="sr-only">Filter by operation</span><select aria-label="Filter by operation" onChange={(event) => { setStatus(event.target.value); setPage(1); }} value={status}><option value="all">All records</option><option value="prediction">Predictions</option><option value="evaluation">Evaluations</option></select></label>
          </div>
        </div>
        {loading ? <LoadingState label="Loading experiment ledger…" /> : (
          <>
            <div className="table-wrap">
              <table className="data-table">
                <thead><tr><th>Saved record ID</th><th>Operation</th><th>Input scene</th><th>Model</th><th>Status</th><th>Created</th><th>Duration</th><th>Details</th></tr></thead>
                <tbody>
                  {current.map((record) => (
                    <tr key={`${record.kind}-${record.id}`}>
                      <td><strong>EXP-{String(record.id).padStart(5, "0")}</strong></td>
                      <td><strong>{record.kind === "prediction" ? "Cloud removal" : "Evaluation"}</strong><span className="table-secondary">{record.kind}</span></td>
                      <td>Not saved<span className="table-secondary">Input files are discarded</span></td>
                      <td>{record.kind === "prediction" ? "Configured U-Net" : "Reference evaluation"}</td>
                      <td><StatusBadge tone="success">Completed</StatusBadge></td>
                      <td>{formatDate(record.created_at)}</td>
                      <td><span className="table-secondary">Not recorded</span></td>
                      <td><button className="table-action" onClick={() => setSelected(record)} type="button">View record <ExternalLink size={12} /></button></td>
                    </tr>
                  ))}
                  {current.length === 0 && <tr><td colSpan={8}><div className="empty-state"><h3>{records.length ? "No matching records" : "No saved operations yet"}</h3><p>{records.length ? "Try another query or filter." : "Successful predictions and reference evaluations will appear here after they are sent to the API."}</p><Link className="button button-secondary" href="/studio">Open cloud removal studio <ArrowRight size={13} /></Link></div></td></tr>}
                </tbody>
              </table>
            </div>
            <div className="pagination">
              <span>Page {Math.min(page, pageCount)} of {pageCount}</span>
              <div className="pagination-controls">
                <button aria-label="Previous page" disabled={page <= 1} onClick={() => setPage((value) => Math.max(1, value - 1))} type="button"><ChevronLeft size={14} /></button>
                <button aria-label="Next page" disabled={page >= pageCount} onClick={() => setPage((value) => Math.min(pageCount, value + 1))} type="button"><ChevronRight size={14} /></button>
              </div>
            </div>
          </>
        )}
      </Panel>
      <div className="history-footer">
        <div className="kpi-icon"><TimerReset size={14} /></div>
        <span>Need to rerun an operation? The API does not retain source imagery; select the inputs again in the studio.</span>
        <Link className="inline-link" href="/studio">Start a reconstruction <ArrowRight size={13} /></Link>
      </div>

      {selected && <Modal title={`Saved record EXP-${String(selected.id).padStart(5, "0")}`} detail={`Created ${formatDate(selected.created_at)} · ${selected.kind} summary from the API ledger`} onClose={() => setSelected(null)}>
        <div className="timeline">
          <div className="timeline-item"><strong>{selected.kind === "prediction" ? "Prediction summary recorded" : "Evaluation summary recorded"}</strong><span>{formatDate(selected.created_at)}</span></div>
        </div>
        <div className="muted-divider" />
        <pre className="json-preview">{JSON.stringify(selected.metrics, null, 2)}</pre>
        <div className="modal-actions">
          <Link className="button button-secondary" href="/studio" onClick={() => setSelected(null)}><RefreshCcw size={13} /> Select inputs again</Link>
          <button className="button button-quiet" onClick={() => setSelected(null)} type="button">Close</button>
        </div>
      </Modal>}
    </>
  );
}
