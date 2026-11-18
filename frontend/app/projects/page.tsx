"use client";

import { useMemo, useRef, useState, useSyncExternalStore } from "react";
import { ArrowDownUp, Database, FolderPlus, HardDriveUpload, Layers3, Search } from "lucide-react";
import { Button, EmptyState, ErrorAlert, Modal, Notice, Panel, SectionTitle, StatusBadge } from "@/components/ui";
import { useResource } from "@/hooks/use-resource";
import type { SceneMetadata } from "@/lib/api";

type Project = { id: string; name: string; createdAt: string };
type LocalDataset = { id: string; name: string; size: number; modifiedAt: string; projectId: string | null };
const PROJECTS_KEY = "orbit.projects.v1";
const DATASETS_KEY = "orbit.datasets.v1";
const STORAGE_EVENT = "orbit-workspace-storage";

export default function ProjectsPage() {
  const projectsJson = useSyncExternalStore(subscribeToWorkspace, () => readStoredValue(PROJECTS_KEY), () => "[]");
  const filesJson = useSyncExternalStore(subscribeToWorkspace, () => readStoredValue(DATASETS_KEY), () => "[]");
  const projectState = useMemo(() => parseList<Project>(projectsJson), [projectsJson]);
  const fileState = useMemo(() => parseList<LocalDataset>(filesJson), [filesJson]);
  const projects = projectState.items;
  const files = fileState.items;
  const [showProject, setShowProject] = useState(false);
  const [projectName, setProjectName] = useState("");
  const [selectedProject, setSelectedProject] = useState("");
  const [search, setSearch] = useState("");
  const [sort, setSort] = useState("name");
  const [notice, setNotice] = useState<string | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const sample = useResource<SceneMetadata>("/dataset/sample");
  const inventory = useResource<{ configured: boolean; file_count: number; note: string }>("/dataset/info");

  function saveProjects(next: Project[]) {
    try {
      localStorage.setItem(PROJECTS_KEY, JSON.stringify(next));
      window.dispatchEvent(new Event(STORAGE_EVENT));
      return true;
    } catch (error) {
      setNotice(`Could not save project metadata: ${error instanceof Error ? error.message : "browser storage is unavailable"}`);
      return false;
    }
  }

  function saveFiles(next: LocalDataset[]) {
    try {
      localStorage.setItem(DATASETS_KEY, JSON.stringify(next));
      window.dispatchEvent(new Event(STORAGE_EVENT));
      return true;
    } catch (error) {
      setNotice(`Could not save file metadata: ${error instanceof Error ? error.message : "browser storage is unavailable"}`);
      return false;
    }
  }

  function createProject(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const name = projectName.trim();
    if (!name) return;
    const next = [{ id: crypto.randomUUID(), name, createdAt: new Date().toISOString() }, ...projects];
    if (!saveProjects(next)) return;
    setProjectName("");
    setShowProject(false);
    setNotice(`Project “${name}” saved in this browser.`);
  }

  function handleUpload(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    if (!/\.(tif|tiff)$/i.test(file.name)) {
      setNotice("Only GeoTIFF files are accepted here. The selected file was not saved.");
      return;
    }
    const item: LocalDataset = {
      id: crypto.randomUUID(),
      name: file.name,
      size: file.size,
      modifiedAt: new Date(file.lastModified).toISOString(),
      projectId: selectedProject || null,
    };
    if (!saveFiles([item, ...files])) return;
    setNotice("File metadata saved in this browser only. No image bytes were uploaded or validated.");
  }

  const filteredFiles = useMemo(() => {
    return files
      .filter((file) => file.name.toLowerCase().includes(search.toLowerCase()))
      .sort((a, b) => sort === "date" ? b.modifiedAt.localeCompare(a.modifiedAt) : a.name.localeCompare(b.name));
  }, [files, search, sort]);

  return (
    <>
      <SectionTitle
        eyebrow="WORKSPACE"
        title="Projects & datasets"
        detail="Browse the configured SEN12MS-CR-TS inventory and organize project metadata for this browser."
        action={
          <>
            <Button onClick={() => fileInput.current?.click()} variant="secondary"><HardDriveUpload size={15} /> Add GeoTIFF</Button>
            <Button onClick={() => setShowProject(true)}><FolderPlus size={15} /> New project</Button>
          </>
        }
      />
      <input accept=".tif,.tiff" aria-label="Choose a GeoTIFF" hidden onChange={handleUpload} ref={fileInput} type="file" />
      <div className="toolbar project-toolbar">
        <label className="field">
          <span>Attach to project (local only)</span>
          <select onChange={(event) => setSelectedProject(event.target.value)} value={selectedProject}>
            <option value="">No project</option>
            {projects.map((project) => <option key={project.id} value={project.id}>{project.name}</option>)}
          </select>
        </label>
        <div className="project-inventory">
          <Database size={15} />
          <span>SEN12MS-CR-TS inventory</span>
          <strong>{inventory.data?.configured ? inventory.data.file_count.toLocaleString() : inventory.loading ? "…" : "Unavailable"}</strong>
          <StatusBadge tone={inventory.data?.configured ? "success" : "warning"}>{inventory.data?.configured ? "Configured" : "Not configured"}</StatusBadge>
        </div>
      </div>
      {notice && <Notice tone="info">{notice}</Notice>}
      {(projectState.error || fileState.error) && <Notice tone="warning">{projectState.error || fileState.error}</Notice>}
      <Notice tone="warning">
        Project and added-file records are saved in browser local storage only. The API has no project or dataset upload endpoint; files are not sent to a server, and their bands, CRS, pairing, and alignment have not been validated.
      </Notice>
      {inventory.error && <ErrorAlert message={inventory.error} onRetry={() => void inventory.refresh()} />}

      <div className="panel-grid grid-2 projects-panels">
        <Panel className="panel-pad">
          <div className="panel-heading">
            <div><h2>Projects</h2><p>{`${projects.length} browser workspace${projects.length === 1 ? "" : "s"}`}</p></div>
            <span className="kpi-icon"><FolderPlus size={14} /></span>
          </div>
          <div className="project-list">
            {projects.length ? projects.map((project) => (
              <article className="project-row" key={project.id}>
                <span className="project-icon"><Layers3 size={16} /></span>
                <span className="project-name"><strong>{project.name}</strong><small>Created {new Date(project.createdAt).toLocaleDateString()} · browser only</small></span>
                <span className="project-count">{files.filter((file) => file.projectId === project.id).length} files</span>
              </article>
            )) : <EmptyState title="No projects yet" detail="Create a browser-local project to group file metadata and research notes." action={<Button onClick={() => setShowProject(true)} variant="secondary"><FolderPlus size={14} /> Create project</Button>} />}
          </div>
        </Panel>

        <Panel className="panel-pad scene-summary">
          <div className="panel-heading">
            <div><h2>Held-out scene metadata</h2><p>Read from the first complete test-manifest pair</p></div>
            <StatusBadge tone="cyan">Test split</StatusBadge>
          </div>
          {sample.data ? (
            <div className="scene-metadata">
              <Meta label="Scene" value={sample.data.scene_id} />
              <Meta label="Acquisition date" value={sample.data.acquisition_date} />
              <Meta label="Optical sensor" value={`Sentinel-2 · ${sample.data.cloudy_optical.bands} bands`} />
              <Meta label="SAR sensor" value={`Sentinel-1 · ${sample.data.sar.bands} bands`} />
              <Meta label="Dimensions" value={`${sample.data.cloudy_optical.width} × ${sample.data.cloudy_optical.height}px`} />
              <Meta label="CRS" value={sample.data.cloudy_optical.crs ?? "Not recorded"} />
              <Meta label="Coordinates" value={sample.data.center?.map((value) => value.toFixed(5)).join(", ") ?? "Not recorded"} />
            </div>
          ) : sample.error ? <ErrorAlert message={sample.error} onRetry={() => void sample.refresh()} /> : <div className="loading-state">Reading scene metadata…</div>}
        </Panel>
      </div>

      <Panel className="panel-pad datasets-panel">
        <div className="panel-heading panel-heading-wrap">
          <div><h2>Scene files</h2><p>Actual sample metadata and browser-local file records are shown separately.</p></div>
          <div className="toolbar">
            <label className="search-inline"><Search size={14} /><input aria-label="Filter datasets" onChange={(event) => setSearch(event.target.value)} placeholder="Filter files…" value={search} /></label>
            <label className="sort-control"><ArrowDownUp size={13} /><select aria-label="Sort datasets" onChange={(event) => setSort(event.target.value)} value={sort}><option value="name">Name</option><option value="date">Date</option></select></label>
          </div>
        </div>
        <div className="table-wrap">
          <table className="data-table">
            <thead><tr><th>Scene / file</th><th>Sensor</th><th>Bands</th><th>Dimensions</th><th>CRS / date</th><th>Validation</th></tr></thead>
            <tbody>
              {sample.data && <tr>
                <td><strong>{sample.data.scene_id}</strong><span className="table-secondary">Held-out manifest · cloudy optical</span></td>
                <td>Sentinel-2</td><td>{sample.data.cloudy_optical.bands}</td><td>{sample.data.cloudy_optical.width} × {sample.data.cloudy_optical.height}</td>
                <td>{sample.data.cloudy_optical.crs ?? "—"}<span className="table-secondary">{sample.data.acquisition_date}</span></td>
                <td><StatusBadge tone="success">Manifest pair found</StatusBadge></td>
              </tr>}
              {filteredFiles.map((file) => (
                <tr key={file.id}>
                  <td><strong>{file.name}</strong><span className="table-secondary">{(file.size / (1024 * 1024)).toFixed(2)} MB · local metadata only</span></td>
                  <td>Not inspected</td><td>Not inspected</td><td>Not inspected</td>
                  <td>{new Date(file.modifiedAt).toLocaleDateString()}</td><td><StatusBadge tone="warning">Not uploaded / validated</StatusBadge></td>
                </tr>
              ))}
              {!sample.data && filteredFiles.length === 0 && <tr><td colSpan={6}><EmptyState title="No scene records" detail="Connect the API to load a real manifest sample, or add local file metadata." /></td></tr>}
            </tbody>
          </table>
        </div>
      </Panel>

      {showProject && <Modal title="Create a project" detail="This project is saved to local browser storage, not the backend." onClose={() => setShowProject(false)}>
        <form onSubmit={createProject}>
          <label className="field"><span>Project name</span><input autoFocus className="text-input" maxLength={80} onChange={(event) => setProjectName(event.target.value)} placeholder="e.g. Seasonal cloud-removal study" required value={projectName} /></label>
          <div className="modal-actions"><Button onClick={() => setShowProject(false)} type="button" variant="quiet">Cancel</Button><Button disabled={!projectName.trim()} type="submit"><FolderPlus size={14} /> Create project</Button></div>
        </form>
      </Modal>}
    </>
  );
}

function Meta({ label, value }: { label: string; value: string }) {
  return <div><span>{label}</span><strong>{value}</strong></div>;
}

function subscribeToWorkspace(onChange: () => void) {
  window.addEventListener("storage", onChange);
  window.addEventListener(STORAGE_EVENT, onChange);
  return () => {
    window.removeEventListener("storage", onChange);
    window.removeEventListener(STORAGE_EVENT, onChange);
  };
}

function readStoredValue(key: string): string {
  try {
    return window.localStorage.getItem(key) ?? "[]";
  } catch {
    return "__browser_storage_unavailable__";
  }
}

function parseList<T>(serialized: string): { items: T[]; error: string | null } {
  try {
    const parsed: unknown = JSON.parse(serialized);
    if (!Array.isArray(parsed)) throw new Error("stored value is not a list");
    return { items: parsed as T[], error: null };
  } catch {
    return { items: [], error: "Saved browser workspace data is unreadable; existing records were left unchanged." };
  }
}
