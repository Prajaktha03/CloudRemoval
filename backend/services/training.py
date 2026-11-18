from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import os
import shlex
import subprocess
import threading
import tempfile
import uuid


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TrainingJobManager:
    def __init__(self, command: str | None) -> None:
        self.command = command
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="training")
        self._lock = threading.Lock()
        self._jobs: dict[str, dict] = {}
        self._active_job: str | None = None

    def start(self, config_path: str | None = None) -> dict:
        if not self.command:
            raise RuntimeError("Training is disabled. Set TRAIN_COMMAND to an operator-controlled training command.")
        try:
            args = shlex.split(self.command, posix=True)
        except ValueError as exc:
            raise RuntimeError("TRAIN_COMMAND must be a valid quoted command") from exc
        if not args:
            raise RuntimeError("TRAIN_COMMAND must not be empty")
        with self._lock:
            if self._active_job is not None:
                active = self._jobs[self._active_job]
                if active["status"] in {"queued", "running"}:
                    raise ValueError(f"Training job {self._active_job} is already {active['status']}")
            job_id = str(uuid.uuid4())
            job = {
                "job_id": job_id,
                "status": "queued",
                "started_at": _now(),
                "finished_at": None,
                "return_code": None,
                "message": None,
            }
            self._jobs[job_id] = job
            self._active_job = job_id
            if config_path:
                args.extend(["--config", config_path])
            self._executor.submit(self._run, job_id, args)
            return dict(job)

    def _run(self, job_id: str, args: list[str]) -> None:
        with self._lock:
            self._jobs[job_id]["status"] = "running"
        try:
            with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as output:
                result = subprocess.run(args, check=False, stdout=output, stderr=subprocess.STDOUT, timeout=None)
                output.seek(0, os.SEEK_END)
                output.seek(max(0, output.tell() - 2000))
                message = output.read()[-2000:].strip() or None
            with self._lock:
                job = self._jobs[job_id]
                job["return_code"] = result.returncode
                job["status"] = "succeeded" if result.returncode == 0 else "failed"
                job["message"] = message
                job["finished_at"] = _now()
        except OSError as exc:
            with self._lock:
                job = self._jobs[job_id]
                job.update(status="failed", message=str(exc), finished_at=_now())

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None
