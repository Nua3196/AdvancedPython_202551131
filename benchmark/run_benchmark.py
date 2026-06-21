"""
benchmark/run_benchmark.py

before/after SAM3 batch script를 동일 조건으로 순차 실행하고,
전체 실행 시간과 peak memory를 CSV로 저장.

측정 지표:
- version: before / after
- repeat: 반복 번호
- elapsed_sec: 전체 실행 시간
- peak_rss_mb: 프로세스 트리 기준 CPU 메모리 peak RSS
- peak_gpu_memory_mb: nvidia-smi 기준 GPU memory peak

예시:
python benchmark/run_benchmark.py \
  --before_script src/before/run_sam3_batch.py \
  --after_script src/after/run_sam3_batch_after.py \
  --input_dir data/inputs \
  --results_csv results/benchmark_results.csv \
  --output_root results/benchmark_outputs \
  --prompts "broccoli" \
  --repeats 3 \
  --gpu_id 0
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


try:
    import psutil
except ImportError:
    psutil = None


@dataclass(frozen=True)
class BenchmarkCase:
    version: str
    script_path: Path


@dataclass
class BenchmarkResult:
    version: str
    repeat: int
    elapsed_sec: float
    peak_rss_mb: Optional[float]
    peak_gpu_memory_mb: Optional[float]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark before/after SAM3 batch scripts under identical conditions."
    )

    parser.add_argument("--before_script", required=True, type=Path)
    parser.add_argument("--after_script", required=True, type=Path)
    parser.add_argument("--input_dir", required=True, type=Path)

    parser.add_argument(
        "--results_csv",
        default=Path("results/benchmark_results.csv"),
        type=Path,
    )
    parser.add_argument(
        "--output_root",
        default=Path("results/benchmark_outputs"),
        type=Path,
    )

    parser.add_argument("--prompts", nargs="+", required=True)
    parser.add_argument("--repeats", type=int, default=3)

    parser.add_argument("--gpu_id", default="0")
    parser.add_argument("--python_bin", default=sys.executable)

    parser.add_argument("--suffix", default="_sam3")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--fps_out", type=float, default=None)
    parser.add_argument("--prompt_frame", type=int, default=0)

    parser.add_argument(
        "--keep_outputs",
        action="store_true",
        help="Do not delete output dirs before each run.",
    )

    parser.add_argument(
        "--poll_interval",
        type=float,
        default=0.2,
        help="Memory polling interval in seconds.",
    )

    return parser.parse_args()


class ProcessTreeMonitor:
    """
    실행 중인 subprocess와 그 자식 프로세스들의 메모리 사용량을 주기적으로 측정.

    peak_rss_mb:
        psutil 기준 parent + children RSS 합의 peak.

    peak_gpu_memory_mb:
        nvidia-smi query-compute-apps 기준 target process tree의 GPU memory 합 peak.
    """

    def __init__(self, pid: int, poll_interval: float = 0.2):
        self.pid = pid
        self.poll_interval = poll_interval
        self.peak_rss_mb: Optional[float] = None
        self.peak_gpu_memory_mb: Optional[float] = None

        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=2)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                pids = self._collect_process_tree_pids()

                rss_mb = self._get_process_tree_rss_mb(pids)
                if rss_mb is not None:
                    if self.peak_rss_mb is None or rss_mb > self.peak_rss_mb:
                        self.peak_rss_mb = rss_mb

                gpu_mb = self._get_gpu_memory_mb_for_pids(pids)
                if gpu_mb is not None:
                    if self.peak_gpu_memory_mb is None or gpu_mb > self.peak_gpu_memory_mb:
                        self.peak_gpu_memory_mb = gpu_mb

            except Exception:
                pass

            time.sleep(self.poll_interval)

    def _collect_process_tree_pids(self) -> List[int]:
        try:
            proc = psutil.Process(self.pid)
        except psutil.NoSuchProcess:
            return []

        processes = [proc]

        try:
            processes.extend(proc.children(recursive=True))
        except psutil.Error:
            pass

        pids: List[int] = []

        for process in processes:
            try:
                if process.is_running():
                    pids.append(process.pid)
            except psutil.Error:
                continue

        return pids

    def _get_process_tree_rss_mb(self, pids: List[int]) -> Optional[float]:
        if not pids:
            return None

        total_bytes = 0

        for pid in pids:
            try:
                proc = psutil.Process(pid)
                total_bytes += proc.memory_info().rss
            except psutil.Error:
                continue

        return total_bytes / (1024 * 1024)

    def _get_gpu_memory_mb_for_pids(self, pids: List[int]) -> Optional[float]:
        """
        nvidia-smi에서 현재 GPU를 사용하는 process별 memory를 읽고,
        benchmark 대상 process tree에 해당하는 pid만 합산.
        """
        if not pids:
            return None

        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader,nounits",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            return None

        if completed.returncode != 0:
            return None

        target_pids = set(pids)
        total_mb = 0.0
        found = False

        for line in completed.stdout.splitlines():
            line = line.strip()

            if not line:
                continue

            parts = [part.strip() for part in line.split(",")]

            if len(parts) < 2:
                continue

            try:
                pid = int(parts[0])
                used_mb = float(parts[1])
            except ValueError:
                continue

            if pid in target_pids:
                total_mb += used_mb
                found = True

        return total_mb if found else None


def build_command(
    *,
    python_bin: str,
    script_path: Path,
    input_dir: Path,
    output_dir: Path,
    prompts: List[str],
    suffix: str,
    alpha: float,
    fps_out: Optional[float],
    prompt_frame: int,
    gpu_id: str,
) -> List[str]:
    command = [
        python_bin,
        str(script_path),
        "--input_dir",
        str(input_dir),
        "--output_dir",
        str(output_dir),
        "--suffix",
        suffix,
        "--alpha",
        str(alpha),
        "--prompt_frame",
        str(prompt_frame),
        "--gpu_ids",
        str(gpu_id),
    ]

    if fps_out is not None:
        command.extend(["--fps_out", str(fps_out)])

    command.append("--prompts")
    command.extend(prompts)

    return command


def run_one_case(
    *,
    case: BenchmarkCase,
    repeat: int,
    args,
) -> BenchmarkResult:
    """
    before 또는 after 스크립트 하나를 1회 실행하고 지표를 반환.
    """
    output_dir = args.output_root / case.version / f"repeat_{repeat}"

    if output_dir.exists() and not args.keep_outputs:
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    command = build_command(
        python_bin=args.python_bin,
        script_path=case.script_path,
        input_dir=args.input_dir,
        output_dir=output_dir,
        prompts=args.prompts,
        suffix=args.suffix,
        alpha=args.alpha,
        fps_out=args.fps_out,
        prompt_frame=args.prompt_frame,
        gpu_id=args.gpu_id,
    )

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    print(f"\n[RUN] {case.version} repeat={repeat}")
    print("[INFO] Sequential benchmark: next run starts after this process exits.")
    print(" ".join(command))

    start_time = time.perf_counter()

    log_dir = args.output_root / "logs" / case.version
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / f"repeat_{repeat}.log"

    with log_path.open("w", encoding="utf-8") as log_f:
        log_f.write(f"[RUN] {case.version} repeat={repeat}\n")
        log_f.write("[COMMAND]\n")
        log_f.write(" ".join(command) + "\n\n")
        log_f.write("[OUTPUT]\n")
        log_f.flush()

        proc = subprocess.Popen(
            command,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
        )

        monitor = ProcessTreeMonitor(
            pid=proc.pid,
            poll_interval=args.poll_interval,
        )
        monitor.start()

        try:
            return_code = proc.wait()
        finally:
            monitor.stop()

    elapsed_sec = time.perf_counter() - start_time

    if return_code != 0:
        raise RuntimeError(
            f"Benchmark failed: version={case.version}, "
            f"repeat={repeat}, return_code={return_code}"
        )

    return BenchmarkResult(
        version=case.version,
        repeat=repeat,
        elapsed_sec=elapsed_sec,
        peak_rss_mb=monitor.peak_rss_mb,
        peak_gpu_memory_mb=monitor.peak_gpu_memory_mb,
    )


def save_results(results: List[BenchmarkResult], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "version",
        "repeat",
        "elapsed_sec",
        "peak_rss_mb",
        "peak_gpu_memory_mb",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            writer.writerow(
                {
                    "version": result.version,
                    "repeat": result.repeat,
                    "elapsed_sec": result.elapsed_sec,
                    "peak_rss_mb": result.peak_rss_mb,
                    "peak_gpu_memory_mb": result.peak_gpu_memory_mb,
                }
            )


def main() -> int:
    args = parse_args()

    if psutil is None:
        raise RuntimeError(
            "psutil is required for benchmark memory measurement. "
            "Install it with: pip install psutil"
        )

    cases = [
        BenchmarkCase(version="before", script_path=args.before_script),
        BenchmarkCase(version="after", script_path=args.after_script),
    ]

    results: List[BenchmarkResult] = []

    for repeat in range(args.repeats):
        for case in cases:
            result = run_one_case(
                case=case,
                repeat=repeat,
                args=args,
            )

            results.append(result)

            rss_text = (
                f"{result.peak_rss_mb:.1f}MB"
                if result.peak_rss_mb is not None
                else "N/A"
            )
            gpu_text = (
                f"{result.peak_gpu_memory_mb:.1f}MB"
                if result.peak_gpu_memory_mb is not None
                else "N/A"
            )

            print(
                f"[DONE] {result.version} repeat={result.repeat} | "
                f"time={result.elapsed_sec:.3f}s | "
                f"rss_peak={rss_text} | "
                f"gpu_peak={gpu_text}"
            )

    save_results(results, args.results_csv)

    print(f"\n[REPORT] Benchmark results saved to: {args.results_csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())