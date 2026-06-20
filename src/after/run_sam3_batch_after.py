"""
[사용법]
1) 입력 폴더(mp4들) + 프롬프트 지정.
2) 단일 프롬프트:   --prompt "crab stick"
   여러 프롬프트:    --prompts "crab stick" "tomato" "egg"
3) 예시:
   python run_sam3_batch.py --input_dir ./inputs --output_dir ./outputs --prompt "crab stick"
   python run_sam3_batch.py --input_dir ./inputs --prompts "crab stick" "tomato"

[결과]
- 입력 폴더 내 모든 .mp4를 처리하여 결과 영상을 저장합니다.
- 출력 파일명: <원본이름><suffix>__<prompt>.mp4  (기본 suffix: _sam3)
"""
import os
import sys
import argparse
import gc
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import torch
from sam3.model_builder import build_sam3_video_predictor

import subprocess
from tempfile import NamedTemporaryFile

from tqdm import tqdm
from sam3.visualization_utils import load_frame, render_masklet_frame

import csv
import time
import tracemalloc
from functools import wraps

def parse_args():
    ap = argparse.ArgumentParser(description="Batch-run SAM3 video segmentation with one/multiple text prompts.")
    ap.add_argument("--input_dir", required=True, help="Folder containing input mp4 files.")
    ap.add_argument("--output_dir", default=None, help="Where to save output videos. (default: input_dir)")
    ap.add_argument("--suffix", default="_sam3", help="Suffix inserted after stem. (default: _sam3)")
    ap.add_argument("--alpha", type=float, default=0.5, help="Mask overlay alpha. (default: 0.5)")
    ap.add_argument("--fps_out", type=float, default=None, help="Output video FPS. (default: same as input video)")
    ap.add_argument("--skip_existing", action="store_true", help="Skip if output file already exists.")

    # 프롬프트 핸들링
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--prompt", type=str, default=None, help="Single text prompt.")
    g.add_argument("--prompts", nargs="+", default=None, help="Multiple text prompts. (space-separated)")

    # 옵션: 프레임 선택
    ap.add_argument("--prompt_frame", type=int, default=0, help="Frame index to add the text prompt. (default: 0)")

    # 옵션: GPU 선택
    ap.add_argument("--gpu_ids", type=str, default=None, help='nvidia-smi GPU ids, comma-separated. e.g. "1,3"')

    return ap.parse_args()

@dataclass(frozen=True)
class BatchRunConfig:
    input_dir: Path
    output_dir: Path
    suffix: str
    alpha: float
    fps_out: Optional[float]
    skip_existing: bool
    prompt_frame: int
    gpu_ids: Optional[str]

@dataclass(frozen=True)
class VideoPromptJob:
    video_path: Path
    prompt: str
    output_path: Path

@dataclass
class PromptRunResult:
    video_name: str
    prompt: str
    output_path: Path
    status: str
    error: Optional[str] = None
    elapsed_sec: Optional[float] = None
    peak_cpu_memory_mb: Optional[float] = None
    peak_cuda_memory_mb: Optional[float] = None

    @classmethod
    def done(cls, job: VideoPromptJob) -> "PromptRunResult":
        return cls(
            video_name=job.video_path.name,
            prompt=job.prompt,
            output_path=job.output_path,
            status="done",
        )

    @classmethod
    def skipped(cls, job: VideoPromptJob) -> "PromptRunResult":
        return cls(
            video_name=job.video_path.name,
            prompt=job.prompt,
            output_path=job.output_path,
            status="skipped",
        )

    @classmethod
    def failed(cls, job: VideoPromptJob, error: Exception) -> "PromptRunResult":
        return cls(
            video_name=job.video_path.name,
            prompt=job.prompt,
            output_path=job.output_path,
            status="failed",
            error=f"{type(error).__name__}: {error}",
        )
    
    def attach_metrics(
        self,
        elapsed_sec: float,
        peak_cpu_memory_mb: Optional[float],
        peak_cuda_memory_mb: Optional[float],
    ) -> "PromptRunResult":
        self.elapsed_sec = elapsed_sec
        self.peak_cpu_memory_mb = peak_cpu_memory_mb
        self.peak_cuda_memory_mb = peak_cuda_memory_mb
        return self

def slug_prompt(prompt: str, max_len: int = 40) -> str:
    s = prompt.strip().lower()
    s = re.sub(r"\s+", "_", s)                 # 공백 -> _
    s = re.sub(r"[^0-9a-zA-Z가-힣_+-]+", "", s) # 파일명 위험 문자 제거
    return s[:max_len] if len(s) > max_len else s


def list_videos(folder: Path) -> List[Path]:
    p = Path(folder)
    if not p.exists():
        raise FileNotFoundError(f"input_dir does not exist: {p}")
    if not p.is_dir():
        raise NotADirectoryError(f"input_dir is not a directory: {p}")
    return [f for f in sorted(p.iterdir()) if f.is_file() and f.suffix.lower() == ".mp4"]


def get_video_fps(video_path: Path, default_fps: float = 10.0) -> float:
    cap = cv2.VideoCapture(str(video_path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
    finally:
        cap.release()
    if fps is None or fps <= 0:
        return default_fps
    return float(fps)

def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def build_config_from_args(args) -> BatchRunConfig:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir

    return BatchRunConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        suffix=args.suffix,
        alpha=args.alpha,
        fps_out=args.fps_out,
        skip_existing=args.skip_existing,
        prompt_frame=args.prompt_frame,
        gpu_ids=args.gpu_ids,
    )

def resolve_prompts(args) -> List[str]:
    if args.prompt is not None:
        return [args.prompt]
    return list(args.prompts)

def make_output_path(config: BatchRunConfig, video_path: Path, prompt: str) -> Path:
    pslug = slug_prompt(prompt)
    return config.output_dir / f"{video_path.stem}{config.suffix}__{pslug}.mp4"


def build_video_prompt_jobs(
    videos: List[Path],
    prompts: List[str],
    config: BatchRunConfig,
) -> List[VideoPromptJob]:
    jobs: List[VideoPromptJob] = []

    for video_path in videos:
        for prompt in prompts:
            jobs.append(
                VideoPromptJob(
                    video_path=video_path,
                    prompt=prompt,
                    output_path=make_output_path(config, video_path, prompt),
                )
            )

    return jobs


def group_jobs_by_video(
    jobs: List[VideoPromptJob],
) -> Dict[Path, List[VideoPromptJob]]:
    grouped: Dict[Path, List[VideoPromptJob]] = {}

    for job in jobs:
        grouped.setdefault(job.video_path, []).append(job)

    return grouped

def is_job_skippable(job: VideoPromptJob, config: BatchRunConfig) -> bool:
    """
    skip 여부 판단.
    """
    return config.skip_existing and job.output_path.exists()


def filter_runnable_jobs(
    jobs: List[VideoPromptJob],
    config: BatchRunConfig,
) -> tuple[List[VideoPromptJob], List[PromptRunResult]]:
    """
    실행할 job과 skip할 job을 분리.
    """
    runnable_jobs: List[VideoPromptJob] = []
    skipped_results: List[PromptRunResult] = []

    for job in jobs:
        if is_job_skippable(job, config):
            skipped_results.append(PromptRunResult.skipped(job))
        else:
            runnable_jobs.append(job)

    return runnable_jobs, skipped_results

class Sam3VideoSession:
    """
    SAM3 predictor 세션의 start, reset, close 책임을 담당.
    """

    def __init__(self, predictor, video_path: Path):
        self.predictor = predictor
        self.video_path = video_path
        self.session_id: Optional[str] = None

    def __enter__(self) -> "Sam3VideoSession":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        cleanup_cuda()

        return False

    def start(self) -> None:
        response = self.predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=str(self.video_path),
            )
        )
        self.session_id = response["session_id"]

    def reset(self) -> None:
        if self.session_id is None:
            self.start()
            return

        try:
            self.predictor.handle_request(
                request=dict(
                    type="reset_session",
                    session_id=self.session_id,
                )
            )
        except Exception:
            self.close()
            self.start()
            cleanup_cuda()

    def close(self) -> None:
        if self.session_id is None:
            return

        try:
            self.predictor.handle_request(
                request=dict(
                    type="close_session",
                    session_id=self.session_id,
                )
            )
        finally:
            self.session_id = None

    def require_session_id(self) -> str:
        if self.session_id is None:
            raise RuntimeError("SAM3 session is not started.")
        return self.session_id
    
class RunReporter:
    """
    진행 상황 출력과 최종 실패 보고를 담당.
    """

    def on_start(self, videos: List[Path], jobs: List[VideoPromptJob]) -> None:
        print(f"[INFO] Found {len(videos)} video file(s).")
        print(f"[INFO] Planned {len(jobs)} video-prompt job(s).")

    def on_video_start(self, video_path: Path, num_jobs: int) -> None:
        print(f"\n[VIDEO] Processing {video_path.name} ({num_jobs} prompt job(s))")

    def on_result(self, result: PromptRunResult) -> None:
        metric_text = ""

        if result.elapsed_sec is not None:
            metric_text += f" | time={result.elapsed_sec:.3f}s"

        if result.peak_cpu_memory_mb is not None:
            metric_text += f" | cpu_peak={result.peak_cpu_memory_mb:.1f}MB"

        if result.peak_cuda_memory_mb is not None:
            metric_text += f" | cuda_peak={result.peak_cuda_memory_mb:.1f}MB"

        if result.status == "done":
            print(
                f"[DONE] {result.video_name} + '{result.prompt}' "
                f"-> {result.output_path.name}{metric_text}"
            )
        elif result.status == "skipped":
            print(
                f"[SKIP] {result.video_name} + '{result.prompt}' "
                f"-> {result.output_path.name} (already exists)"
            )
        else:
            print(
                f"\n[ERROR] {result.video_name} + '{result.prompt}'{metric_text}\n"
                f"{result.error}\n"
            )

    def on_finish(self, results: List[PromptRunResult]) -> None:
        print("\n[DONE] All video files processed.")

        failed_results = [r for r in results if r.status == "failed"]

        if not failed_results:
            print("[REPORT] No failures to report.")
            return

        print("\n" + "=" * 60)
        print("[REPORT] FAILED COMBINATIONS")
        print("=" * 60)

        for result in failed_results:
            print(f"- Video: {result.video_name} + Prompt: '{result.prompt}'")
            print(f"  Error: {result.error}")

        print("=" * 60 + "\n")

def save_results_csv(results: List[PromptRunResult], csv_path: Path) -> None:
    """
    PromptRunResult 목록을 CSV로 저장.
    보고서의 실행 시간 / 메모리 사용량 표 작성에 사용.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "video_name",
        "prompt",
        "output_path",
        "status",
        "elapsed_sec",
        "peak_cpu_memory_mb",
        "peak_cuda_memory_mb",
        "error",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            writer.writerow(
                {
                    "video_name": result.video_name,
                    "prompt": result.prompt,
                    "output_path": str(result.output_path),
                    "status": result.status,
                    "elapsed_sec": result.elapsed_sec,
                    "peak_cpu_memory_mb": result.peak_cpu_memory_mb,
                    "peak_cuda_memory_mb": result.peak_cuda_memory_mb,
                    "error": result.error,
                }
            )

def iter_propagated_outputs(predictor, session_id: str):
    """
    SAM3 predictor의 handle_stream_request() 결과를 frame 단위로 반환.
    """
    for response in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
            propagation_direction="forward",
        )
    ):
        yield response["frame_index"], response["outputs"]


def measure_prompt_run(func):
    """
    단일 prompt job 실행 시간과 메모리 사용량을 측정하는 decorator.

    측정 대상:
    - elapsed_sec: wall-clock 실행 시간
    - peak_cpu_memory_mb: tracemalloc 기준 Python-level peak memory
    - peak_cuda_memory_mb: torch.cuda 기준 GPU peak allocated memory
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        was_tracing = tracemalloc.is_tracing()

        if not was_tracing:
            tracemalloc.start()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        start_time = time.perf_counter()

        try:
            result = func(*args, **kwargs)
            return result

        finally:
            elapsed_sec = time.perf_counter() - start_time

            current_bytes, peak_bytes = tracemalloc.get_traced_memory()
            peak_cpu_memory_mb = peak_bytes / (1024 * 1024)

            if not was_tracing:
                tracemalloc.stop()

            if torch.cuda.is_available():
                peak_cuda_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            else:
                peak_cuda_memory_mb = None

            # func가 정상적으로 PromptRunResult를 반환한 경우에만 metric을 붙입니다.
            # try-finally 구조상 return 직전 result를 직접 수정하기 위해 locals()를 사용합니다.
            if "result" in locals() and isinstance(result, PromptRunResult):
                result.attach_metrics(
                    elapsed_sec=elapsed_sec,
                    peak_cpu_memory_mb=peak_cpu_memory_mb,
                    peak_cuda_memory_mb=peak_cuda_memory_mb,
                )

    return wrapper


def save_masklet_video_streaming(
    video_path: Path,
    output_stream,
    out_path: Path,
    alpha: float = 0.5,
    fps: float = 10,
) -> None:
    """
    기존 SAM3 save_masklet_video()와 유사하게 동작하되,
    전체 frame list와 전체 outputs dict를 만들지 않는 streaming 저장 함수.
    """

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    temp_file = NamedTemporaryFile(
        suffix=".mp4",
        delete=False,
        dir=str(out_path.parent),
    )
    temp_path = temp_file.name
    temp_file.close()

    writer = None
    progress = None

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if width <= 0 or height <= 0:
            raise RuntimeError(f"Invalid video size: {video_path}")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))

        if not writer.isOpened():
            raise RuntimeError(f"Cannot open video writer: {temp_path}")

        output_iter = iter(output_stream)

        try:
            next_frame_idx, next_outputs = next(output_iter)
        except StopIteration:
            next_frame_idx, next_outputs = None, None

        frame_idx = 0
        progress = tqdm(desc=f"Saving {out_path.name}", unit="frame")

        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            img = load_frame(frame_rgb)

            frame_outputs = None

            # output stream이 현재 frame보다 뒤처져 있으면 따라잡기.
            while next_frame_idx is not None and next_frame_idx < frame_idx:
                try:
                    next_frame_idx, next_outputs = next(output_iter)
                except StopIteration:
                    next_frame_idx, next_outputs = None, None
                    break

            # 현재 frame에 해당하는 SAM3 output이 있으면 기존 SAM3 렌더링 함수를 사용.
            if next_frame_idx == frame_idx:
                frame_outputs = next_outputs

                try:
                    next_frame_idx, next_outputs = next(output_iter)
                except StopIteration:
                    next_frame_idx, next_outputs = None, None

            if frame_outputs is not None:
                overlay = render_masklet_frame(
                    img,
                    frame_outputs,
                    frame_idx=frame_idx,
                    alpha=alpha,
                )
            else:
                # output이 없는 frame은 원본 frame을 그대로 저장.
                overlay = img

            writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

            frame_idx += 1
            progress.update(1)

    finally:
        cap.release()

        if writer is not None:
            writer.release()

        if progress is not None:
            progress.close()

    try:
        # 기존 SAM3 함수처럼 ffmpeg로 재인코딩.
        subprocess.run(
            ["ffmpeg", "-y", "-i", temp_path, str(out_path)],
            check=True,
        )

        print(f"Re-encoded video saved to {out_path}")
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass

@measure_prompt_run
def run_single_prompt_job(
    predictor,
    job: VideoPromptJob,
    session: Sam3VideoSession,
    fps_out: float,
    config: BatchRunConfig,
) -> PromptRunResult:
    """
    하나의 video-prompt job을 실행.
    단일 프롬프트에 대한 add_prompt, propagate, save만 담당함.
    """

    try:
        session.reset()
        session_id = session.require_session_id()

        with torch.inference_mode():
            predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=config.prompt_frame,
                    text=job.prompt,
                )
            )

            output_stream = iter_propagated_outputs(
                predictor=predictor,
                session_id=session_id,
            )

            save_masklet_video_streaming(
                video_path=job.video_path,
                output_stream=output_stream,
                out_path=job.output_path,
                alpha=config.alpha,
                fps=fps_out,
            )

        return PromptRunResult.done(job)

    except torch.OutOfMemoryError as e:
        session.close()
        cleanup_cuda()
        return PromptRunResult.failed(job, e)

    except Exception as e:
        session.close()
        cleanup_cuda()
        return PromptRunResult.failed(job, e)

    finally:
        cleanup_cuda()


def run_video_jobs(
    predictor,
    video_path: Path,
    jobs: List[VideoPromptJob],
    config: BatchRunConfig,
) -> List[PromptRunResult]:
    """
    하나의 영상에 속한 여러 prompt job을 실행.
    기존 run_video_with_prompts()가 하던 비디오 단위 처리를 대체함.
    """

    if not jobs:
        return []

    runnable_jobs, skipped_results = filter_runnable_jobs(
        jobs=jobs,
        config=config,
    )

    if not runnable_jobs:
        return skipped_results

    if config.fps_out is None:
        resolved_fps_out = get_video_fps(video_path, default_fps=10.0)
    else:
        resolved_fps_out = config.fps_out

    results: List[PromptRunResult] = []
    results.extend(skipped_results)

    with Sam3VideoSession(predictor, video_path) as session:
        for job in runnable_jobs:
            result = run_single_prompt_job(
                predictor=predictor,
                job=job,
                session=session,
                fps_out=resolved_fps_out,
                config=config,
            )
            results.append(result)

    cleanup_cuda()

    return results

def main() -> int:
    args = parse_args()

    config = build_config_from_args(args)
    prompts = resolve_prompts(args)

    if config.gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_ids
        print(f"[GPU] using CUDA_VISIBLE_DEVICES={config.gpu_ids}")

    config.output_dir.mkdir(parents=True, exist_ok=True)

    videos = list_videos(config.input_dir)

    if not videos:
        print(f"No .mp4 files found in: {config.input_dir}")
        return 0

    jobs = build_video_prompt_jobs(
        videos=videos,
        prompts=prompts,
        config=config,
    )

    jobs_by_video = group_jobs_by_video(jobs)

    reporter = RunReporter()
    reporter.on_start(videos, jobs)

    gpus_to_use = list(range(torch.cuda.device_count()))
    print(f"[GPU] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"[GPU] using visible cuda device indices: {gpus_to_use}")

    predictor = None
    all_results: List[PromptRunResult] = []

    try:
        print("Loading SAM3 predictor...")
        predictor = build_sam3_video_predictor(gpus_to_use=gpus_to_use)

        for video_path in videos:
            video_jobs = jobs_by_video.get(video_path, [])
            reporter.on_video_start(video_path, len(video_jobs))

            video_results = run_video_jobs(
                predictor=predictor,
                video_path=video_path,
                jobs=video_jobs,
                config=config,
            )

            for result in video_results:
                reporter.on_result(result)

            all_results.extend(video_results)

    finally:
        if predictor is not None:
            try:
                predictor.shutdown()
            except Exception:
                pass

        cleanup_cuda()

    reporter.on_finish(all_results)

    result_csv_path = config.output_dir / "prompt_run_metrics.csv"
    save_results_csv(all_results, result_csv_path)
    print(f"[REPORT] Benchmark results saved to: {result_csv_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())