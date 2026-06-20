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
from pathlib import Path
from typing import List, Optional

import cv2
import torch
from sam3.model_builder import build_sam3_video_predictor
from sam3.visualization_utils import save_masklet_video

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

if __name__ ==  "__main__":
    # 환경 변수 세팅
    args = parse_args()

    if args.gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        print(f"[GPU] using CUDA_VISIBLE_DEVICES={args.gpu_ids}")

    # Helper
    def slug_prompt(prompt: str, max_len: int = 40) -> str:
        s = prompt.strip().lower()
        s = re.sub(r"\s+", "_", s)                 # 공백 -> _
        s = re.sub(r"[^0-9a-zA-Z가-힣_+-]+", "", s) # 파일명 위험 문자 제거
        return s[:max_len] if len(s) > max_len else s


    def list_videos(folder: str):
        p = Path(folder)
        if not p.exists():
            raise FileNotFoundError(f"input_dir does not exist: {p}")
        if not p.is_dir():
            raise NotADirectoryError(f"input_dir is not a directory: {p}")
        return [f for f in sorted(p.iterdir()) if f.is_file() and f.suffix.lower() == ".mp4"]

    def read_video_frames_rgb(video_path: str):
        """mp4 비디오 -> 프레임 리스트(RGB)"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        
        video_frames_for_vis = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            video_frames_for_vis.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return video_frames_for_vis

    def get_video_fps(video_path: str, default_fps: float = 10.0) -> float:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps is None or fps <= 0:
            return default_fps
        return float(fps)

    def cleanup_cuda():
        gc.collect()
        torch.cuda.empty_cache()

    def run_video_with_prompts(
            predictor,
            video_path: Path,
            prompts: List[str],
            out_root: Path,
            suffix: str,
            fps_out: Optional[float],
            skip_existing: bool,
            alpha: float = 0.5,
            frame_index_for_prompt: int = 0,
    ) -> List[str]:
        
        failed_prompts = []
        # 프레임 로드
        video_frames_for_vis = read_video_frames_rgb(str(video_path))

        if fps_out is None:
            fps_out = get_video_fps(str(video_path), default_fps=10.0)

        session_id = None

        try:
            for prompt in prompts:
                pslug = slug_prompt(prompt)
                out_path = out_root / f"{video_path.stem}{suffix}__{pslug}.mp4"

                if skip_existing and out_path.exists():
                    print(f"[SKIP] {video_path.name} + '{prompt}' -> {out_path.name} (already exists)")
                    continue

                if session_id is None:
                    # 세션 시작
                    response = predictor.handle_request(
                        request=dict(
                            type="start_session",
                            resource_path=str(video_path),
                        )
                    )
                    session_id = response["session_id"]
                else:
                    try:
                        predictor.handle_request(
                            request=dict(
                                type="reset_session",
                                session_id=session_id,
                            )
                        )
                    except Exception:
                        # reset이 안 될 땐 세션 닫고 재시작
                        predictor.handle_request(
                            request=dict(
                                type="close_session",
                                session_id=session_id,
                            )
                        )
                        response = predictor.handle_request(
                            request=dict(
                                type="start_session",
                                resource_path=str(video_path),
                            )
                        )
                        session_id = response["session_id"]
                        cleanup_cuda()
                
                try:
                    # 텍스트 프롬프트 추가
                    response = predictor.handle_request(
                        request=dict(
                            type="add_prompt",
                            session_id=session_id,
                            frame_index=frame_index_for_prompt,
                            text=prompt,
                        )
                    )
                    # 비디오로 예측 전파하고 수집
                    outputs_per_frame = {}
                    for r in predictor.handle_stream_request(
                        request=dict(
                            type="propagate_in_video",
                            session_id=session_id,
                            propagation_direction="forward",
                        )
                    ):
                        outputs_per_frame[r["frame_index"]] = r["outputs"]

                    # 비디오 저장
                    save_masklet_video(
                        video_frames_for_vis,
                        outputs_per_frame,
                        out_path=str(out_path),
                        fps=fps_out,
                        alpha=alpha,
                    )

                    print(f"[DONE] {video_path.name} + '{prompt}' -> {out_path.name}")
                except torch.OutOfMemoryError as e:
                    print(f"\n[OOM ERROR] {video_path.name} + '{prompt}'\n")
                    predictor.handle_request(request=dict(type="close_session", session_id=session_id))
                    session_id = None
                    failed_prompts.append(prompt)
                except Exception as e:
                    print(f"\n[ERROR] {video_path.name} + '{prompt}'\n{str(e)}\n")
                    predictor.handle_request(request=dict(type="close_session", session_id=session_id))
                    session_id = None
                    failed_prompts.append(prompt)
                finally:
                    if 'outputs_per_frame' in locals():
                        del outputs_per_frame
                    cleanup_cuda()

        finally:
            if session_id is not None:
                # 세션 종료 + 메모리 정리
                predictor.handle_request(
                    request=dict(
                        type="close_session",
                        session_id=session_id,
                    )
                )
            cleanup_cuda()

        return failed_prompts

    # Main
    input_dir = Path(args.input_dir)
    out_root = Path(args.output_dir) if args.output_dir else input_dir
    out_root.mkdir(parents=True, exist_ok=True)

    if args.prompt is not None:
        prompts = [args.prompt]
    else:
        prompts = list(args.prompts)

    videos = list_videos(input_dir)
    if not videos:
        print(f"No .mp4 files found in: {input_dir}")
        sys.exit(0)
    
    gpus_to_use = list(range(torch.cuda.device_count()))
    print(f"[GPU] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"[GPU] using visible cuda device indices: {gpus_to_use}")

    try:
        print("Loading SAM3 predictor...")
        predictor = build_sam3_video_predictor(gpus_to_use=gpus_to_use)

        failed_combis = []

        for vp in videos:
            print(f"\n[VIDEO] Processing {vp.name}")
            failed_prompts = run_video_with_prompts(
                predictor=predictor,
                video_path=vp,
                prompts=prompts,
                out_root=out_root,
                suffix=args.suffix,
                fps_out=args.fps_out,
                skip_existing=args.skip_existing,
                alpha=args.alpha,
                frame_index_for_prompt=args.prompt_frame,
            )

            for fp in failed_prompts:
                failed_combis.append((vp.name, fp))
        
        print("\n[DONE] All video files processed.")

        if failed_combis:
            print("\n" + "="*60)
            print("[REPORT] FAILED COMBINATIONS]")
            print("="*60)
            for v_name, p in failed_combis:
                print(f"- Video: {v_name} + Prompt: '{p}'")
            print("="*60 + "\n")
        else:
            print("[REPORT] No failures to report.")
    finally:
        if predictor is not None:
            try:
                predictor.shutdown()
            except Exception:
                pass
        cleanup_cuda()