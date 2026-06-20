# SAM3 배치 비디오 세그멘테이션 코드 최적화 프로젝트

본 저장소는 **고급파이썬프로그래밍및자동화** 최종 과제를 위해 작성한 프로젝트입니다.
기존에 사용하던 SAM3 기반 배치 비디오 세그멘테이션 코드를 대상으로, 수업에서 학습한 Python 구조화 기법, 클래스 기반 설계, 데이터클래스, 단일 책임 원칙, 제너레이터/스트리밍 처리, 데코레이터 기반 측정, 벤치마크 코드를 적용하여 코드 구조와 실행 특성을 개선했습니다.

## 1. 프로젝트 목적

기존 코드는 입력 폴더 안의 여러 `.mp4` 파일을 대상으로 하나 이상의 텍스트 프롬프트를 적용하고, SAM3를 통해 비디오 세그멘테이션 결과 영상을 저장하는 배치 실행 스크립트입니다.

본 프로젝트의 목적은 다음과 같습니다.

* 최적화 전 코드와 최적화 후 코드를 비교 가능하게 정리
* SAM3 비디오 세그멘테이션 배치 실행 과정의 구조 개선
* 설정값, 실행 단위, 실행 결과, 세션 관리, 출력 보고 책임 분리
* 메모리 사용량을 줄이기 위한 스트리밍 기반 결과 저장 방식 적용
* 동일 조건에서 before/after 코드를 실행하는 벤치마크 코드 작성
* 실행 시간, CPU 메모리, GPU 메모리 사용량을 CSV로 기록

## 2. 저장소 구조

```text
ADVANCEDPYTHON/
├─ benchmark/
│  └─ run_benchmark.py
├─ data/
│  └─ sample_inputs/
│     ├─ sample_01.mp4
│     ├─ sample_02.mp4
│     ├─ sample_03.mp4
│     └─ sample_04.mp4
├─ report/
│  └─ report.pdf
├─ results/
│  └─ benchmark_results.csv
├─ src/
│  ├─ before/
│  │  └─ run_sam3_batch.py
│  └─ after/
│     ├─ run_sam3_batch_after.py
│     ├─ run_sam3_batch_afterC.py
│     └─ run_sam3_batch_afterCB.py
├─ .gitignore
├─ requirements.txt
└─ README.md
```

## 3. 포함하지 않는 파일

본 저장소에는 다음 항목을 포함하지 않습니다.

* Meta SAM3 원본 저장소
* SAM3 checkpoint 파일
* 비공개 또는 사용 제한이 있는 원본 연구 데이터
* 실행 결과로 생성된 출력 비디오
* 벤치마크 실행 중 생성되는 반복 output 폴더
* 실행 로그 파일

SAM3 원본 코드는 별도로 clone하여 설치해야 합니다.

## 4. 데이터 안내

기존 실험에는 공개가 제한된 cooking video dataset을 사용했으나, 해당 데이터는 라이선스 및 이용 조건 문제로 본 저장소에 포함하지 않았습니다.

대신 본 저장소에는 공개 가능한 소규모 샘플 `.mp4` 파일을 포함하여, 코드 실행 및 벤치마크를 재현할 수 있도록 구성했습니다.

샘플 입력 영상은 다음 경로에 위치합니다.

```text
data/sample_inputs/
```

사용자는 필요에 따라 이 폴더의 영상을 다른 `.mp4` 파일로 교체하여 실행할 수 있습니다.

## 5. 실행 환경 준비

### 5.1 SAM3 설치

본 저장소는 SAM3 원본 코드를 포함하지 않습니다.
먼저 SAM3를 별도로 clone하고 editable 모드로 설치합니다.

```bash
git clone https://github.com/facebookresearch/sam3.git external/sam3
pip install -e external/sam3
```

만약 SAM3를 본 프로젝트 외부에 clone하고 싶다면 다음과 같이 설치해도 됩니다.

```bash
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
```

### 5.2 PyTorch 설치

PyTorch는 CUDA 버전에 따라 설치 명령이 달라질 수 있으므로, 본 저장소의 `requirements.txt`에는 고정하지 않았습니다.

예를 들어 CUDA 12.1 환경에서는 다음과 같이 설치할 수 있습니다.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

사용 중인 CUDA 버전에 맞는 설치 명령은 PyTorch 공식 설치 안내를 따르십시오.

### 5.3 기타 의존성 설치

```bash
pip install -r requirements.txt
```

`requirements.txt`에는 다음과 같은 보조 패키지가 포함됩니다.

```text
opencv-python
psutil
tqdm
numpy
```

## 6. 최적화 전 코드 실행

최적화 전 코드는 `src/before/run_sam3_batch.py`에 있습니다.

```bash
python src/before/run_sam3_batch.py \
  --input_dir data/sample_inputs \
  --output_dir outputs/before \
  --prompts "tomato" "egg" "knife" "pan" \
  --gpu_ids "0"
```

단일 프롬프트만 사용할 경우 다음과 같이 실행할 수 있습니다.

```bash
python src/before/run_sam3_batch.py \
  --input_dir data/sample_inputs \
  --output_dir outputs/before \
  --prompt "tomato" \
  --gpu_ids "0"
```

## 7. 최적화 후 코드 실행

최종 개선 버전은 `src/after/run_sam3_batch_afterCB.py`입니다.

```bash
python src/after/run_sam3_batch_afterCB.py \
  --input_dir data/sample_inputs \
  --output_dir outputs/after \
  --prompts "tomato" "egg" "knife" "pan" \
  --gpu_ids "0"
```

이미 생성된 결과를 건너뛰고 실행하려면 `--skip_existing` 옵션을 추가합니다.

```bash
python src/after/run_sam3_batch_afterCB.py \
  --input_dir data/sample_inputs \
  --output_dir outputs/after \
  --prompts "tomato" "egg" "knife" "pan" \
  --gpu_ids "0" \
  --skip_existing
```

## 8. 출력 파일 형식

결과 비디오는 다음 규칙으로 저장됩니다.

```text
<원본파일명><suffix>__<prompt>.mp4
```

기본 suffix는 `_sam3`입니다.

예를 들어 입력 파일이 `sample_01.mp4`이고 프롬프트가 `tomato`라면 결과 파일명은 다음과 같습니다.

```text
sample_01_sam3__tomato.mp4
```

## 9. 벤치마크 실행

before/after 코드를 동일 조건에서 순차 실행하고, 실행 시간과 메모리 사용량을 측정하려면 다음 명령을 사용합니다.

```bash
python benchmark/run_benchmark.py \
  --before_script src/before/run_sam3_batch.py \
  --after_script src/after/run_sam3_batch_afterCB.py \
  --input_dir data/sample_inputs \
  --results_csv results/benchmark_results.csv \
  --output_root results/benchmark_outputs \
  --prompts "tomato" "egg" "knife" "pan" \
  --repeats 3 \
  --gpu_id 0
```

벤치마크는 다음 항목을 기록합니다.

* version: before 또는 after
* repeat: 반복 번호
* elapsed_sec: 전체 실행 시간
* peak_rss_mb: 프로세스 트리 기준 CPU peak RSS 메모리
* peak_gpu_memory_mb: `nvidia-smi` 기준 GPU peak memory

결과 CSV는 다음 위치에 저장됩니다.

```text
results/benchmark_results.csv
```

`results/benchmark_results.csv`는 보고서 작성 및 before/after 비교를 위해 GitHub에 포함합니다.
반면 벤치마크 실행 중 생성되는 출력 비디오와 로그는 GitHub에 포함하지 않습니다.

## 10. 주요 개선 내용

### 10.1 설정값과 실행 단위 분리

기존 코드는 argparse 결과, 입력 영상, 프롬프트, 출력 경로, 실행 상태가 하나의 흐름 안에 섞여 있었습니다.
개선 코드에서는 `BatchRunConfig`, `VideoPromptJob`, `PromptRunResult`와 같은 데이터클래스를 사용하여 설정값, 실행 단위, 실행 결과를 명확히 분리했습니다.

### 10.2 세션 관리 책임 분리

SAM3 predictor의 `start_session`, `reset_session`, `close_session` 호출을 별도 세션 관리 객체로 분리했습니다.
이를 통해 비디오별 세션 생명주기를 명확하게 관리하고, 예외 발생 시에도 세션과 CUDA 캐시 정리가 수행되도록 했습니다.

### 10.3 진행 상황 출력과 실패 보고 분리

기존 코드는 처리 로직 내부에서 직접 출력과 실패 보고를 수행했습니다.
개선 코드에서는 `RunReporter`를 통해 진행 상황 출력과 최종 실패 보고를 분리했습니다.

### 10.4 스트리밍 기반 결과 저장

기존 방식은 전체 비디오 프레임을 리스트로 메모리에 올리고, 프레임별 segmentation output도 딕셔너리에 모은 뒤 영상을 저장했습니다.
개선 버전에서는 가능한 범위에서 프레임과 output을 순차 처리하는 스트리밍 저장 방식을 적용하여 peak memory 사용량을 줄이는 방향으로 구조를 개선했습니다.

### 10.5 벤치마크 코드 분리

before/after 스크립트를 동일 조건으로 순차 실행하는 `benchmark/run_benchmark.py`를 작성했습니다.
이를 통해 단순한 체감 속도 비교가 아니라, 실행 시간과 메모리 사용량을 CSV로 기록하여 보고서에 정량적으로 반영할 수 있도록 했습니다.

## 11. 재현성 관련 주의사항

벤치마크 결과는 다음 조건에 영향을 받을 수 있습니다.

* 사용 GPU 종류와 VRAM 크기
* CUDA 및 PyTorch 버전
* SAM3 checkpoint
* 입력 영상의 길이, 해상도, FPS
* 프롬프트 개수
* 백그라운드 프로세스의 GPU 메모리 사용량
* warm-up 여부

따라서 before/after 비교는 반드시 같은 입력 영상, 같은 프롬프트, 같은 GPU, 같은 실행 환경에서 수행해야 합니다.

## 12. GitHub 포함/제외 기준

본 저장소에 포함하는 항목은 다음과 같습니다.

```text
포함
- src/before/run_sam3_batch.py
- src/after/*.py
- benchmark/run_benchmark.py
- data/sample_inputs/*.mp4
- results/benchmark_results.csv
- report/report.pdf
- README.md
- requirements.txt
```

본 저장소에서 제외하는 항목은 다음과 같습니다.

```text
제외
- sam3/
- external/sam3/
- checkpoints/
- outputs/
- results/benchmark_outputs/
- *.log
- *.pt, *.pth, *.ckpt, *.safetensors
```

## 13. 참고 실행 예시

최종 개선 코드만 빠르게 실행하려면 다음 명령을 사용합니다.

```bash
python src/after/run_sam3_batch_afterCB.py \
  --input_dir data/sample_inputs \
  --output_dir outputs/after \
  --prompts "tomato" "egg" "knife" "pan" \
  --gpu_ids "0"
```

벤치마크까지 수행하려면 다음 명령을 사용합니다.

```bash
python benchmark/run_benchmark.py \
  --before_script src/before/run_sam3_batch.py \
  --after_script src/after/run_sam3_batch_afterCB.py \
  --input_dir data/sample_inputs \
  --results_csv results/benchmark_results.csv \
  --output_root results/benchmark_outputs \
  --prompts "tomato" "egg" "knife" "pan" \
  --repeats 3 \
  --gpu_id 0
```
