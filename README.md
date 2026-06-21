# SAM3 배치 비디오 세그멘테이션 코드 최적화 프로젝트

본 저장소는 **고급파이썬프로그래밍및자동화** 최종 과제를 위해 작성한 프로젝트다.

기존에 사용하던 SAM3 기반 배치 비디오 세그멘테이션 코드를 대상으로, 수업에서 학습한 Python 구조화 기법, 클래스 기반 설계, 데이터클래스, 단일 책임 원칙, 제너레이터/스트리밍 처리, 데코레이터 기반 측정, 벤치마크 코드를 적용하여 코드 구조와 실행 특성을 개선했다.

기존 코드와 개선 코드를 동일 조건에서 실행할 수 있는 벤치마크 코드를 구성했으며, 실행 시간, CPU 메모리 사용량, GPU 메모리 사용량을 벤치마크 CSV로 기록할 수 있다.

## 1. 저장소 구조

```text
├─ benchmark/
│  └─ run_benchmark.py
├─ data/
│  └─ inputs/
│     ├─ FHD10/
│     │  ├─ input1.mp4
│     │  ├─ input2.mp4
│     │  ├─ input3.mp4
│     │  └─ input4.mp4
│     ├─ FHD15/
│     ├─ FHD20/
│     ├─ FHD25/
│     └─ HD10/
├─ report/
├─ results/
│  └─ HD10/
│     ├─ benchmark_outputs/
│     │  ├─ after/
│     │  ├─ before/
│     └─ benchmark_results.csv
├─ src/
│  ├─ before/
│  │  └─ run_sam3_batch.py
│  └─ after/
│     ├─ run_sam3_batch_after.py
│     ├─ run_sam3_batch_afterC.py
│     └─ run_sam3_batch_afterCB.py
├─ .gitignore
├─ environment.yml
└─ README.md
```

주요 파일의 역할은 다음과 같다.

```text
benchmark/run_benchmark.py
- before/after 코드를 동일 조건에서 순차 실행하고 실행 시간 및 메모리 사용량을 측정한다.

src/before/run_sam3_batch.py
- 최적화 전 SAM3 배치 비디오 세그멘테이션 실행 코드다.

src/after/run_sam3_batch_after.py
- 최종 개선 버전 실행 코드다.

environment.yml
- conda 가상환경 재현을 위한 실행 환경 파일이다.

results/
- 벤치마크 결과 CSV 및 실행 결과가 저장되는 폴더다. 하위의 HD10 폴더에는 실제 보고서 작성에 쓰인 실험 수행 결과 파일들이 존재한다.
```

본 저장소에는 다음 항목을 포함하지 않는다.

```text
- Meta SAM3 원본 저장소
- SAM3 checkpoint 파일
- 비공개 또는 사용 제한이 있는 원본 연구 데이터
- 실행 결과로 생성되는 출력 비디오
- 벤치마크 실행 중 생성되는 반복 output 폴더
- 실행 로그 파일
```

SAM3 원본 코드는 별도로 clone하여 설치해야 한다.

## 2. 데이터 안내

기존 실험에는 공개가 제한된 cooking video dataset을 사용했으나, 해당 데이터는 라이선스 및 이용 조건 문제로 본 저장소에 포함하지 않았다.

대신 본 저장소에는 pixabay에서 수집한 공개 가능한 소규모 샘플 `.mp4` 파일을 포함하여, 코드 실행 및 벤치마크를 재현할 수 있도록 구성했다.

입력 영상은 다음 경로에 위치한다.

```text
data/inputs/
```

현재 입력 데이터는 해상도와 fps 조건에 따라 다음과 같이 구분되어 있다.

```text
data/inputs/FHD10/
data/inputs/FHD15/
data/inputs/FHD20/
data/inputs/FHD25/
data/inputs/HD10/
```

예를 들어 `FHD10` 조건의 입력 영상은 다음 경로에 위치한다.

```text
data/inputs/FHD10/
```

사용자는 필요에 따라 각 폴더의 영상을 다른 `.mp4` 파일로 교체하여 실행할 수 있다.

결과 비디오는 기본적으로 다음 규칙으로 저장된다.

```text
<원본파일명><suffix>__<prompt>.mp4
```

기본 suffix는 `_sam3`이다.

예를 들어 입력 파일이 `input1.mp4`이고 프롬프트가 `broccoli`라면 결과 파일명은 다음과 같다.

```text
input1_sam3__broccoli.mp4
```

## 3. 실행 환경 준비

본 프로젝트는 conda 가상환경에서 실행하는 것을 권장한다.
프로젝트를 clone한 뒤, 제공된 `environment.yml` 파일을 이용해 실행 환경을 구성한다.

### 3.1 프로젝트 clone

먼저 본 프로젝트 저장소를 clone한다.

```bash
git clone https://github.com/Nua3196/AdvancedPython_202551131.git
cd AdvancedPython_202551131
```

이후 명령은 기본적으로 본 프로젝트 폴더인 `AdvancedPython_202551131`에서 실행한다.

### 3.2 conda 가상환경 생성

프로젝트에 포함된 `environment.yml` 파일을 사용하여 가상환경을 생성한다.

기본적으로는 `environment.yml` 파일 안의 `name:` 항목에 정의된 이름으로 가상환경이 생성된다.

```bash
conda env create -f environment.yml
```

생성이 완료되면 해당 환경을 활성화한다.

```bash
conda activate <environment.yml의 name 값>
```

원하는 이름으로 가상환경을 만들고 싶다면 `-n` 옵션을 사용할 수 있다.

```bash
conda env create -f environment.yml -n <원하는_환경이름>
conda activate <원하는_환경이름>
```

예를 들어 `sam3_hw`라는 이름으로 환경을 만들고 싶다면 다음과 같이 실행한다.

```bash
conda env create -f environment.yml -n sam3_hw
conda activate sam3_hw
```

PyTorch CUDA wheel은 일반 PyPI가 아니라 PyTorch 전용 index에서 설치해야 하므로, `environment.yml` 생성 후 별도로 설치한다.

```bash
python -m pip install torch==2.7.0+cu126 torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu126
```

### 3.3 SAM3 설치

본 프로젝트는 SAM3 원본 코드를 저장소에 포함하지 않는다.
따라서 본 프로젝트를 받은 상위 폴더 안에 `sam3` 저장소를 별도로 clone한 뒤, 현재 활성화된 conda 환경에 editable mode로 설치한다.

권장 폴더 구조는 다음과 같다.

```text
상위폴더/
├─ AdvancedPython_202551131/
└─ sam3/
```

먼저 본 프로젝트 폴더의 상위 폴더로 이동한다.

```bash
cd ..
```

그다음 SAM3 저장소를 clone한다.

```bash
git clone https://github.com/facebookresearch/sam3.git sam3
```

SAM3 폴더로 이동하여 현재 활성화된 conda 환경에 설치한다.

```bash
cd sam3
pip install -e .
```

이때 반드시 앞에서 생성한 conda 가상환경이 활성화된 상태여야 한다.

설치가 완료되면 다시 본 프로젝트 폴더로 돌아온다.

```bash
cd ../AdvancedPython_202551131
```

### 3.4 설치 확인

현재 활성화된 가상환경에서 주요 패키지가 정상적으로 import되는지 확인한다.

```bash
python -c "import torch; print(torch.__version__)"
python -c "import cv2; print(cv2.__version__)"
```

SAM3가 정상적으로 설치되었는지도 확인한다.

```bash
pip show sam3
```

CUDA 사용 가능 여부는 다음 명령으로 확인할 수 있다.

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

단, `environment.yml`은 conda 환경 안의 패키지 정보를 재현하기 위한 파일이다.
시스템에 설치된 NVIDIA driver나 GPU 자체 설정까지 포함하지는 않으므로, CUDA를 사용하려면 실행 서버에 적절한 NVIDIA driver가 설치되어 있어야 한다.

## 4. 실행 방법

### 4.1 최적화 전 코드 실행

최적화 전 코드는 `src/before/run_sam3_batch.py`에 있다.

```bash
python src/before/run_sam3_batch.py \
  --input_dir data/inputs/FHD10 \
  --output_dir outputs/before/FHD10 \
  --prompts "broccoli" \
  --gpu_ids "0"
```

다른 입력 조건을 사용하려면 `--input_dir`와 `--output_dir`를 변경한다.

```bash
python src/before/run_sam3_batch.py \
  --input_dir data/inputs/HD10 \
  --output_dir outputs/before/HD10 \
  --prompts "broccoli" \
  --gpu_ids "0"
```

### 4.2 최적화 후 코드 실행

최종 개선 버전은 `src/after/run_sam3_batch_after.py`이다.

```bash
python src/after/run_sam3_batch_after.py \
  --input_dir data/inputs/FHD10 \
  --output_dir outputs/after/FHD10 \
  --prompts "broccoli" \
  --gpu_ids "0"
```

이미 생성된 결과를 건너뛰고 실행하려면 `--skip_existing` 옵션을 추가한다.

```bash
python src/after/run_sam3_batch_after.py \
  --input_dir data/inputs/FHD10 \
  --output_dir outputs/after/FHD10 \
  --prompts "broccoli" \
  --gpu_ids "0" \
  --skip_existing
```

### 4.3 중간 개선 단계 실행

최종 개선 전 단계 코드를 실행하려면 `run_sam3_batch_afterCB.py`를 사용한다.

```bash
python src/after/run_sam3_batch_afterCB.py \
  --input_dir data/inputs/FHD10 \
  --output_dir outputs/afterCB/FHD10 \
  --prompts "broccoli" \
  --gpu_ids "0"
```

전전 단계 코드를 실행하려면 `run_sam3_batch_afterC.py`를 사용한다.

```bash
python src/after/run_sam3_batch_afterC.py \
  --input_dir data/inputs/FHD10 \
  --output_dir outputs/afterC/FHD10 \
  --prompts "broccoli" \
  --gpu_ids "0"
```

### 4.4 벤치마크 실행

before/after 코드를 동일 조건에서 순차 실행하고, 실행 시간과 메모리 사용량을 측정하려면 다음 명령을 사용한다.

```bash
python benchmark/run_benchmark.py \
  --before_script src/before/run_sam3_batch.py \
  --after_script src/after/run_sam3_batch_after.py \
  --input_dir data/inputs/FHD10 \
  --results_csv results/FHD10/benchmark_results.csv \
  --output_root results/FHD10/benchmark_outputs \
  --prompts "broccoli" \
  --repeats 3 \
  --gpu_id 0
```

다른 입력 조건을 벤치마크하려면 `--input_dir`, `--results_csv`, `--output_root`를 함께 변경한다.

```bash
python benchmark/run_benchmark.py \
  --before_script src/before/run_sam3_batch.py \
  --after_script src/after/run_sam3_batch_after.py \
  --input_dir data/inputs/HD10 \
  --results_csv results/HD10/benchmark_results.csv \
  --output_root results/HD10/benchmark_outputs \
  --prompts "broccoli" \
  --repeats 3 \
  --gpu_id 0
```

벤치마크는 다음 항목을 CSV로 기록한다.

```text
version
- before 또는 after

repeat
- 반복 번호

elapsed_sec
- 전체 실행 시간

peak_rss_mb
- 프로세스 트리 기준 CPU peak RSS 메모리

peak_gpu_memory_mb
- nvidia-smi 기준 GPU peak memory
```

결과 CSV는 다음 위치에 저장된다.

```text
results/<입력조건>/benchmark_results.csv
```

예시는 다음과 같다.

```text
results/FHD10/benchmark_results.csv
results/HD10/benchmark_results.csv
```

벤치마크 실행 중 생성되는 출력 비디오와 로그는 GitHub에 포함하지 않는다.

### 4.5 재현성 관련 주의사항

벤치마크 결과는 다음 조건에 영향을 받을 수 있다.

```text
- 사용 GPU 종류와 VRAM 크기
- CUDA 및 PyTorch 버전
- SAM3 checkpoint
- 입력 영상의 길이, 해상도, FPS
- 프롬프트 개수
- 백그라운드 프로세스의 GPU 메모리 사용량
- warm-up 여부
```

따라서 before/after 비교는 반드시 같은 입력 영상, 같은 프롬프트, 같은 GPU, 같은 실행 환경에서 수행해야 한다.
