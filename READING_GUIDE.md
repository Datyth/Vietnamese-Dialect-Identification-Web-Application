# Reading Guide

Tài liệu này giúp người mới bắt đầu đọc hiểu project theo cách đơn giản nhất:
biết project đang làm gì, dữ liệu đi qua những bước nào, nên đọc file nào trước,
và chạy lệnh nào để kiểm tra.

## 1. Project Này Là Gì?

Project này xây một pipeline nhận diện phương ngữ tiếng Việt theo 3 vùng:

| Nhãn trong project | Ý nghĩa |
| --- | --- |
| `Northern` | Giọng miền Bắc |
| `Central` | Giọng miền Trung |
| `Southern` | Giọng miền Nam |

Dataset chính là ViMD. Dataset gốc có thông tin vùng/province, nhưng project này
chỉ gom về 3 nhãn lớn ở trên. Project không dự đoán quê thật, danh tính người
nói, hay tỉnh/thành cụ thể.

Tên repo có "Web Application", nhưng trạng thái code hiện tại mới đi tới dữ
liệu, tiền xử lý audio, EDA tối thiểu, baseline MFCC, lightweight CNN,
PhoWhisper-base experiment và final evaluation/error analysis. Inference
pipeline và web app vẫn là phase sau trong `PLAN.md`.

## 2. Trạng Thái Hiện Tại

Theo `PLAN.md`, repo đã có các phần chính sau:

| Phase | Trạng thái trong repo |
| --- | --- |
| Phase 1: Metadata | Đã chuẩn bị metadata và subset audio cân bằng. |
| Phase 2: Audio preprocessing | Đã chuẩn hoá audio về mono 16 kHz, 16 giây. |
| Phase 3: Data EDA | Đã có báo cáo kiểm tra tối thiểu. |
| Phase 4: MFCC baselines | Đã train Logistic Regression và SVM. |
| Phase 5: Lightweight CNN | Đã có log-Mel feature, CNN nhỏ và script train PyTorch. |
| Phase 6: PhoWhisper-base | Đã fine-tune PhoWhisper-base cho 3-class classification. |
| Phase 7: Final evaluation | Đã tổng hợp metrics và error analysis. |
| Phase 8+ | Chưa có inference hoặc web app. |

Subset hiện tại có 390 file audio đã tải:

| Split | Northern | Central | Southern | Tổng |
| --- | ---: | ---: | ---: | ---: |
| train | 100 | 100 | 100 | 300 |
| valid | 15 | 15 | 15 | 45 |
| test | 15 | 15 | 15 | 45 |

## 3. Workflow Tổng Quát

Hãy hình dung dữ liệu đi qua pipeline như sau:

```text
ViMD Parquet shards
  -> data/processed/metadata_clean.csv
  -> data/processed/audio_16k/
  -> data/processed/audio_preprocessed_16s/
  -> data/processed/preprocessed_metadata.csv
  -> MFCC mean/std features -> Logistic Regression + SVM
  -> log-Mel spectrograms -> lightweight CNN
  -> PhoWhisper input_features -> PhoWhisper-base classifier
  -> final comparison + error analysis
  -> outputs/metrics/, outputs/models/, outputs/reports/
```

Nói ngắn gọn:

1. `prepare_metadata.py` đọc metadata từ ViMD, map nhãn về 3 vùng, chọn subset
   audio nhỏ và cân bằng.
2. `preprocess_audio.py` chuẩn hoá từng file audio để mọi file có cùng format.
3. `mfcc.py` biến waveform thành vector đặc trưng MFCC cho baseline truyền
   thống.
4. `logmel.py` biến waveform thành log-Mel spectrogram cho CNN.
5. `train_baseline.py` train Logistic Regression và SVM.
6. `train_cnn.py` train lightweight CNN bằng PyTorch.
7. `train_phowhisper.py` fine-tune PhoWhisper-base bằng Transformers.
8. `final_evaluation.py` tổng hợp metrics và sinh error analysis.
9. Các kết quả được ghi vào `outputs/` và tóm tắt trong `reports/`.

## 4. Nên Đọc File Nào Trước?

Đọc theo thứ tự này sẽ dễ hiểu nhất.

### Bước 1: Đọc tài liệu định hướng

- `PLAN.md`: phạm vi từng phase, phần nào được làm, phần nào chưa.
- `README.md`: lệnh chạy chính và kết quả hiện tại.
- `OVERVIEW.md`: bối cảnh bài toán và mục tiêu ban đầu.

Nếu bạn mới vào repo, luôn đọc `PLAN.md` trước để tránh tưởng rằng project đã có
CNN hoặc web app.

### Bước 2: Xem dữ liệu đã sinh ra

Các file nên mở:

- `data/processed/metadata_clean.csv`
- `data/processed/preprocessed_metadata.csv`
- `data/processed/split_class_counts.csv`
- `outputs/reports/phase1_dataset_summary.json`
- `outputs/reports/phase2_preprocessing_summary.json`

Một dòng trong `metadata_clean.csv` đại diện cho một audio sample. Các cột quan
trọng:

| Cột | Ý nghĩa |
| --- | --- |
| `sample_id` | ID dạng `split:filename`, ví dụ `train:17_0191.wav`. |
| `source_split` | Split gốc: `train`, `valid`, hoặc `test`. |
| `source_region` | Nhãn vùng gốc của ViMD: `North`, `Central`, `South`. |
| `label` | Nhãn chuẩn của project: `Northern`, `Central`, `Southern`. |
| `speaker_id` | ID người nói nếu dataset có. Dùng để kiểm tra overlap. |
| `audio_path` | Đường dẫn audio 16 kHz đã tải ở Phase 1. |
| `audio_status` | `downloaded` nếu file được chọn vào subset local. |

`preprocessed_metadata.csv` thêm các cột sau:

| Cột | Ý nghĩa |
| --- | --- |
| `preprocessed_audio_path` | Đường dẫn audio sau khi chuẩn hoá 16 giây. |
| `preprocessing_status` | `preprocessed` nếu xử lý thành công. |
| `preprocessed_sample_rate` | Luôn là `16000` nếu đúng. |
| `preprocessed_samples` | Luôn là `256000` vì 16 kHz x 16 giây. |
| `preprocessed_rms`, `preprocessed_peak` | Thống kê âm lượng sau xử lý. |

### Bước 3: Đọc code xử lý audio

File chính:

- `src/utils/audio.py`

Đây là module nên hiểu kỹ vì training và inference sau này đều nên dùng chung
logic này. Các hàm quan trọng:

| Hàm | Vai trò |
| --- | --- |
| `load_audio` | Đọc audio bằng SoundFile, trả về waveform mono `float32`. |
| `resample_audio` | Đổi sample rate về 16 kHz bằng `soxr`. |
| `trim_silence` | Cắt khoảng lặng ở đầu/cuối audio. |
| `normalize_volume` | Chuẩn hoá RMS, tránh âm lượng quá nhỏ hoặc quá lớn. |
| `fix_length` | Center-crop hoặc zero-pad về đúng 256,000 samples. |
| `preprocess_waveform` | Chạy toàn bộ pipeline trên waveform. |
| `preprocess_file` | Chạy pipeline cho một file và ghi WAV PCM 16-bit. |

Thông số quan trọng nằm ngay đầu file:

```python
TARGET_SAMPLE_RATE = 16_000
FIXED_DURATION_SECONDS = 16.0
TARGET_SAMPLES = 256000
```

### Bước 4: Đọc script preprocess

File chính:

- `src/data/preprocess_audio.py`

Script này đọc `metadata_clean.csv`, lấy các dòng có
`audio_status=downloaded`, xử lý từng file audio, rồi ghi:

- `data/processed/audio_preprocessed_16s/`
- `data/processed/preprocessed_metadata.csv`
- `data/processed/preprocess_audio_issues.csv`
- `outputs/reports/phase2_preprocessing_summary.json`
- `outputs/reports/data_eda.md`

Điểm cần chú ý:

- Script kiểm tra input field bắt buộc trước khi chạy.
- Nếu output đã tồn tại, script sẽ từ chối ghi đè trừ khi có `--overwrite`.
- Mỗi file lỗi sẽ được ghi vào `preprocess_audio_issues.csv` thay vì bị bỏ qua
  âm thầm.

### Bước 5: Đọc feature MFCC

File chính:

- `src/features/mfcc.py`

MFCC là cách biến audio thành vector số để model truyền thống học được. Trong
project này, MFCC được implement bằng NumPy thay vì thêm dependency nặng như
librosa.

Hàm quan trọng nhất:

```python
mfcc_mean_std(waveform, sample_rate=16000)
```

Hàm này:

1. Tính ma trận MFCC theo từng frame thời gian.
2. Lấy mean của 13 hệ số MFCC.
3. Lấy standard deviation của 13 hệ số MFCC.
4. Ghép lại thành vector 26 chiều.

Vì vậy trong `baseline_results.json`, feature dimension là `26`.

### Bước 6: Đọc script train baseline

File chính:

- `src/training/train_baseline.py`

Script này làm 4 việc:

1. Đọc `data/processed/preprocessed_metadata.csv`.
2. Load từng audio đã preprocess và kiểm tra đúng 16 kHz, 256,000 samples.
3. Trích MFCC mean/std cho từng sample.
4. Train 2 model scikit-learn:
   - `logistic_regression`
   - `svm`

Output chính:

- `outputs/metrics/baseline_results.json`
- `outputs/metrics/*_confusion_matrix.csv`
- `outputs/models/logistic_regression_mfcc.pkl`
- `outputs/models/svm_mfcc.pkl`
- `outputs/reports/phase4_baseline_report.md`

Kết quả hiện tại:

| Model | Valid Accuracy | Valid Macro F1 | Test Accuracy | Test Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.6000 | 0.5981 | 0.6222 | 0.6292 |
| SVM | 0.6889 | 0.6918 | 0.6222 | 0.6264 |

Theo validation macro F1, baseline tốt nhất hiện tại là `svm`.

### Bước 7: Đọc Phase 5 CNN

Các file chính:

- `src/features/logmel.py`
- `src/models/cnn.py`
- `src/training/train_cnn.py`

`logmel.py` tạo spectrogram dạng `[n_mels, frames]`, mặc định là 64 Mel bins.
Khi đưa vào CNN, script train thêm channel dimension để tensor có dạng:

```text
[batch, 1, n_mels, frames]
```

`src/models/cnn.py` định nghĩa `LightweightCNN`: 3 convolution blocks nhỏ,
adaptive average pooling, dropout và linear classifier cho 3 nhãn.

`train_cnn.py` làm các việc chính:

1. Đọc `data/processed/preprocessed_metadata.csv`.
2. Load audio đã preprocess và kiểm tra đúng 16 kHz, 256,000 samples.
3. Trích log-Mel spectrogram.
4. Chọn device: `mps`, `cuda` hoặc `cpu`.
5. Train CNN với early stopping theo validation macro F1.
6. Lưu checkpoint, metrics, confusion matrix, training log và report.

Checkpoint được ghi vào `outputs/models/lightweight_cnn_logmel.pt` nhưng thư mục
`outputs/models/` đang được ignore để tránh commit model artifact.

Kết quả Phase 5 nằm ở:

- `outputs/metrics/cnn_results.json`
- `outputs/metrics/cnn_training_log.csv`
- `outputs/reports/phase5_cnn_report.md`

Run hiện tại chọn `cpu` vì PyTorch trong môi trường này báo `mps=False` và
`cuda=False`. Nếu cài PyTorch build có MPS hoặc CUDA, `--device auto` sẽ ưu tiên
`mps`, rồi `cuda`, rồi `cpu`.

### Bước 8: Đọc Phase 6 PhoWhisper-base

Các file chính:

- `src/training/train_phowhisper.py`
- `outputs/metrics/phowhisper_results.json`
- `outputs/reports/phase6_phowhisper_report.md`

PhoWhisper-base là model pretrained lớn hơn CNN nhiều: khoảng 74M parameters và
published PyTorch weights khoảng 290 MB. Script dùng
`WhisperForAudioClassification` để fine-tune cho 3 nhãn dialect, không dùng ASR
generation.

Run hiện tại:

| Split | Accuracy | Macro F1 |
| --- | ---: | ---: |
| train | 0.9933 | 0.9933 |
| valid | 0.6667 | 0.6623 |
| test | 0.7111 | 0.7113 |

Script chọn device theo thứ tự `mps`, `cuda`, rồi `cpu` khi dùng `--device auto`.
Run hiện tại dùng `mps`, early-stopped ở epoch 6 và chọn best epoch 3.

### Bước 9: Đọc Phase 7 final evaluation

File chính:

- `src/evaluation/final_evaluation.py`
- `outputs/metrics/final_comparison.csv`
- `outputs/metrics/final_sample_errors.csv`
- `outputs/reports/error_analysis.md`

Phase 7 chọn best model theo validation macro F1. Hiện tại best model vẫn là
Phase 4 SVM, dù PhoWhisper-base có test macro F1 cao nhất.

### Bước 10: Đọc tests

Tests nằm trong `tests/`:

| File | Kiểm tra gì |
| --- | --- |
| `tests/test_prepare_metadata.py` | Mapping label, chọn shard, ưu tiên speaker khác nhau. |
| `tests/test_audio_preprocessing.py` | Shape audio, trim silence, pad/crop, stereo input. |
| `tests/test_mfcc.py` | Shape MFCC và giá trị hữu hạn. |
| `tests/test_logmel.py` | Shape log-Mel và giá trị hữu hạn. |
| `tests/test_cnn.py` | Forward pass CNN và device resolver. |
| `tests/test_phowhisper.py` | Device resolver và split metadata cho PhoWhisper. |
| `tests/test_final_evaluation.py` | Tổng hợp final metrics và error output fields. |

Đây là nơi tốt để người mới hiểu kỳ vọng hành vi của từng module mà không phải
đọc toàn bộ script một lần.

## 5. Các Lệnh Hay Dùng

Cài dependency trong virtual environment local:

```bash
uv venv .venv --python 3.10
uv pip install --python .venv/bin/python -r requirements.txt
```

Chạy unit tests:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Chuẩn bị metadata và audio subset từ ViMD:

```bash
.venv/bin/python -m src.data.prepare_metadata
```

Lệnh trên cần network để đọc Hugging Face. Nếu chỉ muốn đọc metadata remote mà
không tải audio, dùng:

```bash
.venv/bin/python -m src.data.prepare_metadata --metadata-only
```

Chạy lại preprocessing:

```bash
.venv/bin/python -m src.data.preprocess_audio --overwrite
```

Train lại baseline MFCC:

```bash
.venv/bin/python -m src.training.train_baseline --overwrite
```

Train lightweight CNN:

```bash
.venv/bin/python -m src.training.train_cnn --overwrite
```

Chọn device rõ ràng nếu cần:

```bash
.venv/bin/python -m src.training.train_cnn --overwrite --device mps
.venv/bin/python -m src.training.train_cnn --overwrite --device cuda
.venv/bin/python -m src.training.train_cnn --overwrite --device cpu
```

Fine-tune PhoWhisper-base:

```bash
.venv/bin/python -m src.training.train_phowhisper --overwrite --device auto
```

Tạo final comparison và error analysis:

```bash
.venv/bin/python -m src.evaluation.final_evaluation --overwrite
```

## 6. Quy Tắc Nhỏ Khi Đọc Và Sửa Code

- Project chỉ support 3 class: `Northern`, `Central`, `Southern`.
- Không thêm province-level classification hoặc speaker identity prediction.
- Không viết một pipeline audio riêng cho app. Hãy dùng lại `src/utils/audio.py`.
- Không ghi đè output quan trọng nếu script không có `--overwrite`.
- Nếu thêm code mới, nên có một test hoặc sanity check nhỏ.
- Sau khi implement task, cập nhật `reports/implementation_report.md`.

## 7. Những Thứ Chưa Có Trong Repo

Đừng mất thời gian tìm các phần này trong code hiện tại:

- Chưa có `src/inference/predict.py`.
- Chưa có Streamlit app hoặc web UI.
- Chưa có ONNX export.

Các phần này thuộc phase sau trong `PLAN.md` và chỉ nên thêm khi có yêu cầu rõ
ràng.

## 8. Cách Đọc Nhanh Trong 30 Phút

Nếu bạn chỉ có ít thời gian, hãy đọc theo checklist này:

1. Đọc `PLAN.md` để hiểu scope và phase.
2. Đọc `README.md` để biết lệnh chạy và kết quả hiện tại.
3. Mở `data/processed/preprocessed_metadata.csv` để nhìn schema thật.
4. Đọc `src/utils/audio.py` để hiểu audio được chuẩn hoá thế nào.
5. Đọc `src/features/mfcc.py`, tập trung vào `mfcc_mean_std`.
6. Đọc `src/training/train_baseline.py`, tập trung vào `extract_features`,
   `build_models`, `evaluate` và `main`.
7. Đọc `src/features/logmel.py` và `src/models/cnn.py`.
8. Đọc `src/training/train_cnn.py`, tập trung vào `resolve_device`,
   `extract_logmel_features`, `train_one_epoch`, `evaluate_model` và `main`.
9. Đọc `src/training/train_phowhisper.py`, tập trung vào `resolve_device`,
   `extract_input_features`, `train_one_epoch`, `evaluate_model` và `main`.
10. Đọc `src/evaluation/final_evaluation.py`.
11. Mở `outputs/metrics/baseline_results.json`, `outputs/metrics/cnn_results.json`,
   `outputs/metrics/phowhisper_results.json` và
   `outputs/metrics/final_comparison.csv` để xem metric cuối cùng.

Sau checklist này, bạn sẽ nắm được luồng chính của project từ dữ liệu đến model
baseline, CNN, PhoWhisper và final evaluation.
