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

Tên repo có "Web Application", nhưng trạng thái code hiện tại mới đi tới các
bước dữ liệu, tiền xử lý audio, EDA tối thiểu và baseline MFCC. CNN, inference
pipeline và web app vẫn là phase sau trong `PLAN.md`.

## 2. Trạng Thái Hiện Tại

Theo `PLAN.md`, repo đã có các phần chính sau:

| Phase | Trạng thái trong repo |
| --- | --- |
| Phase 1: Metadata | Đã chuẩn bị metadata và subset audio cân bằng. |
| Phase 2: Audio preprocessing | Đã chuẩn hoá audio về mono 16 kHz, 16 giây. |
| Phase 3: Data EDA | Đã có báo cáo kiểm tra tối thiểu. |
| Phase 4: MFCC baselines | Đã train Logistic Regression và SVM. |
| Phase 5+ | Chưa có CNN, PhoWhisper, inference hoặc web app. |

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
  -> MFCC mean/std features
  -> Logistic Regression + SVM
  -> outputs/metrics/, outputs/models/, outputs/reports/
```

Nói ngắn gọn:

1. `prepare_metadata.py` đọc metadata từ ViMD, map nhãn về 3 vùng, chọn subset
   audio nhỏ và cân bằng.
2. `preprocess_audio.py` chuẩn hoá từng file audio để mọi file có cùng format.
3. `mfcc.py` biến waveform thành vector đặc trưng MFCC.
4. `train_baseline.py` train model truyền thống bằng các vector MFCC.
5. Các kết quả được ghi vào `outputs/` và tóm tắt trong `reports/`.

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

### Bước 7: Đọc tests

Tests nằm trong `tests/`:

| File | Kiểm tra gì |
| --- | --- |
| `tests/test_prepare_metadata.py` | Mapping label, chọn shard, ưu tiên speaker khác nhau. |
| `tests/test_audio_preprocessing.py` | Shape audio, trim silence, pad/crop, stereo input. |
| `tests/test_mfcc.py` | Shape MFCC và giá trị hữu hạn. |

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

## 6. Quy Tắc Nhỏ Khi Đọc Và Sửa Code

- Project chỉ support 3 class: `Northern`, `Central`, `Southern`.
- Không thêm province-level classification hoặc speaker identity prediction.
- Không viết một pipeline audio riêng cho app. Hãy dùng lại `src/utils/audio.py`.
- Không ghi đè output quan trọng nếu script không có `--overwrite`.
- Nếu thêm code mới, nên có một test hoặc sanity check nhỏ.
- Sau khi implement task, cập nhật `reports/implementation_report.md`.

## 7. Những Thứ Chưa Có Trong Repo

Đừng mất thời gian tìm các phần này trong code hiện tại:

- Chưa có lightweight CNN.
- Chưa có PhoWhisper experiment.
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
7. Mở `outputs/metrics/baseline_results.json` để xem metric cuối cùng.

Sau checklist này, bạn sẽ nắm được luồng chính của project từ dữ liệu đến model
baseline.
