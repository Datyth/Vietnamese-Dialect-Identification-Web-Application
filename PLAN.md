## Phase Definitions

Phần này chia project thành các phase nhỏ để dễ triển khai, kiểm tra tiến độ và tránh mở rộng scope quá mức. Mỗi phase có **scope** rõ ràng và **expected outputs** cụ thể.

### Phase 0: Project setup and scope freeze

**Scope**

- Chốt phạm vi project ở mức 3 classes: Northern, Central, Southern.
- Tạo cấu trúc repo tối thiểu để code, data, outputs và app không bị lẫn nhau.
- Tạo config trung tâm cho các tham số chính như sample rate, duration, số class và label names.
- Chưa xử lý dataset, chưa train model, chưa làm web app.

**Expected outputs**

- Repo structure cơ bản.
- `README.md` bản nháp.
- `requirements.txt` hoặc dependency file theo convention của repo.
- `configs/config.yaml`.
- `.gitignore` cho data, model artifacts, logs và cache.
- Một script hoặc command kiểm tra project import được.

### Phase 1: Dataset acquisition and metadata preparation

**Scope**

- Lấy hoặc mount dataset Vietnamese dialect speech.
- Kiểm tra schema dataset gốc.
- Mapping province-level hoặc dialect-level labels về 3 vùng: Northern, Central, Southern.
- Tạo metadata chuẩn để các phase sau dùng lại.
- Chưa preprocess audio hàng loạt, chưa extract features, chưa train model.

**Expected outputs**

- `data/metadata.csv` hoặc `data/processed/metadata_clean.csv`.
- Label mapping file nếu dataset có province-level labels.
- Bảng thống kê số samples theo class.
- Bảng thống kê số speakers theo class nếu có speaker ID.
- Danh sách file lỗi hoặc missing audio nếu có.

### Phase 2: Audio preprocessing pipeline

**Scope**

- Xây pipeline chuẩn hóa audio dùng chung cho training và inference.
- Các bước chính: load audio, convert mono, resample 16 kHz, normalize volume, trim hoặc pad về fixed duration.
- Không train model trong phase này.
- Không viết logic riêng cho app khác với training.

**Expected outputs**

- `src/utils/audio.py` hoặc module tương đương theo repo convention.
- `src/data/preprocess_audio.py` hoặc preprocessing script tương đương.
- Optional processed audio folder nếu quyết định lưu audio đã chuẩn hóa.
- Log các file audio lỗi.
- Unit tests hoặc sanity checks cho waveform shape, sample rate và duration.

### Phase 3: Data exploration and validation

**Scope**

- Phân tích dữ liệu sau khi có metadata và preprocessing cơ bản.
- Kiểm tra class imbalance, duration distribution, file lỗi, speaker overlap và split quality.
- Tạo một số visualization để đưa vào report.
- Không train model chính trong phase này.

**Expected outputs**

- `outputs/reports/data_eda.md`.
- Class distribution figure.
- Duration histogram.
- Sample waveform figure.
- Sample log-Mel spectrogram figure.
- Split validation summary.

### Phase 4: Traditional baseline models

**Scope**

- Train baseline đơn giản bằng MFCC features.
- Models chính: Logistic Regression và SVM.
- Feature vector dùng MFCC aggregated bằng mean và standard deviation theo time axis.
- Không dùng CNN hoặc PhoWhisper trong phase này.
- Không làm app trong phase này.

**Expected outputs**

- MFCC feature extraction module.
- Logistic Regression baseline.
- SVM baseline.
- Saved baseline models nếu cần.
- `outputs/metrics/baseline_results.json` hoặc equivalent.
- Confusion matrix cho baseline.
- Short baseline report.

### Phase 5: Lightweight CNN model

**Scope**

- Train model deep learning chính từ scratch.
- Input chính là log-Mel spectrogram.
- Model là lightweight CNN, không phải architecture quá lớn.
- Tập trung vào reproducible training, validation và test evaluation.
- Không fine-tune pretrained model trong phase này.

**Expected outputs**

- Log-Mel feature extraction module.
- CNN dataset class hoặc dataloader.
- Lightweight CNN model definition.
- Training script.
- Best checkpoint theo validation macro F1.
- `outputs/metrics/cnn_results.json`.
- CNN confusion matrix.
- Training log hoặc learning curve.

### Phase 6: PhoWhisper-base experiment

**Scope**

- Chạy experiment với một pretrained speech model duy nhất: PhoWhisper-base.
- Ưu tiên freeze encoder và train classifier nhỏ trước.
- Partial fine-tuning chỉ là stretch nếu còn thời gian và compute.
- Không để phase này block toàn bộ project nếu compute hoặc dependency gặp vấn đề.

**Expected outputs**

- PhoWhisper feature extraction hoặc encoder wrapper.
- Classifier head cho 3-class dialect classification.
- Training hoặc embedding extraction script.
- `outputs/metrics/phowhisper_results.json`.
- Confusion matrix cho PhoWhisper experiment.
- Model size và latency estimate nếu chạy được.

### Phase 7: Final evaluation and error analysis

**Scope**

- Tổng hợp kết quả từ baseline, CNN và PhoWhisper nếu có.
- So sánh model theo accuracy, macro F1, per-class F1, model size và CPU inference latency.
- Phân tích lỗi dự đoán sai để hiểu confusion giữa các vùng.
- Không train thêm model mới trừ khi cần rerun do lỗi rõ ràng.

**Expected outputs**

- `outputs/metrics/final_comparison.csv`.
- Final comparison table.
- Confusion matrix của best model.
- `outputs/reports/error_analysis.md`.
- File hoặc bảng sample errors gồm filepath, true label, predicted label, confidence, duration và notes.

### Phase 8: Inference pipeline and web application

**Scope**

- Xây inference pipeline dùng best model đã chọn.
- Xây web app demo cho upload audio hoặc record audio nếu khả thi.
- App hiển thị prediction, confidence, top predictions, waveform hoặc log-Mel visualization và latency.
- Không thêm model mới trong phase này.
- Không dùng app để claim speaker identity, hometown, ethnicity hoặc personal background.

**Expected outputs**

- `src/inference/predict.py`.
- `app/streamlit_app.py` hoặc web app entrypoint tương đương.
- Best model artifact được load bởi app.
- Demo app chạy local.
- Inference latency hiển thị trong app.
- Disclaimer trong app.


### Phase 9: Extended Deep Learning Experiments

**Scope**

* Thực nghiệm thêm các mô hình deep learning từ E1 đến E5 để so sánh với baseline truyền thống và lightweight CNN chính.
* Mục tiêu là đánh giá trade-off giữa:

  * Accuracy.
  * Macro F1.
  * Per-class F1.
  * Model size.
  * Inference latency.
  * Khả năng deploy thực tế.
* Các experiment trong phase này chỉ được thực hiện sau khi MVP đã hoàn thành.
* Không để phase này block final report nếu thiếu GPU, dependency conflict hoặc pretrained model khó setup.
* Không thêm class mới ngoài 3 nhãn: Northern, Central, Southern.

---

#### Shared experimental setup

Tất cả experiment E1–E5 cần dùng chung setup để kết quả so sánh công bằng.

**Data setup**

* Dùng cùng `metadata_clean.csv`.
* Dùng cùng train/validation/test split.
* Không để speaker overlap giữa train, validation và test nếu dataset có speaker ID.
* Input audio được chuẩn hóa giống pipeline chính:

  * Mono.
  * Sample rate 16 kHz.
  * Fixed duration, ví dụ 10s hoặc 16s.
  * Normalize volume.
  * Trim hoặc pad audio.

**Evaluation metrics**

Mỗi model cần báo cáo:

* Accuracy.
* Macro F1.
* Precision, recall, F1 theo từng class.
* Confusion matrix.
* Model size.
* Average inference latency.
* Training time nếu đo được.

**Output format**

```text
outputs/
  metrics/
    e1_mobilenetv3_results.json
    e2_efficientnetb0_results.json
    e3_wav2vec2_results.json
    e4_phowhisper_results.json
    e5_vipvl_chunkformer_results.json
    e6_whisper_base_results.json
    deep_learning_comparison.csv

  figures/
    e1_mobilenetv3_confusion_matrix.png
    e2_efficientnetb0_confusion_matrix.png
    e3_wav2vec2_confusion_matrix.png
    e4_phowhisper_confusion_matrix.png
    e5_vipvl_chunkformer_confusion_matrix.png
    e6_whisper_base_confusion_matrix.png
    deep_learning_comparison.png

  reports/
    extended_deep_learning_experiments.md
```

---

#### E1: Log-Mel Spectrogram + MobileNetV3-Small

**Goal**

Xây dựng một deep learning baseline nhẹ, dễ train và dễ deploy bằng cách dùng log-Mel spectrogram như ảnh 2D.

**Model pipeline**

```text
Audio
→ Audio Preprocessing
→ Log-Mel Spectrogram
→ MobileNetV3-Small
→ Global Average Pooling
→ Linear Classifier
→ 3 Dialect Classes
```

**Setup**

* Input feature: log-Mel spectrogram.
* Model backbone: MobileNetV3-Small.
* Pretraining:

  * Option 1: ImageNet pretrained weights.
  * Option 2: train from scratch nếu muốn tránh domain mismatch.
* Classifier head:

  * Dropout.
  * Linear layer.
  * Softmax over 3 classes.

**Suggested hyperparameters**

```yaml
experiment_name: e1_mobilenetv3_logmel
feature: log_mel_spectrogram
model: mobilenetv3_small
sample_rate: 16000
duration_sec: 16
n_mels: 64
n_fft: 1024
hop_length: 512
num_classes: 3
batch_size: 16
learning_rate: 1e-4
weight_decay: 1e-4
dropout: 0.3
max_epochs: 50
early_stopping_patience: 8
metric_for_best_model: macro_f1
```

**Expected outputs**

* `src/features/logmel.py`.
* `src/models/mobilenetv3_classifier.py`.
* `src/training/train_e1_mobilenetv3.py`.
* `configs/experiments/e1_mobilenetv3.yaml`.
* `outputs/metrics/e1_mobilenetv3_results.json`.
* Confusion matrix.
* Learning curve.
* Model size and latency estimate.

---

#### E2: Log-Mel Spectrogram + EfficientNet-B0

**Goal**

Thử một CNN mạnh hơn MobileNetV3 nhưng vẫn tương đối nhẹ để kiểm tra xem tăng capacity có cải thiện performance không.

**Model pipeline**

```text
Audio
→ Audio Preprocessing
→ Log-Mel Spectrogram
→ EfficientNet-B0
→ Global Average Pooling
→ Linear Classifier
→ 3 Dialect Classes
```

**Setup**

* Input feature: log-Mel spectrogram.
* Model backbone: EfficientNet-B0.
* Pretraining:

  * Có thể dùng ImageNet pretrained weights.
  * Nếu dùng spectrogram 1 channel, có thể:

    * Repeat thành 3 channels.
    * Hoặc sửa first convolution để nhận 1 channel.
* Classifier head:

  * Dropout.
  * Linear layer.
  * Softmax over 3 classes.

**Suggested hyperparameters**

```yaml
experiment_name: e2_efficientnetb0_logmel
feature: log_mel_spectrogram
model: efficientnet_b0
sample_rate: 16000
duration_sec: 16
n_mels: 64
n_fft: 1024
hop_length: 512
num_classes: 3
input_channels: 3
batch_size: 16
learning_rate: 1e-4
weight_decay: 1e-4
dropout: 0.3
max_epochs: 50
early_stopping_patience: 8
metric_for_best_model: macro_f1
```

**Expected outputs**

* `src/models/efficientnet_classifier.py`.
* `src/training/train_e2_efficientnet.py`.
* `configs/experiments/e2_efficientnetb0.yaml`.
* `outputs/metrics/e2_efficientnetb0_results.json`.
* Confusion matrix.
* Learning curve.
* Model size and latency estimate.

---

#### E3: wav2vec2 Vietnamese Encoder + Classifier

**Goal**

Đánh giá pretrained speech encoder cho tiếng Việt, dùng acoustic representation học sẵn thay vì chỉ train CNN từ spectrogram.

**Model pipeline**

```text
Audio
→ Audio Preprocessing
→ wav2vec2 Vietnamese Encoder
→ Mean Pooling / Attention Pooling
→ Classifier Head
→ 3 Dialect Classes
```

**Setup**

* Model chính: `wav2vec2-base-vietnamese-250h` hoặc checkpoint wav2vec2 tiếng Việt tương đương.
* Bỏ ASR/CTC head.
* Dùng encoder output làm representation.
* Gắn classifier head cho 3-class classification.

**Training strategy**

Stage 1:

* Freeze encoder.
* Train classifier head.

Stage 2:

* Unfreeze một số layer cuối nếu GPU cho phép.
* Fine-tune với learning rate nhỏ.

**Suggested hyperparameters**

```yaml
experiment_name: e3_wav2vec2_vietnamese
model_name: wav2vec2_base_vietnamese
sample_rate: 16000
duration_sec: 16
num_classes: 3
pooling: mean
freeze_encoder_epochs: 5
fine_tune_last_n_layers: 4
batch_size: 4
learning_rate_head: 1e-3
learning_rate_encoder: 1e-5
weight_decay: 1e-4
dropout: 0.3
max_epochs: 20
early_stopping_patience: 5
metric_for_best_model: macro_f1
```

**Expected outputs**

* `src/models/wav2vec2_classifier.py`.
* `src/training/train_e3_wav2vec2.py`.
* `configs/experiments/e3_wav2vec2.yaml`.
* `outputs/metrics/e3_wav2vec2_results.json`.
* Confusion matrix.
* Classification report.
* Model size and latency estimate.

---

#### E4: PhoWhisper-base Encoder + Classifier

**Goal**

Đánh giá PhoWhisper-base cho Vietnamese dialect classification, tận dụng pretrained model đã thích nghi tốt hơn với tiếng Việt so với Whisper gốc.

**Model pipeline**

```text
Audio
→ PhoWhisper Feature Extractor
→ PhoWhisper Encoder
→ Mean Pooling / Attention Pooling
→ Classifier Head
→ 3 Dialect Classes
```

**Setup**

* Model chính: `vinai/PhoWhisper-base`.
* Chỉ dùng encoder.
* Không dùng decoder để sinh transcript.
* Không dùng speech-to-text output làm feature chính.
* Gắn classifier head cho 3 classes.

**Training strategy**

Option A: frozen encoder

* Freeze PhoWhisper encoder.
* Train classifier head.
* Phù hợp khi GPU yếu.

Option B: partial fine-tuning

* Unfreeze 2–4 layer cuối.
* Fine-tune với learning rate nhỏ.
* Chỉ làm nếu frozen encoder chưa đủ tốt và còn compute.

Option C: offline embedding extraction

* Extract embedding từ PhoWhisper encoder.
* Lưu embedding ra file.
* Train Logistic Regression hoặc MLP nhỏ trên embedding.
* Dùng khi không đủ GPU để end-to-end training.

**Suggested hyperparameters**

```yaml
experiment_name: e4_phowhisper_base
model_name: vinai_phowhisper_base
sample_rate: 16000
duration_sec: 16
num_classes: 3
pooling: mean
freeze_encoder_epochs: 5
fine_tune_last_n_layers: 4
batch_size: 4
learning_rate_head: 1e-3
learning_rate_encoder: 5e-6
weight_decay: 1e-4
dropout: 0.3
max_epochs: 20
early_stopping_patience: 5
metric_for_best_model: macro_f1
```

**Expected outputs**

* `src/models/phowhisper_classifier.py`.
* `src/features/extract_phowhisper_embeddings.py` nếu dùng embedding mode.
* `src/training/train_e4_phowhisper.py`.
* `configs/experiments/e4_phowhisper_base.yaml`.
* `outputs/metrics/e4_phowhisper_results.json`.
* Confusion matrix.
* Classification report.
* Model size and latency estimate.

---

#### E5: ViP-VL / ChunkFormer Encoder + Classifier

**Goal**

Thử một pretrained speech encoder mới hơn cho tiếng Việt, ưu tiên performance cao. Experiment này là stretch vì setup có thể phức tạp hơn wav2vec2 và PhoWhisper.

**Model pipeline**

```text
Audio
→ Audio Preprocessing
→ ViP-VL / ChunkFormer Encoder
→ Pooling
→ Classifier Head
→ 3 Dialect Classes
```

**Setup**

* Model chính: ViP-VL hoặc ChunkFormer encoder nếu checkpoint và code có thể chạy ổn định.
* Chỉ dùng encoder representation cho classification.
* Không dùng ASR transcript làm feature chính.
* Nếu không thể fine-tune trực tiếp:

  * Extract embedding offline.
  * Train classifier nhỏ trên embedding.

**Training strategy**

Stage 1:

* Freeze encoder.
* Train classifier head.

Stage 2:

* Partial fine-tuning nếu dependency, GPU và codebase ổn định.

**Suggested hyperparameters**

```yaml
experiment_name: e5_vipvl_chunkformer
model_name: vipvl_chunkformer_encoder
sample_rate: 16000
duration_sec: 16
num_classes: 3
pooling: mean
freeze_encoder_epochs: 5
fine_tune_last_n_layers: 4
batch_size: 4
learning_rate_head: 1e-3
learning_rate_encoder: 5e-6
weight_decay: 1e-4
dropout: 0.3
max_epochs: 20
early_stopping_patience: 5
metric_for_best_model: macro_f1
```

**Expected outputs**

* `src/models/vipvl_chunkformer_classifier.py`.
* `src/features/extract_vipvl_embeddings.py` nếu dùng embedding mode.
* `src/training/train_e5_vipvl_chunkformer.py`.
* `configs/experiments/e5_vipvl_chunkformer.yaml`.
* `outputs/metrics/e5_vipvl_chunkformer_results.json`.
* Confusion matrix.
* Classification report.
* Model size and latency estimate.
* Ghi chú rõ ràng nếu không chạy được do checkpoint, dependency hoặc compute.

---

#### E6: Original Whisper-base Encoder + Classifier

**Goal**

So sánh PhoWhisper-base với Whisper-base gốc có kích thước tương đương để kiểm
tra lợi ích của checkpoint thích nghi tiếng Việt.

**Model pipeline**

```text
Audio
→ Whisper Feature Extractor
→ Whisper-base Encoder
→ Classification Head
→ 3 Dialect Classes
```

**Setup**

* Model chính: `openai/whisper-base`.
* Dùng cùng setup với PhoWhisper-base frozen encoder để so sánh công bằng.
* Không dùng transcript ASR làm feature chính.

**Suggested hyperparameters**

```yaml
experiment_name: e6_whisper_base_original
model_name: openai/whisper-base
sample_rate: 16000
duration_sec: 16
num_classes: 3
training_mode: frozen_encoder
batch_size: 4
learning_rate: 1e-4
weight_decay: 1e-4
max_epochs: 20
early_stopping_patience: 5
metric_for_best_model: macro_f1
```

**Expected outputs**

* `src/training/train_e6_whisper.py`.
* `configs/experiments/e6_whisper_base.yaml`.
* `outputs/metrics/e6_whisper_base_results.json`.
* Confusion matrix.
* Classification report.
* Model size and latency estimate.

---

#### Final comparison for E1–E6

Sau khi chạy các experiment, tổng hợp kết quả vào một bảng chung. Nếu E6 đã
được chạy, thêm E6 vào cùng bảng để so sánh Whisper-base gốc với PhoWhisper-base.

**Comparison table columns**

```text
experiment_id
model_name
input_type
pretrained
trainable_setting
accuracy
macro_f1
northern_f1
central_f1
southern_f1
model_size_mb
cpu_latency_ms
gpu_latency_ms
notes
```

**Expected output**

* `outputs/metrics/deep_learning_comparison.csv`.
* `outputs/reports/extended_deep_learning_experiments.md`.

**Example summary format**

| Experiment | Model                | Input    | Pretrained         | Main purpose                     |
| ---------- | -------------------- | -------- | ------------------ | -------------------------------- |
| E1         | MobileNetV3-Small    | Log-Mel  | ImageNet / Scratch | Lightweight CNN baseline         |
| E2         | EfficientNet-B0      | Log-Mel  | ImageNet / Scratch | Stronger CNN baseline            |
| E3         | wav2vec2 Vietnamese  | Waveform | Vietnamese speech  | Vietnamese pretrained encoder    |
| E4         | PhoWhisper-base      | Waveform | Vietnamese Whisper | Robust Vietnamese speech encoder |
| E5         | ViP-VL / ChunkFormer | Waveform | Vietnamese speech  | High-performance stretch model   |

---

#### Acceptance criteria

**Minimum**

* Chạy được ít nhất E1, E2 và một pretrained encoder trong E3–E5.
* Có accuracy, macro F1 và confusion matrix.
* Có bảng so sánh với MFCC baseline và lightweight CNN ở phase trước.

**Good**

* Chạy được E1–E4.
* Có model size và latency estimate.
* Có phân tích model nào tốt nhất theo performance và model nào tốt nhất theo deploy.

**Excellent**

* Chạy được đầy đủ E1–E5.
* Có freeze vs partial fine-tuning comparison cho ít nhất một pretrained model.
* Có báo cáo rõ trade-off giữa CNN nhỏ và pretrained speech encoder.

---

#### Risks and fallback plan

**Risk 1: GPU không đủ bộ nhớ**

Fallback:

* Giảm batch size.
* Dùng gradient accumulation.
* Freeze encoder.
* Chỉ fine-tune classifier head.
* Extract embedding offline rồi train classifier nhỏ.

**Risk 2: Pretrained model setup phức tạp**

Fallback:

* Ưu tiên chạy E1–E4 trước.
* Đưa E5 vào stretch experiment.
* Ghi rõ lý do nếu không chạy được.

**Risk 3: Pretrained model không tốt hơn CNN**

Fallback:

* Không ép pretrained model là best model.
* Báo cáo trung thực.
* Phân tích nguyên nhân có thể do dataset nhỏ, domain mismatch, audio ngắn, label noise hoặc overfitting.

**Risk 4: Latency quá cao để deploy**

Fallback:

* Dùng MobileNetV3 hoặc EfficientNet-B0 làm model deploy.
* Giữ pretrained model cho phần analysis/performance comparison.
### Phase 10: Hybrid PhoWhisper + CNN Fusion Experiment

**Scope**

- Thực nghiệm mô hình hybrid cho phân loại 3 dialect labels: `Northern`,
  `Central`, `Southern`.
- Dùng cùng waveform đã preprocess ở 16 kHz, fixed duration 16 giây và cùng
  train/validation/test split hiện tại.
- Global branch dùng pretrained `vinai/PhoWhisper-base` encoder, freeze toàn bộ
  parameters, không dùng decoder và không sinh hoặc dùng ASR transcript. Chọn
  PhoWhisper vì E4 hiện là baseline tốt nhất trên tập thực nghiệm hiện tại.
- Local branch dùng log-Mel spectrogram theo style feature extraction hiện tại
  và phần `features` của E2 EfficientNetB0-style đã train. Mặc định chỉ
  fine-tune nhẹ 2 child module có tham số cuối của CNN với learning rate nhỏ
  để thử cải thiện Central recall/F1 mà vẫn giữ phần lớn local encoder ổn định.
- Fusion mặc định là gated giữa embedding global và local:
  `z = alpha * z_global + (1 - alpha) * z_local`. Concat fusion là option nếu
  cần chạy ablation.
- Global PhoWhisper embedding giữ nguyên 512 chiều. Local EfficientNetB0
  features 128 chiều được project lên 512 chiều trước khi gated fusion. Head
  phân loại dùng `512 -> 256 -> 3`.
- Train projection/fusion layers, classifier head và tail CNN được chọn;
  PhoWhisper encoder vẫn frozen. Đặt `CNN_TRAINABLE_LAYERS=0` để chạy lại
  ablation EfficientNetB0 local branch frozen hoàn toàn.
- Tập trung phân tích Central recall/F1 và lỗi Central -> Northern,
  Central -> Southern.
- Không claim hometown, identity, ethnicity hoặc personal background.

**Expected outputs**

- `configs/experiments/e7_whisper_cnn_fusion.yaml`.
- `src/models/whisper_cnn_fusion.py`.
- `src/training/train_e7_whisper_cnn_fusion.py`.
- `scripts/train_e7_whisper_cnn_fusion_mps.sh`.
- `outputs/models/e7_whisper_cnn_fusion.pt`.
- `outputs/metrics/e7_whisper_cnn_fusion_results.json`.
- `outputs/metrics/e7_whisper_cnn_fusion_training_log.csv`.
- `outputs/metrics/e7_whisper_cnn_fusion_valid_confusion_matrix.csv`.
- `outputs/metrics/e7_whisper_cnn_fusion_test_confusion_matrix.csv`.
- `outputs/reports/phase10_whisper_cnn_fusion_report.md`.
- Updated `outputs/metrics/model_method_comparison.csv` after E7 training.

**Comparison targets**

- MobileNetV3-style / CNN log-Mel baseline.
- PhoWhisper-base frozen baseline.
- Frozen `openai/whisper-base` baseline.
- Vietnamese wav2vec2 frozen-embedding baseline.

**Interpretation criteria**

- If E7 improves Central recall/F1 over frozen PhoWhisper-base, that supports the
  hypothesis that local CNN features add complementary dialect cues.
- If E7 does not improve over PhoWhisper, report that the pretrained global
  representations may already capture the useful signal and that fusion adds
  latency/complexity.



### Phase 11: Residual-Gated PhoWhisper + CNN Fusion Experiment

**Scope**

- Thực nghiệm E8 như một biến thể cải tiến của E7, vẫn chỉ phân loại 3 nhãn
  `Northern`, `Central`, `Southern`.
- Giữ frozen `vinai/PhoWhisper-base` encoder làm global branch chính. Global
  embedding là mean-pooled hidden state 512 chiều.
- Local branch dùng E2 EfficientNetB0-style log-Mel checkpoint. Mặc định
  fine-tune nhẹ 2 child module có tham số cuối với `cnn_learning_rate=1e-5`.
- Fusion mặc định là residual-gated:
  `z = g + beta * sigmoid(W[g;P(l)] + b) * P(l)`.
- `P(l)` project local embedding `128 -> 512` bằng `LayerNorm(128)` và
  `Linear(128,512)`; final projection không dùng ReLU để residual có thể cộng
  hoặc trừ theo từng feature.
- `beta` là learnable scalar, khởi tạo `0.1`. Khi `beta=0`, fused embedding
  giảm đúng về PhoWhisper global embedding.
- Classifier dùng projector/classifier của PhoWhisper baseline:
  `Linear(512,256) -> Linear(256,3)`, warm-start từ
  `outputs/models/phowhisper_pretrained_frozen_encoder.pt` nếu có.
- Batch size mặc định là `14` cho Apple MPS full-RAM run. Không dùng dropout
  mặc định.
- Giữ `concat` và legacy `gated` làm ablation nếu cần.

**Expected outputs**

- `configs/experiments/e8_whisper_cnn_residual_fusion.yaml`.
- `src/training/train_e8_whisper_cnn_residual_fusion.py`.
- `scripts/train_e8_whisper_cnn_residual_fusion_mps.sh`.
- `outputs/models/e8_whisper_cnn_residual_fusion.pt`.
- `outputs/metrics/e8_whisper_cnn_residual_fusion_results.json`.
- `outputs/metrics/e8_whisper_cnn_residual_fusion_training_log.csv`.
- `outputs/metrics/e8_whisper_cnn_residual_fusion_valid_confusion_matrix.csv`.
- `outputs/metrics/e8_whisper_cnn_residual_fusion_test_confusion_matrix.csv`.
- `outputs/reports/phase11_whisper_cnn_residual_fusion_report.md`.
- Updated `outputs/metrics/model_method_comparison.csv` after E8 training.

**Diagnostics to record**

- Fusion type, beta init, beta learned.
- Overall residual gate mean and mean gate value by true dialect class.
- Trainable parameter counts.
- CNN fine-tuning child module names and CNN learning rate.
- PhoWhisper head warm-start metadata.

**Interpretation criteria**

- E8 is useful only if it improves validation macro F1 or Central F1/recall over
  E4/E7 without adding too much latency.
- If E8 underperforms E4, report that the residual local branch did not add
  enough complementary signal for this dataset/run.



### Phase 12: Final report, reproducibility and cleanup

**Scope**

- Hoàn thiện tài liệu, final report và hướng dẫn chạy.
- Dọn repo để người khác có thể reproduce kết quả.
- Không mở thêm experiment lớn.
- Không thêm feature mới vào app trừ khi sửa lỗi nhỏ.

**Expected outputs**

- Final README.
- Final report.
- Reproducible commands cho preprocessing, training, evaluation và app.
- Final metrics table.
- Saved config và split file.
- Clean `.gitignore`.
- Optional demo screenshots.

### MVP Scope

Nếu thời gian bị giới hạn, MVP chỉ gồm:

**Scope**

- Metadata preparation.
- Audio preprocessing.
- MFCC + Logistic Regression baseline.
- MFCC + SVM baseline nếu kịp.
- Lightweight CNN.
- Evaluation bằng accuracy, macro F1 và confusion matrix.
- Streamlit upload demo với best available model.

**Expected outputs**

- Clean metadata.
- Preprocessing pipeline.
- Ít nhất một traditional baseline.
- Một CNN model.
- Final comparison table.
- Confusion matrix.
- Web app upload audio và predict được.
- Short final report.

**Out of scope for MVP**

- Full PhoWhisper fine-tuning.
- ONNX export.
- Microphone recording.
- Province-level dialect classification.
- Speaker identity or hometown prediction.
- Production deployment.

### Stretch Scope

Các phần chỉ làm nếu MVP đã hoàn thành:

**Scope**

- PhoWhisper partial fine-tuning.
- ONNX export.
- CPU inference optimization.
- Microphone recording in web app.
- More detailed error dashboard.
- Data augmentation experiments.

**Expected outputs**

- ONNX model nếu export thành công.
- Latency comparison before and after optimization.
- Microphone recording support nếu ổn định.
- Additional report section for stretch experiments.
