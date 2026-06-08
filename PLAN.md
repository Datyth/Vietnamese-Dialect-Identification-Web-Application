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

### Phase 9: Final report, reproducibility and cleanup

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