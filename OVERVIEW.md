## 1. Title

**Lightweight Vietnamese Dialect Identification Web Application**

## 2. Background

Vietnamese dialects vary strongly across regions, especially between Northern, Central, and Southern speech. Recent research such as **Multi-Dialect Vietnamese: Task, Dataset, Baseline Models and Challenges** introduced ViMD, a Vietnamese multi-dialect speech dataset covering 63 provincial dialects, which makes dialect identification a suitable speech processing task for a student project.

## 3. Problem Statement

This project aims to build a lightweight system that classifies short Vietnamese speech recordings into regional dialect groups. The initial scope is three classes: **Northern**, **Central**, and **Southern**.

## 4. Objectives

- Build a complete speech processing pipeline from audio input to dialect prediction.
- Compare traditional baselines, a lightweight CNN, and one selected pretrained speech model: **PhoWhisper-base**.
- Keep the modeling scope focused on local inference for three-region classification.
- Deploy the best model as an interactive web application.
- Evaluate both model quality and deployment feasibility.

## 5. Proposed Method

### 5.1 Dataset

Use a Vietnamese dialect speech dataset such as ViMD if accessible. If needed, reduce the original dialect labels into three regional groups: Northern, Central, and Southern.

### 5.2 Preprocessing

- Convert audio to mono.
- Resample to 16 kHz.
- Normalize volume.
- Trim or pad audio to a fixed duration.
- Extract MFCC or log-Mel spectrogram features.

### 5.3 Models

- **Traditional baseline**: MFCC features with Logistic Regression and SVM classifiers.
- **Main CNN model**: lightweight CNN trained from scratch using log-Mel spectrogram input.
- **Modern pretrained model**: use **PhoWhisper-base only** as the selected pretrained speech model. The encoder will be used to extract speech representations, followed by a small classifier for Northern, Central, and Southern dialect prediction.
- **Local deployment candidate**: compare the lightweight CNN and PhoWhisper-base classifier using macro F1-score, model size, and CPU inference latency.
- **Deployment extension**: export the best trained model to ONNX for faster CPU inference.

### 5.4 Web Application

Deploy a simple web app where users can upload or record audio and receive:

- Predicted dialect group.
- Confidence score.
- Top predictions.
- Audio waveform or Mel spectrogram visualization.
- Inference latency.

## 6. Evaluation

The system will be evaluated using:

- Accuracy.
- Macro F1-score.
- Confusion matrix.
- Model size.
- CPU inference latency.
- Qualitative error analysis on confused dialect groups.

## 7. Expected Output

- A trained Vietnamese dialect identification model.
- A short experimental report comparing traditional baselines, the lightweight CNN, and PhoWhisper-base.
- A web application demo for audio upload or microphone recording.
- Deployment analysis showing whether the model is lightweight enough for personal-machine inference.

## 8. Scope and Limitations

This project does not aim to identify a speaker's real hometown or identity. It only predicts regional speech patterns from short audio samples. The first version focuses on three broad regions instead of all 63 provincial dialects to keep the task feasible for a final course project.

## 9. References

### 9.1 Dataset and Vietnamese dialect identification

- Nguyen Van Dinh, Thanh Chi Dang, Luan Thanh Nguyen, and Kiet Van Nguyen. 2024. **Multi-Dialect Vietnamese: Task, Dataset, Baseline Models and Challenges**. EMNLP 2024. https://aclanthology.org/2024.emnlp-main.426/
- ViMD Dataset repository. https://github.com/nguyen-dv/ViMD_Dataset
- ViMD Dataset on Hugging Face. https://huggingface.co/datasets/nguyendv02/ViMD_Dataset

### 9.2 Audio preprocessing and traditional baselines

- Librosa documentation: MFCC feature extraction. https://librosa.org/doc/main/generated/librosa.feature.mfcc.html
- Librosa documentation: Mel spectrogram feature extraction. https://librosa.org/doc/main/generated/librosa.feature.melspectrogram.html
- scikit-learn documentation: Logistic Regression. https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- scikit-learn documentation: Support Vector Classification. https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

### 9.3 Selected pretrained speech model

- Le, Thanh-Thien, Nguyen, Linh The, and Nguyen, Dat Quoc. 2024. **PhoWhisper: Automatic Speech Recognition for Vietnamese**. ICLR 2024 Tiny Papers. https://arxiv.org/abs/2406.02555
- VinAI Research PhoWhisper repository. https://github.com/VinAIResearch/PhoWhisper

### 9.4 Deployment documents

- ONNX Runtime documentation. https://onnxruntime.ai/docs/
- Streamlit documentation. https://docs.streamlit.io/

## Recommended Structure

```markdown
vietnamese-dialect-id/
├── README.md
├── requirements.txt
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── metadata.csv
├── notebooks/
│   └── 01_exploration.ipynb
├── src/
│   ├── data/
│   │   ├── prepare_metadata.py
│   │   └── preprocess_audio.py
│   ├── features/
│   │   ├── mfcc.py
│   │   └── logmel.py
│   ├── models/
│   │   ├── classical.py
│   │   ├── cnn.py
│   │   └── phowhisper.py
│   ├── training/
│   │   ├── train_classical.py
│   │   ├── train_cnn.py
│   │   └── train_phowhisper.py
│   ├── evaluation/
│   │   ├── evaluate.py
│   │   └── plot_confusion_matrix.py
│   ├── inference/
│   │   └── predict.py
│   └── utils/
│       ├── audio.py
│       └── seed.py
├── outputs/
│   ├── models/
│   ├── metrics/
│   ├── figures/
│   └── logs/
└── app/
    └── streamlit_app.py
```