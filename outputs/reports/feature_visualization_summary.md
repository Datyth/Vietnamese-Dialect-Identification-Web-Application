# Feature Visualization Summary

## Purpose

Báo cáo này tổng hợp các audio và hình ảnh đặc trưng được tạo từ mẫu thật trong dataset của project. Mục tiêu là hỗ trợ mục 2.5 của báo cáo: minh họa và phân tích waveform, MFCC, và log-Mel spectrogram bằng ví dụ, hình ảnh, và biểu đồ có thể dùng trong slide thuyết trình.

## Dataset/sample selection

Notebook đọc dữ liệu từ `manifest: data/processed/preprocessed_metadata.csv` và ưu tiên split `train`. Mỗi lớp được chọn tối đa `5` mẫu với `random_state = 42`. Nếu một lớp có ít hơn số mẫu yêu cầu, notebook dùng toàn bộ mẫu hiện có và in cảnh báo.

| sample ID | dialect | original path | exported audio path | duration | sample rate |
| --- | --- | --- | --- | --- | --- |
| train:26_0177.wav | Northern | data/processed/audio_16k/train/Northern/26_0177.wav | outputs/audio/feature_visualization/Northern_sample_01_preprocessed.wav | 18.14 | 16000 |
| train:12_0210.wav | Northern | data/processed/audio_16k/train/Northern/12_0210.wav | outputs/audio/feature_visualization/Northern_sample_02_preprocessed.wav | 22.41 | 16000 |
| train:11_0103.wav | Northern | data/processed/audio_16k/train/Northern/11_0103.wav | outputs/audio/feature_visualization/Northern_sample_03_preprocessed.wav | 13.98 | 16000 |
| train:30_0201.wav | Northern | data/processed/audio_16k/train/Northern/30_0201.wav | outputs/audio/feature_visualization/Northern_sample_04_preprocessed.wav | 23.50 | 16000 |
| train:17_0150.wav | Northern | data/processed/audio_16k/train/Northern/17_0150.wav | outputs/audio/feature_visualization/Northern_sample_05_preprocessed.wav | 18.50 | 16000 |
| train:49_0005.wav | Central | data/processed/audio_16k/train/Central/49_0005.wav | outputs/audio/feature_visualization/Central_sample_01_preprocessed.wav | 19.44 | 16000 |
| train:48_0160.wav | Central | data/processed/audio_16k/train/Central/48_0160.wav | outputs/audio/feature_visualization/Central_sample_02_preprocessed.wav | 30.24 | 16000 |
| train:47_0061.wav | Central | data/processed/audio_16k/train/Central/47_0061.wav | outputs/audio/feature_visualization/Central_sample_03_preprocessed.wav | 17.24 | 16000 |
| train:85_0163.wav | Central | data/processed/audio_16k/train/Central/85_0163.wav | outputs/audio/feature_visualization/Central_sample_04_preprocessed.wav | 15.56 | 16000 |
| train:43_0034.wav | Central | data/processed/audio_16k/train/Central/43_0034.wav | outputs/audio/feature_visualization/Central_sample_05_preprocessed.wav | 19.67 | 16000 |
| train:72_0197.wav | Southern | data/processed/audio_16k/train/Southern/72_0197.wav | outputs/audio/feature_visualization/Southern_sample_01_preprocessed.wav | 20.83 | 16000 |
| train:84_0134.wav | Southern | data/processed/audio_16k/train/Southern/84_0134.wav | outputs/audio/feature_visualization/Southern_sample_02_preprocessed.wav | 18.98 | 16000 |
| train:95_0046.wav | Southern | data/processed/audio_16k/train/Southern/95_0046.wav | outputs/audio/feature_visualization/Southern_sample_03_preprocessed.wav | 19.32 | 16000 |
| train:70_0014.wav | Southern | data/processed/audio_16k/train/Southern/70_0014.wav | outputs/audio/feature_visualization/Southern_sample_04_preprocessed.wav | 8.20 | 16000 |
| train:60_0111.wav | Southern | data/processed/audio_16k/train/Southern/60_0111.wav | outputs/audio/feature_visualization/Southern_sample_05_preprocessed.wav | 17.94 | 16000 |

## Generated audio files

Các file audio đã tiền xử lý dùng để nghe trực tiếp khi so sánh với đặc trưng; các file original giúp kiểm tra khác biệt trước và sau tiền xử lý.

| dialect | sample ID | audio path | purpose |
| --- | --- | --- | --- |
| Northern | train:26_0177.wav | outputs/audio/feature_visualization/Northern_sample_01_preprocessed.wav | Bản audio đã tiền xử lý để nghe cùng waveform/MFCC/log-Mel. |
| Northern | train:26_0177.wav | outputs/audio/feature_visualization/Northern_sample_01_original.wav | Bản audio gốc được export lại để so sánh trước/sau tiền xử lý. |
| Northern | train:12_0210.wav | outputs/audio/feature_visualization/Northern_sample_02_preprocessed.wav | Bản audio đã tiền xử lý để nghe cùng waveform/MFCC/log-Mel. |
| Northern | train:12_0210.wav | outputs/audio/feature_visualization/Northern_sample_02_original.wav | Bản audio gốc được export lại để so sánh trước/sau tiền xử lý. |
| Northern | train:11_0103.wav | outputs/audio/feature_visualization/Northern_sample_03_preprocessed.wav | Bản audio đã tiền xử lý để nghe cùng waveform/MFCC/log-Mel. |
| Northern | train:11_0103.wav | outputs/audio/feature_visualization/Northern_sample_03_original.wav | Bản audio gốc được export lại để so sánh trước/sau tiền xử lý. |
| Northern | train:30_0201.wav | outputs/audio/feature_visualization/Northern_sample_04_preprocessed.wav | Bản audio đã tiền xử lý để nghe cùng waveform/MFCC/log-Mel. |
| Northern | train:30_0201.wav | outputs/audio/feature_visualization/Northern_sample_04_original.wav | Bản audio gốc được export lại để so sánh trước/sau tiền xử lý. |
| Northern | train:17_0150.wav | outputs/audio/feature_visualization/Northern_sample_05_preprocessed.wav | Bản audio đã tiền xử lý để nghe cùng waveform/MFCC/log-Mel. |
| Northern | train:17_0150.wav | outputs/audio/feature_visualization/Northern_sample_05_original.wav | Bản audio gốc được export lại để so sánh trước/sau tiền xử lý. |
| Central | train:49_0005.wav | outputs/audio/feature_visualization/Central_sample_01_preprocessed.wav | Bản audio đã tiền xử lý để nghe cùng waveform/MFCC/log-Mel. |
| Central | train:49_0005.wav | outputs/audio/feature_visualization/Central_sample_01_original.wav | Bản audio gốc được export lại để so sánh trước/sau tiền xử lý. |
| Central | train:48_0160.wav | outputs/audio/feature_visualization/Central_sample_02_preprocessed.wav | Bản audio đã tiền xử lý để nghe cùng waveform/MFCC/log-Mel. |
| Central | train:48_0160.wav | outputs/audio/feature_visualization/Central_sample_02_original.wav | Bản audio gốc được export lại để so sánh trước/sau tiền xử lý. |
| Central | train:47_0061.wav | outputs/audio/feature_visualization/Central_sample_03_preprocessed.wav | Bản audio đã tiền xử lý để nghe cùng waveform/MFCC/log-Mel. |
| Central | train:47_0061.wav | outputs/audio/feature_visualization/Central_sample_03_original.wav | Bản audio gốc được export lại để so sánh trước/sau tiền xử lý. |
| Central | train:85_0163.wav | outputs/audio/feature_visualization/Central_sample_04_preprocessed.wav | Bản audio đã tiền xử lý để nghe cùng waveform/MFCC/log-Mel. |
| Central | train:85_0163.wav | outputs/audio/feature_visualization/Central_sample_04_original.wav | Bản audio gốc được export lại để so sánh trước/sau tiền xử lý. |
| Central | train:43_0034.wav | outputs/audio/feature_visualization/Central_sample_05_preprocessed.wav | Bản audio đã tiền xử lý để nghe cùng waveform/MFCC/log-Mel. |
| Central | train:43_0034.wav | outputs/audio/feature_visualization/Central_sample_05_original.wav | Bản audio gốc được export lại để so sánh trước/sau tiền xử lý. |
| Southern | train:72_0197.wav | outputs/audio/feature_visualization/Southern_sample_01_preprocessed.wav | Bản audio đã tiền xử lý để nghe cùng waveform/MFCC/log-Mel. |
| Southern | train:72_0197.wav | outputs/audio/feature_visualization/Southern_sample_01_original.wav | Bản audio gốc được export lại để so sánh trước/sau tiền xử lý. |
| Southern | train:84_0134.wav | outputs/audio/feature_visualization/Southern_sample_02_preprocessed.wav | Bản audio đã tiền xử lý để nghe cùng waveform/MFCC/log-Mel. |
| Southern | train:84_0134.wav | outputs/audio/feature_visualization/Southern_sample_02_original.wav | Bản audio gốc được export lại để so sánh trước/sau tiền xử lý. |
| Southern | train:95_0046.wav | outputs/audio/feature_visualization/Southern_sample_03_preprocessed.wav | Bản audio đã tiền xử lý để nghe cùng waveform/MFCC/log-Mel. |
| Southern | train:95_0046.wav | outputs/audio/feature_visualization/Southern_sample_03_original.wav | Bản audio gốc được export lại để so sánh trước/sau tiền xử lý. |
| Southern | train:70_0014.wav | outputs/audio/feature_visualization/Southern_sample_04_preprocessed.wav | Bản audio đã tiền xử lý để nghe cùng waveform/MFCC/log-Mel. |
| Southern | train:70_0014.wav | outputs/audio/feature_visualization/Southern_sample_04_original.wav | Bản audio gốc được export lại để so sánh trước/sau tiền xử lý. |
| Southern | train:60_0111.wav | outputs/audio/feature_visualization/Southern_sample_05_preprocessed.wav | Bản audio đã tiền xử lý để nghe cùng waveform/MFCC/log-Mel. |
| Southern | train:60_0111.wav | outputs/audio/feature_visualization/Southern_sample_05_original.wav | Bản audio gốc được export lại để so sánh trước/sau tiền xử lý. |

## Generated figures

| figure name | file path | purpose |
| --- | --- | --- |
| Northern_sample_01_features.png | outputs/figures/feature_visualization/individual_samples/Northern_sample_01_features.png | Waveform, MFCC, and log-Mel panels for Northern sample 01. |
| Northern_sample_02_features.png | outputs/figures/feature_visualization/individual_samples/Northern_sample_02_features.png | Waveform, MFCC, and log-Mel panels for Northern sample 02. |
| Northern_sample_03_features.png | outputs/figures/feature_visualization/individual_samples/Northern_sample_03_features.png | Waveform, MFCC, and log-Mel panels for Northern sample 03. |
| Northern_sample_04_features.png | outputs/figures/feature_visualization/individual_samples/Northern_sample_04_features.png | Waveform, MFCC, and log-Mel panels for Northern sample 04. |
| Northern_sample_05_features.png | outputs/figures/feature_visualization/individual_samples/Northern_sample_05_features.png | Waveform, MFCC, and log-Mel panels for Northern sample 05. |
| Central_sample_01_features.png | outputs/figures/feature_visualization/individual_samples/Central_sample_01_features.png | Waveform, MFCC, and log-Mel panels for Central sample 01. |
| Central_sample_02_features.png | outputs/figures/feature_visualization/individual_samples/Central_sample_02_features.png | Waveform, MFCC, and log-Mel panels for Central sample 02. |
| Central_sample_03_features.png | outputs/figures/feature_visualization/individual_samples/Central_sample_03_features.png | Waveform, MFCC, and log-Mel panels for Central sample 03. |
| Central_sample_04_features.png | outputs/figures/feature_visualization/individual_samples/Central_sample_04_features.png | Waveform, MFCC, and log-Mel panels for Central sample 04. |
| Central_sample_05_features.png | outputs/figures/feature_visualization/individual_samples/Central_sample_05_features.png | Waveform, MFCC, and log-Mel panels for Central sample 05. |
| Southern_sample_01_features.png | outputs/figures/feature_visualization/individual_samples/Southern_sample_01_features.png | Waveform, MFCC, and log-Mel panels for Southern sample 01. |
| Southern_sample_02_features.png | outputs/figures/feature_visualization/individual_samples/Southern_sample_02_features.png | Waveform, MFCC, and log-Mel panels for Southern sample 02. |
| Southern_sample_03_features.png | outputs/figures/feature_visualization/individual_samples/Southern_sample_03_features.png | Waveform, MFCC, and log-Mel panels for Southern sample 03. |
| Southern_sample_04_features.png | outputs/figures/feature_visualization/individual_samples/Southern_sample_04_features.png | Waveform, MFCC, and log-Mel panels for Southern sample 04. |
| Southern_sample_05_features.png | outputs/figures/feature_visualization/individual_samples/Southern_sample_05_features.png | Waveform, MFCC, and log-Mel panels for Southern sample 05. |
| 01_waveform_grid.png | outputs/figures/feature_visualization/01_waveform_grid.png | Compare preprocessed waveform shapes across selected samples and dialect classes. |
| 02_mfcc_heatmap_grid.png | outputs/figures/feature_visualization/02_mfcc_heatmap_grid.png | Compare 13-coefficient MFCC maps across selected dialect samples. |
| 03_logmel_spectrogram_grid.png | outputs/figures/feature_visualization/03_logmel_spectrogram_grid.png | Compare 64-bin standardized log-Mel spectrograms used by the CNN pipeline. |
| 04_mfcc_mean_std_vectors_by_sample.png | outputs/figures/feature_visualization/04_mfcc_mean_std_vectors_by_sample.png | Show each selected sample as the 26-D MFCC mean plus standard deviation vector used by traditional ML models. |
| 05_mfcc_class_average_vector.png | outputs/figures/feature_visualization/05_mfcc_class_average_vector.png | Compare class-level averages of selected 26-D MFCC feature vectors. |
| 06_average_logmel_energy_by_class.png | outputs/figures/feature_visualization/06_average_logmel_energy_by_class.png | Compare average raw log-Mel energy profiles across selected samples for each dialect class. |
| 07_duration_distribution_by_class.png | outputs/figures/feature_visualization/07_duration_distribution_by_class.png | Show original duration distribution by class using metadata from the selected split. |

## Waveform analysis

Waveform biểu diễn biên độ tín hiệu theo thời gian. Nó giúp kiểm tra khoảng lặng, độ lớn tương đối, clipping, và hình dạng tổng quát sau tiền xử lý. Waveform không trực tiếp tiết lộ phương ngữ; các khác biệt quan sát được có thể do câu nói, người nói, micro, nhiễu nền hoặc âm lượng ban đầu.

## MFCC analysis

MFCC tóm tắt bao phổ ngắn hạn của tiếng nói. Project dùng 13 hệ số MFCC, sau đó lấy trung bình và độ lệch chuẩn theo thời gian để tạo vector 26 chiều cho Logistic Regression và SVM. Cách biểu diễn này nhỏ gọn, cố định chiều, và phù hợp với các mô hình truyền thống. Heatmap MFCC giúp quan sát pattern phổ theo thời gian nhưng không đủ để kết luận chắc chắn về phương ngữ.

## Log-Mel spectrogram analysis

Log-Mel spectrogram giữ cấu trúc thời gian - tần số với 64 Mel bins. Đây là biểu diễn phù hợp cho CNN vì CNN có thể học các pattern cục bộ theo cả trục thời gian và trục tần số. Trong hình, vùng sáng hơn thường biểu thị năng lượng tương đối mạnh hơn, nhưng cách đọc vẫn phải thận trọng vì nội dung câu và điều kiện thu có ảnh hưởng lớn.

## Class-level comparison

Các hình class-level cho phép so sánh nhiều mẫu Northern, Central và Southern trong cùng một bố cục. Các đường trung bình MFCC và log-Mel energy giúp nhìn xu hướng chung của nhóm mẫu được chọn. Vì số mẫu chỉ là một phần nhỏ của dataset, các xu hướng này nên được xem là gợi ý khám phá, không phải kết luận thống kê cuối cùng.

## Limitations

- Các biểu đồ chỉ dùng mẫu được chọn để minh họa, không phải toàn bộ dataset.
- Khác biệt hình ảnh có thể đến từ phương ngữ, speaker identity, giới tính, nội dung câu, tốc độ nói, điều kiện thu âm hoặc nhiễu.
- Waveform, MFCC và log-Mel là công cụ phân tích thăm dò; kết luận mô hình cần dựa trên đánh giá định lượng như accuracy, macro F1 và confusion matrix.
- Không dùng các hình này để suy luận danh tính, quê quán cụ thể, hoặc thông tin cá nhân của người nói.

## How to use these figures in the presentation

Có thể dùng waveform grid để giới thiệu dữ liệu audio sau tiền xử lý, MFCC grid để giải thích baseline Logistic Regression/SVM, log-Mel grid để giải thích input của CNN, và các hình vector trung bình để minh họa cách đặc trưng được nén thành dạng mô hình có thể học. Khi trình bày, nên nhấn mạnh rằng đây là minh họa trực quan hỗ trợ hiểu pipeline, không phải bằng chứng cuối cùng về khác biệt phương ngữ.

## Gợi ý lời trình bày

Trong phần này, em minh họa các đặc trưng được trích xuất từ audio thật của ba vùng Northern, Central và Southern. Trước hết, waveform cho thấy biên độ tín hiệu theo thời gian, giúp kiểm tra khoảng lặng, độ dài và độ lớn của âm thanh sau tiền xử lý. Sau đó, MFCC chuyển tín hiệu sang một biểu diễn compact của bao phổ ngắn hạn; vì mỗi audio được nén thành vector 26 chiều gồm trung bình và độ lệch chuẩn của 13 hệ số, đặc trưng này phù hợp với Logistic Regression và SVM. Với CNN, em dùng log-Mel spectrogram vì biểu diễn này giữ lại cấu trúc thời gian - tần số, cho phép CNN học các pattern cục bộ trong tiếng nói. Tuy nhiên, các hình này chỉ dùng để khám phá và giải thích pipeline. Khác biệt thị giác có thể đến từ phương ngữ, người nói, nội dung câu, điều kiện thu hoặc nhiễu, nên kết luận cuối cùng vẫn cần dựa trên kết quả đánh giá định lượng của mô hình.
