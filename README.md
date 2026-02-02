AI-driven Pulmonary Hypertension Screening and Severity Grading using Multimodal Echocardiography
This repository contains the official PyTorch implementation of the paper: "AI-driven Pulmonary Hypertension Screening and Severity Grading using Multimodal Echocardiography".

📋Abstract

Pulmonary hypertension (PH) is a life-threatening condition where early detection is crucial but challenging. Conventional echocardiography (specifically Tricuspid Regurgitation Velocity, TRV) often shows suboptimal accuracy and high rates of missing data .

We present a multimodal AI framework that integrates:

Echocardiographic Videos: Spatio-temporal analysis using 3D-ResNet.

Clinical Data: Patient demographics, medical history, and biomarkers processed via MLP.

Vision-Language Pretraining (CLIP): Utilizes diagnostic reports during training to align visual features with semantic medical knowledge .

⚙️Project Structure

The code is organized into modular components for training, data handling, and evaluation.
Plaintext

├── data/

│   ├── __init__.py

│   └── dataset.py          # Custom VideoDataset, preprocessing, and collate_fn

├── models/

│   ├── __init__.py

│   ├── clip_module.py      # 3D-ResNet Video Encoder, BERT Text Encoder, Fusion MLP

│   └── loss.py             # Combined Contrastive (CLIP) + CrossEntropy Loss

├── utils/

│   ├── __init__.py

│   ├── tools.py            # Seeding and reproducibility tools

│   └── visualization.py    # Metric plotting (ROC, Confusion Matrix, T-SNE, GradCAM)

├── train.py                # Main training and validation loop

├── main.py                 # Entry point with argument parsing

├── requirements.txt        # Python dependencies

└── README.md


🛠️Installation
Prerequisites
Python 3.8+
CUDA-enabled GPU
Install Dependencies
Bash
pip install -r requirements.txt
Core Libraries:
torch, torchvision (Deep Learning)
decord, opencv-python (Video Processing)
transformers (BERT Text Encoding)
pandas, scikit-learn (Data & Metrics)


📊Data Preparation
The model requires a CSV file linking video paths to clinical data and labels.
CSV Format
Your training and validation CSV files (e.g., train.csv, val.csv) must contain the following columns matching the VideoDataset implementation:
Column Name	Description	Type
VideoPath	Absolute path to the .mp4/.avi file	String
PHdegree	Label: 0 (Non-PH), 1 (Mild), 2 (Severe)	Int
View	Standard View ID (mapped internally, e.g., 2, 6, 7)	Int
CDFILabel	Modality: 0 (Gray-scale), 1 (Color Doppler)	Int
gray_text	Text description for B-mode (used for CLIP)	String
color_text	Text description for Color mode (used for CLIP)	String
ID	Report/Patient ID (for patient-level aggregation)	String
Clinical Features
The following columns are required for the Clinical MLP:
Age (Continuous)
Gender, PAH, Left_heart_disease, Lung_disease, CTEPH, Causes_Summary, Hypertension, Diabetes, Hyperlipidemia (Categorical/Binary)


🚀Usage
Training
To train the model, run main.py. You can configure hyperparameters via command-line arguments.
Bash
python main.py \
  --train_path "path/to/train.csv" \
  --val_path "path/to/val.csv" \
  --bert_path "path/to/local/bert-base-chinese" \
  --fig_dir "results/experiment_1" \
  --batch_size 8 \
  --epochs 30 \
  --lr 0.0001 \
  --lambda_clip 0.38 \
  --lambda_cls 1.42
Arguments
--lambda_clip: Weight for the Video-Text contrastive loss (Default: 0.38).
--lambda_cls: Weight for the classification CrossEntropy loss (Default: 1.42).
--bert_path: Path to the pretrained BERT model (e.g., chinese-bert-wwm-ext).

Inference & Evaluation
During validation, the script automatically performs Patient-Level Aggregation. It aggregates predictions from multiple videos (views/modalities) belonging to the same 报告ID using a learned weighted voting mechanism.
Results (Confusion Matrices, ROC Curves, Metrics CSVs) are automatically saved to fig_dir.

🧠Model Architecture
The framework consists of three branches:
Video Encoder: A 3D-ResNet18 (pretrained on Kinetics-400) processes 20-frame video clips to extract spatio-temporal features. View and Modality embeddings are concatenated to the visual features.
Clinical Encoder: A 3-layer MLP processes tabular data (demographics + biomarkers) to capture physiological context.
Text Encoder: A BERT model encodes medical reports. This is used primarily during training to align video features with semantic descriptions via CLIP Loss.
Fusion: Video, Metadata, and Clinical features are concatenated and passed through a final classification head.

📝Citation
If you use this code or dataset in your research, please cite our paper:
@article{Zhou2026PHScreening,
  title={AI-driven Pulmonary Hypertension Screening and Severity Grading using Multimodal Echocardiography},
  author={Zhou, Lin and Xu, Qi and Yang, Yan and Zhao, Qifeng and Wu, Han and Zhao, Shilun and Wu, Xinlei and Ye, Lina and Yan, Zhihan and Shen, Dinggang},
  journal={Journal Name (TBD)},
  year={2026}
}

📧Contact
For questions regarding the code or dataset, please contact:
Zhihan Yan, MD: yanzhihanwz@163.com 
Dinggang Shen, PhD: Dinggang.Shen@gmail.com 
