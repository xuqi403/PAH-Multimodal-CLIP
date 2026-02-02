import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, default_collate
from decord import VideoReader, cpu
from PIL import Image
from transformers import BertTokenizer

class VideoDataset(Dataset):
    def __init__(self, df, transform=None, num_frames=20, tokenizer=None, 
                 view_to_idx=None, modality_to_idx=None, scaler=None, bert_path='bert-base-chinese'):
        """
        Args:
            df (pd.DataFrame): Dataframe containing video paths and labels.
            transform (callable, optional): Transform to be applied on a sample.
            num_frames (int): Number of frames to sample from the video.
            tokenizer (BertTokenizer): Pre-initialized tokenizer.
            view_to_idx (dict): Mapping from view ID to index.
            modality_to_idx (dict): Mapping from modality ID to index.
            scaler (StandardScaler): Scaler for clinical features (e.g., Age).
            bert_path (str): Path to pretrained BERT model.
        """
        self.video_paths = df['VideoPath'].tolist()
        self.labels = df['PHdegree'].tolist()
        self.gray_texts = df['gray_text'].tolist()
        self.color_texts = df['color_text'].tolist()
        self.raw_modalities = df['CDFILabel'].tolist()  # 0: gray-scale, 1: color
        self.raw_views = df['View'].tolist()            # Original view values (e.g., 2, 6, 7)
        self.report_ids = df['报告ID'].tolist()           # Report/Patient IDs
        self.transform = transform
        self.num_frames = num_frames
        
        if tokenizer:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = BertTokenizer.from_pretrained(bert_path)

        self.view_to_idx = view_to_idx
        self.modality_to_idx = modality_to_idx

        # Extract clinical data
        self.clinical_data = df[['Gender', 'Age', 'PAH', 
                                 'Left_heart_disease', 'Lung_disease', 'CTEPH', 
                                 'Causes_Summary', 'Hypertension', 
                                 'Diabetes', 'Hyperlipidemia']].values

        # Standardize continuous variables (e.g., Age at index 1)
        if scaler is not None:
            # Flatten is important to keep dimensions consistent
            self.clinical_data[:, 1] = scaler.transform(self.clinical_data[:, 1].reshape(-1, 1)).flatten()

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        raw_view = self.raw_views[idx]
        raw_modality = self.raw_modalities[idx]
        report_id = self.report_ids[idx]

        # Combine texts based on modality (Logic can be adjusted)
        text = str(self.gray_texts[idx]) + '\n' + str(self.color_texts[idx])

        frames = []
        try:
            # Load video using Decord
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)
            if total_frames <= 0:
                raise ValueError(f"Video has no frames: {video_path}")

            # Uniform temporal sampling
            frame_idxs = np.linspace(0, total_frames - 1, self.num_frames).astype(int)
            for frame_idx in frame_idxs:
                frame = vr[frame_idx].asnumpy()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                if self.transform:
                    frame = self.transform(frame)
                frames.append(frame)

            # Stack frames: [C, T, H, W]
            frames = torch.stack(frames, dim=1)

            # Encode text
            encoded_inputs = self.tokenizer(
                text,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
                max_length=256
            )

            # Map View and Modality to indices
            view_idx = torch.tensor(self.view_to_idx[raw_view], dtype=torch.long)
            mod_idx = torch.tensor(raw_modality, dtype=torch.long)

            # Prepare clinical features
            clinical_features = torch.tensor(self.clinical_data[idx], dtype=torch.float32)

            # Safety check for NaNs in clinical data
            if np.isnan(self.clinical_data[idx]).any():
                return None
            
            return frames, label, video_path, encoded_inputs, view_idx, mod_idx, clinical_features, report_id

        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            return None

def custom_collate_fn(batch):
    """
    Filters out None samples (failed loads) before collation.
    """
    # Ensure sample is not None and has correct frame shape
    batch = [b for b in batch if b is not None and b[0].shape[0] == 3] 
    if len(batch) == 0:
        return None
    return default_collate(batch)