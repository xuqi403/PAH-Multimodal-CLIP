import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from torchvision.models.video import r3d_18, R3D_18_Weights

class TextEncoder(nn.Module):
    def __init__(self, pretrained_model_path):
        super(TextEncoder, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_path)
        self.bert = BertModel.from_pretrained(pretrained_model_path)
        self.fc = nn.Linear(self.bert.config.hidden_size, 512)
        # self.classifier = nn.Linear(512, 3) # Optional: text-only classification head

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # Extract [CLS] token
        text_features = self.fc(cls_output)
        # Normalize features for cosine similarity
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        logits = None # self.classifier(text_features)
        return text_features, logits

class VideoEncoder(nn.Module):
    def __init__(self, num_views, num_modalities, num_clinical_features=10):
        super(VideoEncoder, self).__init__()
        # Load R3D-18 Backbone
        self.resnet3d = r3d_18(weights=R3D_18_Weights.DEFAULT)
        self.feature_dim = 512
        # Modify the final fully connected layer of ResNet3D
        self.resnet3d.fc = nn.Linear(self.resnet3d.fc.in_features, self.feature_dim)

        # Embeddings for discrete metadata (View and Modality)
        self.view_embedding = nn.Embedding(num_views, self.feature_dim)
        self.modality_embedding = nn.Embedding(num_modalities, self.feature_dim)

        # MLP for processing continuous/categorical clinical features
        self.mlp_clinical = nn.Sequential(
            nn.Linear(num_clinical_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32)
        )

        # Fusion Layer: 512 (Video) + 512 (View) + 512 (Mod) + 32 (Clinical) = 1568
        self.fc_combined = nn.Linear(512 + 512 + 512 + 32, 512)

    def forward(self, video, view, modality, clinical_features):
        video_features = self.resnet3d(video) # [batch, 512]
        video_features = video_features / video_features.norm(dim=1, keepdim=True)

        view_embeds = self.view_embedding(view)
        modality_embeds = self.modality_embedding(modality)

        clinical_features = self.mlp_clinical(clinical_features) # [batch, 32]

        # Concatenate all features
        combined_features = torch.cat([video_features, view_embeds, modality_embeds, clinical_features], dim=1)
        
        # Project back to shared embedding space
        combined_features = self.fc_combined(combined_features)  # [batch, 512]

        return combined_features

class CLIPModel(nn.Module):
    def __init__(self, num_classes=3, num_views=3, num_modalities=2, num_clinical_features=10, bert_path='bert-base-chinese'):
        super(CLIPModel, self).__init__()
        self.text_encoder = TextEncoder(pretrained_model_path=bert_path)
        self.video_encoder = VideoEncoder(num_views, num_modalities, num_clinical_features)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, video, input_ids, attention_mask, view, modality, clinical_features):
        video_features = self.video_encoder(video, view, modality, clinical_features)
        text_features, _ = self.text_encoder(input_ids, attention_mask)

        # NaN checks for stability
        if torch.isnan(video_features).any(): raise ValueError("NaN in video features")
        if torch.isnan(text_features).any(): raise ValueError("NaN in text features")
        
        # Final classification logits based on video features
        logits = self.classifier(video_features)
        
        return video_features, text_features, logits