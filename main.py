import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from transformers import BertTokenizer

from utils.tools import set_seed, worker_init_fn
from data.dataset import VideoDataset, custom_collate_fn
from models.clip_module import CLIPModel
from train import train_and_evaluate

def parse_args():
    parser = argparse.ArgumentParser(description="PAH Multimodal CLIP Training")
    
    # Grid Search Hyperparameters
    parser.add_argument('--lambda_clip', type=float, default=0.38, help='Weight coefficient for CLIP contrastive loss')
    parser.add_argument('--lambda_cls', type=float, default=1.42, help='Weight coefficient for Classification loss')
    
    # Training Hyperparameters
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--seed', type=int, default=42)
    
    # Paths
    parser.add_argument('--train_path', type=str, default='RHF_train1_text_encoder.csv')
    parser.add_argument('--val_path', type=str, default='RHF_val1_text_encoder.csv')
    parser.add_argument('--bert_path', type=str, default='D:\\PAH\\models\\chinese-bert-wwm-ext')
    parser.add_argument('--fig_dir', type=str, default='fig', help='Directory to save figures and checkpoints')
    
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Handle CUDA memory allocation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    
    # 1. Load Data
    print("Loading Data...")
    train_df = pd.read_csv(args.train_path, encoding='utf-8-sig')
    val_df = pd.read_csv(args.val_path, encoding='utf-8-sig')

    # Basic Preprocessing & Filtering
    # Ensure VideoPath column exists and filter invalid paths
    for df in [train_df, val_df]:
        if 'VideoPath' not in df.columns: df['VideoPath'] = df.iloc[:, 0]
        # Filter for existing video files (mp4/avi)
        df.drop(df[~df['VideoPath'].apply(lambda x: os.path.isfile(x) and x.lower().endswith(('mp4', 'avi')))].index, inplace=True)

    # Filter Views and Modalities
    target_views = [1, 2, 3, 6, 7]
    target_mods = [0, 1]
    train_df = train_df[train_df['View'].isin(target_views) & train_df['CDFILabel'].isin(target_mods)]
    val_df = val_df[val_df['View'].isin(target_views) & val_df['CDFILabel'].isin(target_mods)]

    print(f"Train Samples: {len(train_df)} | Val Samples: {len(val_df)}")

    # 2. Setup Transformations and Scalers
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
    ])

    scaler = StandardScaler()
    scaler.fit(train_df['Age'].values.reshape(-1, 1))

    # Mappings (Dynamic based on training data)
    unique_views = sorted(train_df['View'].unique())
    view_to_idx = {v: i for i, v in enumerate(unique_views)}
    
    unique_mods = sorted(train_df['CDFILabel'].unique())
    modality_to_idx = {v: i for i, v in enumerate(unique_mods)}

    # 3. Datasets & Loaders
    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    
    train_dataset = VideoDataset(train_df, transform=transform, tokenizer=tokenizer, 
                                 view_to_idx=view_to_idx, modality_to_idx=modality_to_idx, 
                                 scaler=scaler, bert_path=args.bert_path)
    
    val_dataset = VideoDataset(val_df, transform=transform, tokenizer=tokenizer, 
                               view_to_idx=view_to_idx, modality_to_idx=modality_to_idx, 
                               scaler=scaler, bert_path=args.bert_path)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=2, collate_fn=custom_collate_fn, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=2, collate_fn=custom_collate_fn, worker_init_fn=worker_init_fn)

    # 4. Model Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel(num_classes=3, 
                      num_views=len(unique_views), 
                      num_modalities=len(unique_mods),
                      bert_path=args.bert_path)
    
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    # 5. Start Training
    print(f"Starting training with loss weights: CLIP={args.lambda_clip}, CLS={args.lambda_cls}")
    train_and_evaluate(model, train_loader, val_loader, optimizer, scheduler, args.epochs, device, args)

if __name__ == "__main__":
    import torch.multiprocessing
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()