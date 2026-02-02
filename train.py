import os
import copy
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score
from utils.visualization import save_confusion_matrix, save_roc_curve, save_metrics, save_loss_acc_curves
from models.loss import clip_loss

def calculate_performance(labels, preds, views, modalities):
    """Calculates accuracy per view and per modality for weighting."""
    view_performance = {}
    mod_performance = {}
    for view in np.unique(views):
        mask = views == view
        acc = accuracy_score(labels[mask], preds[mask]) if mask.sum() > 0 else 0.0
        view_performance[view] = acc
    for mod in np.unique(modalities):
        mask = modalities == mod
        acc = accuracy_score(labels[mask], preds[mask]) if mask.sum() > 0 else 0.0
        mod_performance[mod] = acc
    return view_performance, mod_performance

def patient_level_prediction(report_ids, probs, views, modalities, view_performance, mod_performance):
    """Aggregates video-level predictions into patient-level predictions."""
    unique_reports = np.unique(report_ids)
    patient_preds = []
    patient_probs = []
    
    for report in unique_reports:
        mask = report_ids == report
        patient_probs_current = probs[mask]
        patient_views = views[mask]
        patient_mods = modalities[mask]

        weights = []
        for i, (v, m) in enumerate(zip(patient_views, patient_mods)):
            # Weight = Confidence * View_Acc * Modality_Acc
            prob = np.max(patient_probs_current[i])
            weight = prob * view_performance.get(v, 0.0) * mod_performance.get(m, 0.0)
            weights.append(weight)
        weights = np.array(weights)

        if np.sum(weights) == 0:
            weighted_probs = np.mean(patient_probs_current, axis=0)
        else:
            weighted_probs = np.sum(patient_probs_current * weights[:, None], axis=0) / np.sum(weights)

        patient_preds.append(np.argmax(weighted_probs))
        patient_probs.append(weighted_probs)
        
    return np.array(patient_preds), np.array(patient_probs)

def train_and_evaluate(model, train_loader, val_loader, optimizer, scheduler, num_epochs, device, args):
    """
    Main training routine.
    args: Namespace object containing lambda_clip, lambda_cls, fig_dir, etc.
    """
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    
    # History tracking
    train_loss_history, val_loss_history = [], []
    train_acc_history, val_acc_history = [], []

    # Setup directories
    if not os.path.exists(args.fig_dir):
        os.makedirs(args.fig_dir)

    label_names = ['Normal', 'Mild', 'Severe']

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                data_loader = train_loader
            else:
                model.eval()
                data_loader = val_loader

            running_total_loss = 0.0
            running_clip_loss = 0.0
            running_video_cls_loss = 0.0
            running_corrects = 0
            total_samples = 0

            # Storage for validation analysis
            if phase == 'val':
                epoch_data = {'ids': [], 'views': [], 'mods': [], 'labels': [], 'probs': [], 'preds': [], 'vid_feats': [], 'txt_feats': []}

            if data_loader is None: continue

            for batch in tqdm(data_loader, desc=f"{phase} Batch"):
                if batch is None: continue
                
                videos, labels, _, encoded_inputs, views, modalities, clinical_features, report_ids = batch

                # Move to device
                videos = videos.to(device)
                input_ids = encoded_inputs['input_ids'].squeeze(1).to(device)
                attention_mask = encoded_inputs['attention_mask'].squeeze(1).to(device)
                labels = labels.long().to(device)
                views = views.to(device).long()
                modalities = modalities.to(device).long()
                clinical_features = clinical_features.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    video_features, text_features, video_logits = model(
                        videos, input_ids, attention_mask, views, modalities, clinical_features
                    )

                    # Calculate Losses
                    loss_contrastive = clip_loss(video_features, text_features)
                    loss_cls = nn.CrossEntropyLoss()(video_logits, labels)
                    
                    # --- FORMULA UPDATE ---
                    # Coefficients are passed via args (derived from Grid Search)
                    total_loss = args.lambda_clip * loss_contrastive + args.lambda_cls * loss_cls

                    if phase == 'train':
                        total_loss.backward()
                        optimizer.step()

                    # Metrics
                    preds = video_logits.argmax(dim=1)
                    probs = torch.nn.functional.softmax(video_logits, dim=1)
                    
                    batch_size = videos.size(0)
                    running_corrects += torch.sum(preds == labels)
                    total_samples += batch_size
                    running_total_loss += total_loss.item() * batch_size
                    running_clip_loss += loss_contrastive.item() * batch_size
                    running_video_cls_loss += loss_cls.item() * batch_size

                    # Collect Validation Data
                    if phase == 'val':
                        epoch_data['ids'].extend(report_ids)
                        epoch_data['views'].extend(views.cpu().numpy())
                        epoch_data['mods'].extend(modalities.cpu().numpy())
                        epoch_data['labels'].extend(labels.cpu().numpy())
                        epoch_data['probs'].extend(probs.detach().cpu().numpy())
                        epoch_data['preds'].extend(preds.detach().cpu().numpy())
                        epoch_data['vid_feats'].append(video_features.detach().cpu())
                        epoch_data['txt_feats'].append(text_features.detach().cpu())

            # Epoch Stats
            epoch_total_loss = running_total_loss / total_samples if total_samples > 0 else 0
            epoch_acc = running_corrects.double() / total_samples if total_samples > 0 else 0
            
            print(f'{phase} Total Loss: {epoch_total_loss:.4f} | Acc: {epoch_acc:.4f}')

            if phase == 'train':
                train_acc_history.append(epoch_acc.item())
                train_loss_history.append(epoch_total_loss)
            else:
                val_acc_history.append(epoch_acc.item())
                val_loss_history.append(epoch_total_loss)
                scheduler.step(epoch_total_loss)

                if epoch_total_loss < best_loss:
                    best_loss = epoch_total_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                
                # --- Validation Visualization & Patient Level Logic ---
                if len(epoch_data['vid_feats']) > 0:
                    # 1. Video Level Metrics
                    all_labels = np.array(epoch_data['labels'])
                    all_preds = np.array(epoch_data['preds'])
                    all_probs = np.array(epoch_data['probs'])
                    
                    save_confusion_matrix(confusion_matrix(all_labels, all_preds), epoch, args.fig_dir, prefix='video')
                    save_roc_curve(all_labels, all_probs, epoch, args.fig_dir, label_names, prefix='video')

                    # 2. Patient Level Logic
                    # Convert lists to numpy for indexing
                    e_ids = np.array(epoch_data['ids'])
                    e_views = np.array(epoch_data['views'])
                    e_mods = np.array(epoch_data['mods'])
                    
                    view_perf, mod_perf = calculate_performance(all_labels, all_preds, e_views, e_mods)
                    
                    pat_preds, pat_probs = patient_level_prediction(
                        e_ids, all_probs, e_views, e_mods, view_perf, mod_perf
                    )
                    
                    # Get ground truth for patients
                    unique_reports = np.unique(e_ids)
                    pat_labels = np.array([all_labels[np.where(e_ids == r)[0][0]] for r in unique_reports])
                    
                    pat_acc = accuracy_score(pat_labels, pat_preds)
                    print(f'Epoch {epoch + 1} Patient-level Accuracy: {pat_acc:.4f}')
                    
                    save_confusion_matrix(confusion_matrix(pat_labels, pat_preds), epoch, args.fig_dir, prefix='patient')
                    save_metrics(pat_labels, pat_preds, pat_probs, epoch, args.fig_dir, label_names, prefix='patient')

        # Save Checkpoint
        save_loss_acc_curves(train_loss_history, val_loss_history, train_acc_history, val_acc_history, epoch, args.fig_dir)
        torch.save(model.state_dict(), os.path.join(args.fig_dir, f"clip_model_epoch_{epoch + 1}.pth"))

    print(f'Best validation loss: {best_loss:.4f}')
    model.load_state_dict(best_model_wts)
    return best_loss