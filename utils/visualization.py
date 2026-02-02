import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_score, recall_score, 
    f1_score, accuracy_score, roc_auc_score, precision_recall_curve, 
    average_precision_score
)
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Set font configurations for matplotlib
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

def save_confusion_matrix(cm, epoch, fig_dir, prefix='', label_names=None):
    """
    Plots and saves the confusion matrix heatmap.
    """
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    if label_names is None:
        label_names = ['Normal', 'Mild', 'Severe']

    # Calculate percentages
    cm_sum = cm.sum(axis=1, keepdims=True)
    cm_percentage = cm.astype('float') / cm_sum * 100
    cm_percentage = np.nan_to_num(cm_percentage) 

    plt.figure(figsize=(10, 8))

    # Plot heatmap
    sns.heatmap(cm_percentage, annot=False, fmt='.1f', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names,
                cbar_kws={'label': 'Percentage (%)'})

    # Add counts and percentage text
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            percentage = cm_percentage[i, j]
            text_color = 'white' if percentage > 50 else 'black'
            plt.text(j + 0.5, i + 0.5, f'{int(count)}\n({percentage:.1f}%)',
                     ha='center', va='center', color=text_color, fontsize=24)

    plt.xlabel('Predicted', fontsize=22)
    plt.ylabel('Actual', fontsize=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    filename = f'{prefix}_confusion_matrix_epoch_{epoch + 1}.png' if prefix else f'confusion_matrix_epoch_{epoch + 1}.png'
    plt.savefig(os.path.join(fig_dir, filename), dpi=600, bbox_inches='tight')
    plt.close()
    print(f'Confusion matrix saved to {os.path.join(fig_dir, filename)}')

def save_roc_curve(labels, probs, epoch, fig_dir, label_names, prefix='', n_bootstrap=1000, ci=95):
    """
    Saves ROC curve with AUC and its 95% Confidence Interval.
    """
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    # Define colors dynamically based on number of classes
    label_colors = ['#1E88E5', '#D81B60', '#FFC107', '#004D40', '#D84315'][:len(label_names)]

    labels = np.array(labels)
    probs = np.array(probs)
    n_classes = len(label_names)

    # Binarize labels
    labels_binarized = label_binarize(labels, classes=range(n_classes))

    # Bootstrap for AUC CI
    auc_boot = {i: [] for i in range(n_classes)}

    for b in range(n_bootstrap):
        boot_idx = np.random.randint(0, len(labels), size=len(labels))
        boot_labels = labels_binarized[boot_idx]
        boot_probs  = probs[boot_idx]
        for i in range(n_classes):
            # Handle cases where a class might be missing in the bootstrap sample
            if len(np.unique(boot_labels[:, i])) < 2:
                continue
            fpr_b, tpr_b, _ = roc_curve(boot_labels[:, i], boot_probs[:, i])
            auc_boot[i].append(auc(fpr_b, tpr_b))

    # Calculate CI
    auc_mean, auc_lower, auc_upper = {}, {}, {}
    lower_pct = (100 - ci) / 2
    upper_pct = 100 - lower_pct
    
    for i in range(n_classes):
        if len(auc_boot[i]) == 0:
            auc_mean[i], auc_lower[i], auc_upper[i] = 0.0, 0.0, 0.0
        else:
            arr = np.array(auc_boot[i])
            auc_mean[i]  = arr.mean()
            auc_lower[i] = np.percentile(arr, lower_pct)
            auc_upper[i] = np.percentile(arr, upper_pct)

    # Plot ROC
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(labels_binarized[:, i], probs[:, i])
        plt.plot(
            fpr, tpr,
            color=label_colors[i],
            lw=2,
            label=(
                f"{label_names[i]} "
                f"(AUC = {auc_mean[i]:.2f} "
                f"[{auc_lower[i]:.2f}-{auc_upper[i]:.2f}])"
            )
        )

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    plt.plot([0, 1], [0, 1], linestyle='--', alpha=0.5)
    plt.xlabel('False Positive Rate', fontsize=26)
    plt.ylabel('True Positive Rate', fontsize=26)
    plt.tick_params(labelsize=16)
    plt.legend(fontsize=20, loc='lower right', frameon=False)

    fname = f"{prefix}_roc_epoch_{epoch+1}_with_auc_ci.png" if prefix else f"roc_epoch_{epoch+1}_with_auc_ci.png"
    plt.savefig(os.path.join(fig_dir, fname), dpi=600, bbox_inches='tight')
    plt.close()
    print(f"ROC curve saved to {os.path.join(fig_dir, fname)}")

def save_metrics(y_true, y_pred, y_probs, epoch, fig_dir, label_names, prefix='', n_bootstrap=1000, ci=95):
    """
    Saves evaluation metrics (Precision, Recall, F1, Accuracy, AUC) with 95% CI to a CSV file.
    """
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)
    n_samples = len(y_true)
    n_classes = len(label_names)

    # Point estimates
    prec0 = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec0  = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f10   = f1_score(y_true, y_pred, average='macro', zero_division=0)
    acc0  = accuracy_score(y_true, y_pred)
    try:
        auc0 = roc_auc_score(
            label_binarize(y_true, classes=range(n_classes)),
            y_probs,
            average='macro',
            multi_class='ovr'
        )
    except ValueError:
        auc0 = np.nan

    # Bootstrap
    prec_boot, rec_boot, f1_boot, acc_boot, auc_boot = [], [], [], [], []

    for _ in range(n_bootstrap):
        idx = np.random.randint(0, n_samples, size=n_samples)
        yt  = y_true[idx]
        yp  = y_pred[idx]
        ypr = y_probs[idx]

        prec_boot.append(precision_score(yt, yp, average='macro', zero_division=0))
        rec_boot.append(recall_score(yt, yp, average='macro', zero_division=0))
        f1_boot.append(f1_score(yt, yp, average='macro', zero_division=0))
        acc_boot.append(accuracy_score(yt, yp))
        try:
            auc_boot.append(roc_auc_score(
                label_binarize(yt, classes=range(n_classes)),
                ypr,
                average='macro',
                multi_class='ovr'
            ))
        except ValueError:
            auc_boot.append(np.nan)

    # Helper for CI
    lower_pct = (100 - ci) / 2
    upper_pct = 100 - lower_pct

    def ci_bounds(arr):
        arr = np.array(arr)
        return np.nanpercentile(arr, lower_pct), np.nanpercentile(arr, upper_pct)

    prec_ci = ci_bounds(prec_boot)
    rec_ci  = ci_bounds(rec_boot)
    f1_ci   = ci_bounds(f1_boot)
    acc_ci  = ci_bounds(acc_boot)
    auc_ci  = ci_bounds(auc_boot)

    # Create DataFrame
    metrics = pd.DataFrame({
        'Epoch':       [epoch + 1],
        'Precision':   [f"{prec0:.3f} [{prec_ci[0]:.3f}-{prec_ci[1]:.3f}]"],
        'Recall':      [f"{rec0:.3f} [{rec_ci[0]:.3f}-{rec_ci[1]:.3f}]"],
        'F1-Score':    [f"{f10:.3f} [{f1_ci[0]:.3f}-{f1_ci[1]:.3f}]"],
        'Accuracy':    [f"{acc0:.3f} [{acc_ci[0]:.3f}-{acc_ci[1]:.3f}]"],
        'AUC':         [f"{auc0:.3f} [{auc_ci[0]:.3f}-{auc_ci[1]:.3f}]"]
    })

    # Save to CSV
    metrics_file = os.path.join(fig_dir, f'{prefix}_metrics.csv' if prefix else 'metrics.csv')
    if not os.path.exists(metrics_file):
        metrics.to_csv(metrics_file, index=False)
    else:
        metrics.to_csv(metrics_file, mode='a', header=False, index=False)

    print(f"Evaluation metrics with 95% CI saved to {metrics_file}")

def save_loss_acc_curves(train_loss_history, val_loss_history, train_acc_history, val_acc_history, epoch, fig_dir, prefix=''):
    """
    Plots and saves training and validation loss/accuracy curves.
    """
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    epochs = list(range(1, len(train_loss_history) + 1))

    # Loss Curve
    plt.figure(figsize=(10, 8))
    plt.plot(epochs, train_loss_history, label='Train Loss')
    plt.plot(epochs, val_loss_history, label='Validation Loss')
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.title('Loss Curves', fontsize=22)
    plt.legend(fontsize=16)
    filename = f'{prefix}_loss_curve.png' if prefix else 'loss_curve.png'
    plt.savefig(os.path.join(fig_dir, filename), dpi=600, bbox_inches='tight')
    plt.close()

    # Accuracy Curve
    plt.figure(figsize=(10, 8))
    plt.plot(epochs, train_acc_history, label='Train Accuracy')
    plt.plot(epochs, val_acc_history, label='Validation Accuracy')
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)
    plt.title('Accuracy Curves', fontsize=22)
    plt.legend(fontsize=16)
    filename = f'{prefix}_accuracy_curve.png' if prefix else 'accuracy_curve.png'
    plt.savefig(os.path.join(fig_dir, filename), dpi=600, bbox_inches='tight')
    plt.close()
    print(f'Loss and accuracy curves saved to {fig_dir}')

def plot_tsne(video_features, text_features, video_labels, text_labels, epoch, fig_dir, label_names, prefix=''):
    """
    Visualizes Video and Text features using T-SNE.
    """
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
        
    # Define colors
    label_colors = ['#1E88E5', '#D81B60', '#FFC107', '#004D40', '#D84315'][:len(label_names)]

    # Limit samples for T-SNE speed
    max_samples = 3000
    num_video_samples = video_features.shape[0]
    if num_video_samples > max_samples:
        indices = np.random.choice(num_video_samples, size=max_samples, replace=False)
        indices = torch.from_numpy(indices).long().to(video_features.device)
        video_features_sampled = video_features.index_select(0, indices)
        video_labels_sampled = video_labels.index_select(0, indices)
    else:
        video_features_sampled = video_features
        video_labels_sampled = video_labels

    video_labels_np = video_labels_sampled.cpu().numpy()
    text_labels_np = text_labels.cpu().numpy()

    features = torch.cat([video_features_sampled, text_features], dim=0).cpu().numpy()
    labels_combined = np.concatenate([video_labels_np, text_labels_np])
    data_types = np.array(['Video'] * len(video_labels_np) + ['Text'] * len(text_labels_np))

    label_names_dict = {i: name for i, name in enumerate(label_names)}
    labels_names = [label_names_dict[label] for label in labels_combined]

    # PCA then T-SNE
    pca = PCA(n_components=min(50, features.shape[1]))
    features_reduced = pca.fit_transform(features)
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(features_reduced)

    df = pd.DataFrame({
        'Dimension 1': tsne_results[:, 0],
        'Dimension 2': tsne_results[:, 1],
        'Label': labels_names,
        'Data Type': data_types
    })

    palette = dict(zip(label_names, label_colors))
    markers = {'Video': 'o', 'Text': 'X'}

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df,
        x='Dimension 1',
        y='Dimension 2',
        hue='Label',
        style='Data Type',
        markers=markers,
        palette=palette,
        alpha=0.7,
        s=30
    )

    plt.xlabel('Dimension 1', fontsize=14)
    plt.ylabel('Dimension 2', fontsize=14)
    plt.legend(title='Categories', fontsize=12, title_fontsize=14)

    filename = f'{prefix}_tsne_epoch_{epoch + 1}.png' if prefix else f'tsne_epoch_{epoch + 1}.png'
    plt.savefig(os.path.join(fig_dir, filename), dpi=600, bbox_inches='tight')
    plt.close()
    print(f'T-SNE visualization saved to {os.path.join(fig_dir, filename)}')

def save_patient_probs(report_ids, y_true, y_pred, y_probs, fig_dir, label_names, prefix=''):
    """
    Saves patient-level (report-level) predictions and probabilities to CSV.
    """
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    report_ids = np.array(report_ids)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    data = {
        "Report_ID": report_ids,
        "True_PH": y_true,
        "Pred_PH": y_pred
    }
    
    for i, cls_name in enumerate(label_names):
        col_name = f"Prob_{cls_name}"
        data[col_name] = y_probs[:, i]

    df = pd.DataFrame(data)
    filename = f"{prefix}_patient_probs.csv" if prefix else "patient_probs.csv"
    save_path = os.path.join(fig_dir, filename)

    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"Patient-level probabilities saved to {save_path}")