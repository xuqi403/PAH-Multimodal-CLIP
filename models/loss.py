import torch
import torch.nn as nn

def clip_loss(video_features, text_features, temperature=0.07):
    """
    Computes the contrastive loss between video and text features.
    
    Args:
        video_features (Tensor): [batch_size, feature_dim]
        text_features (Tensor): [batch_size, feature_dim]
        temperature (float): Temperature parameter for scaling logits.
        
    Returns:
        Tensor: The calculated symmetric CLIP loss.
    """
    # logits: [batch_size, batch_size]
    logits = (video_features @ text_features.T) / temperature
    
    labels = torch.arange(len(video_features)).to(video_features.device)
    
    loss_v2t = nn.CrossEntropyLoss()(logits, labels)
    loss_t2v = nn.CrossEntropyLoss()(logits.T, labels)
    
    return (loss_v2t + loss_t2v) / 2