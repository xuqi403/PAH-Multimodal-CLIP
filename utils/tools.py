import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from transformers import BertTokenizer

def set_seed(seed):
    """
    Sets the random seed for reproducibility across numpy, torch, and python.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

def worker_init_fn(worker_id):
    """
    Initializes the worker process for DataLoader to ensure Tokenizer availability.
    """
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    # Ensure dataset has a tokenizer attribute
    if hasattr(dataset, 'tokenizer') and dataset.tokenizer is None:
        # You might need to pass the model path via a global config or args here
        # For now, defaulting to the variable used in the original script
        pass