import torch
import os
import shutil
from torch.utils.tensorboard import SummaryWriter

def save_checkpoint(model, optimizer, fold, epoch, best=False):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    best_model_path = f'checkpoint_foldss{fold}_best.pth'
    torch.save(state, best_model_path)

def setup_logging(log_dir):
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir)
    return SummaryWriter(log_dir)
