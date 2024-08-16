import torch
from sklearn.model_selection import StratifiedKFold
from config import load_config
from sklearn.utils.class_weight import compute_class_weight
from config import config
from dataset import FungiDataset, get_transforms
from model import CustomCNN
from training import train_epoch, validate_epoch
from utils import save_checkpoint, setup_logging
import numpy as np
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import torch.nn as nn

def train_model(config):
    device = config["device"]
    dataset = FungiDataset(config["base_dir"], transform=get_transforms(config), subset='train')
    
    # Compute class weights for handling class imbalance
    class_weights = compute_class_weight('balanced', classes=np.arange(len(dataset.classes)), y=dataset.labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Set up logging
    writer = setup_logging(config["log_dir"])
    
    # Use StratifiedKFold to ensure each fold has a similar class distribution
    skf = StratifiedKFold(n_splits=config["num_folds"], shuffle=True, random_state=config["seed"])

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(dataset)), dataset.labels), 1):

        print(f"Fold {fold}/{config['num_folds']}")
        
        # Extract the labels for the train and validation indices
        train_labels = np.array(dataset.labels)[train_idx]
        val_labels = np.array(dataset.labels)[val_idx]
        print(f"Fold {fold} - Train Class Distribution: {np.bincount(train_labels)}")
        print(f"Fold {fold} - Val Class Distribution: {np.bincount(val_labels)}")

        # Set up the data samplers and loaders
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(dataset, batch_size=config["batch_size"], sampler=train_sampler)
        val_loader = DataLoader(dataset, batch_size=config["batch_size"], sampler=val_sampler)

        # Reinitialize the model for each fold
        model = CustomCNN(num_classes=len(dataset.classes), config=config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

        best_val_loss, patience_counter = float('inf'), 0
        best_model_path = f'checkpoint_foldss{fold}_best.pth'

        for epoch in range(1, config["epochs"] + 1):
            print(f"Epoch {epoch}/{config['epochs']}")

            train_loss, train_accuracy = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_accuracy = validate_epoch(model, val_loader, criterion, device)

            print(f"Train Loss: {train_loss:.4f}, Acc: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/train', train_accuracy, epoch)
            writer.add_scalar('Accuracy/val', val_accuracy, epoch)
            writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)

            # Update the learning rate based on validation loss
            scheduler.step(val_loss)

            # Save the model if it has the best validation loss so far
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                print(f"New best model found for fold {fold} at epoch {epoch}, saving model...")
                save_checkpoint(model, optimizer, fold, epoch,best=True)
                # torch.save({
                #     'model_state_dict': model.state_dict(),
                #     'optimizer_state_dict': optimizer.state_dict(),
                #     'epoch': epoch,
                #     'best_val_loss': best_val_loss,
                # }, best_model_path)
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= config["patience"]:
                print("Early stopping triggered")
                break

    writer.close()

if __name__ == "__main__":
    config = load_config()  # Load the configuration from the YAML file
    train_model(config)