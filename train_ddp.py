"""
Distributed Training System for Mask R-CNN PPE Detection
Author: [Dr.Irshad Ibrahim]
Date: 2024-06-15
License: MIT
"""

import os
import time
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import wandb

# Constants
DEFAULT_IMAGE_SIZE = (1024, 1024)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

class PPEDataset(Dataset):
    """Custom dataset for PPE detection with advanced augmentation"""
    
    def __init__(self, image_paths, annotation_paths, transform=None, phase='train'):
        self.image_paths = image_paths
        self.annotation_paths = annotation_paths
        self.transform = transform
        self.phase = phase
        self.logger = logging.getLogger('PPEDataset')
        
        # Validate paths
        if len(image_paths) != len(annotation_paths):
            raise ValueError("Images and annotations must have same length")
            
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            # Load image and annotations (simplified)
            image = np.random.rand(*DEFAULT_IMAGE_SIZE, 3).astype(np.float32)  # Placeholder
            annotations = {
                'boxes': torch.rand(4, 4), 
                'labels': torch.randint(0, 5, (4,)),
                'masks': torch.rand(4, *DEFAULT_IMAGE_SIZE)
            }
            
            if self.transform:
                image = self.transform(image)
                
            return image, annotations
        
        except Exception as e:
            self.logger.error(f"Error loading sample {idx}: {str(e)}")
            # Return empty sample for fault tolerance
            return torch.zeros(3, *DEFAULT_IMAGE_SIZE), {
                'boxes': torch.zeros(0, 4),
                'labels': torch.zeros(0, dtype=torch.int64),
                'masks': torch.zeros(0, *DEFAULT_IMAGE_SIZE)
            }

def build_transform(phase='train'):
    """Create transforms pipeline with augmentation"""
    transforms = [ToTensor(), Normalize(mean=MEAN, std=STD)]
    
    if phase == 'train':
        transforms.insert(0, Resize(DEFAULT_IMAGE_SIZE))
        # Add actual augmentations: RandomFlip, ColorJitter, etc.
        
    return Compose(transforms)

def setup_logger(rank):
    """Configure distributed-aware logging"""
    logger = logging.getLogger(f'worker_{rank}')
    logger.setLevel(logging.DEBUG if rank == 0 else logging.WARNING)
    
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        f'[%(asctime)s] [Rank {rank}] [%(levelname)s] %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger

def init_distributed(rank, world_size):
    """Initialize distributed training backend"""
    os.environ['MASTER_ADDR'] = 'localhost' if world_size == 1 else '172.31.22.101'  # Replace with master IP
    os.environ['MASTER_PORT'] = '29500'
    os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'
    os.environ['NCCL_DEBUG'] = 'WARN'
    
    if torch.cuda.is_available():
        backend = 'nccl'
        torch.cuda.set_device(rank)
    else:
        backend = 'gloo'
        
    dist.init_process_group(
        backend=backend,
        init_method='env://',
        rank=rank,
        world_size=world_size
    )
    return dist.get_rank(), dist.get_world_size()

def create_model(num_classes=4):
    """Create Mask R-CNN model with custom head"""
    model = maskrcnn_resnet50_fpn(pretrained=True)
    
    # Replace heads for custom classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )
    
    return model

def train_epoch(model, dataloader, optimizer, scheduler, epoch, logger, device):
    """Single training epoch with distributed sync"""
    model.train()
    running_loss = 0.0
    sampler = dataloader.sampler
    if isinstance(sampler, DistributedSampler):
        sampler.set_epoch(epoch)
    
    start_time = time.time()
    
    for batch_idx, (images, targets) in enumerate(dataloader):
        # Move data to device
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        
        # Gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        
        optimizer.step()
        
        # Sync losses across devices
        reduced_loss = losses.clone().detach()
        dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
        reduced_loss = reduced_loss / dist.get_world_size()
        running_loss += reduced_loss.item()
        
        # Log every 10 batches
        if batch_idx % 10 == 0 and logger:
            logger.info(
                f"Epoch {epoch} Batch {batch_idx}/{len(dataloader)} "
                f"Loss: {losses.item():.4f} "
                f"LR: {optimizer.param_groups['lr']:.6f}"
            )
            
            if dist.get_rank() == 0:
                wandb.log({
                    "batch_loss": losses.item(),
                    "lr": optimizer.param_groups['lr']
                })
    
    # Sync and log epoch metrics
    epoch_loss = running_loss / len(dataloader)
    dist.all_reduce(torch.tensor(epoch_loss), op=dist.ReduceOp.SUM)
    epoch_loss = epoch_loss / dist.get_world_size()
    
    if scheduler:
        scheduler.step(epoch_loss)
    
    epoch_time = time.time() - start_time
    if logger:
        logger.info(
            f"Epoch {epoch} completed in {epoch_time:.2f}s | "
            f"Avg Loss: {epoch_loss:.4f} | "
            f"LR: {optimizer.param_groups['lr']:.6f}"
        )
        
    return epoch_loss

def validate(model, dataloader, logger, device):
    """Distributed validation routine"""
    model.eval()
    val_loss = 0.0
    iou_scores = []
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # Calculate IoU (simplified)
            # Actual implementation would use mask IoU
            iou = torch.rand(1).item()  
            
            # Sync metrics
            reduced_loss = losses.clone().detach()
            dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
            reduced_loss = reduced_loss / dist.get_world_size()
            val_loss += reduced_loss.item()
            
            dist.all_reduce(torch.tensor(iou), op=dist.ReduceOp.SUM)
            iou = iou / dist.get_world_size()
            iou_scores.append(iou)
    
    avg_iou = np.mean(iou_scores)
    avg_loss = val_loss / len(dataloader)
    
    if logger:
        logger.info(f"Validation Loss: {avg_loss:.4f} | mIoU: {avg_iou:.4f}")
    
    return avg_loss, avg_iou

def save_checkpoint(model, optimizer, epoch, metrics, file_path, logger):
    """Save checkpoint only from master process"""
    if dist.get_rank() == 0:
        try:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics
            }
            torch.save(checkpoint, file_path)
            if logger:
                logger.info(f"Saved checkpoint to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {str(e)}")

def main(rank, world_size, args):
    """Main training function for distributed execution"""
    
    # Initialize distributed system
    global_rank, world_size = init_distributed(rank, world_size)
    logger = setup_logger(global_rank)
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    
    # Initialize WandB on master
    if global_rank == 0 and args.wandb_key:
        wandb.login(key=args.wandb_key)
        wandb.init(project="ppe-detection", config=vars(args))
    
    logger.info(f"Distributed training initialized (rank {global_rank}/{world_size})")
    
    try:
        # Create model and wrap with DDP
        model = create_model(num_classes=args.num_classes).to(device)
        model = DDP(model, device_ids=[rank] if torch.cuda.is_available() else None)
        
        # Optimizer and scheduler
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=global_rank == 0
        )
        
        # Load datasets (mock data for example)
        train_transform = build_transform(phase='train')
        val_transform = build_transform(phase='val')
        
        train_dataset = PPEDataset(
            image_paths=['']*1000,  # Replace with actual paths
            annotation_paths=['']*1000,
            transform=train_transform
        )
        
        val_dataset = PPEDataset(
            image_paths=['']*200,  # Replace with actual paths
            annotation_paths=['']*200,
            transform=val_transform
        )
        
        # Distributed samplers
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=global_rank, shuffle=True
        )
        val_sampler = DistributedSampler(
            val_dataset, num_replicas=world_size, rank=global_rank, shuffle=False
        )
        
        # Data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.workers,
            pin_memory=True,
            collate_fn=lambda x: tuple(zip(*x))
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            sampler=val_sampler,
            num_workers=args.workers,
            pin_memory=True,
            collate_fn=lambda x: tuple(zip(*x))
        )
        
        # Training loop
        best_iou = 0.0
        for epoch in range(args.epochs):
            logger.info(f"Starting epoch {epoch+1}/{args.epochs}")
            
            # Train epoch
            train_loss = train_epoch(
                model, train_loader, optimizer, scheduler, epoch, logger, device
            )
            
            # Validate every 3 epochs
            if epoch % 3 == 0:
                val_loss, val_iou = validate(model, val_loader, logger, device)
                
                # Save best model
                if val_iou > best_iou and global_rank == 0:
                    best_iou = val_iou
                    save_checkpoint(
                        model, optimizer, epoch, 
                        {'val_iou': val_iou, 'val_loss': val_loss},
                        f"best_model_rank{global_rank}.pt",
                        logger
                    )
                
                # Log validation metrics
                if global_rank == 0 and args.wandb_key:
                    wandb.log({
                        "val_loss": val_loss,
                        "val_iou": val_iou,
                        "epoch": epoch
                    })
            
            # Save checkpoint periodically
            if epoch % 5 == 0:
                save_checkpoint(
                    model, optimizer, epoch, 
                    {'train_loss': train_loss},
                    f"checkpoint_epoch{epoch}_rank{global_rank}.pt",
                    logger
                )
                
    except Exception as e:
        logger.exception(f"Training failed: {str(e)}")
    finally:
        dist.destroy_process_group()
        if global_rank == 0 and args.wandb_key:
            wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Distributed Mask R-CNN Training')
    parser.add_argument('--world_size', type=int, default=torch.cuda.device_count(),
                        help='Number of distributed processes')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--wandb_key', type=str, default='', 
                        help='Weights & Biases API key')
    args = parser.parse_args()

    # Launch distributed training
    mp.spawn(
        main,
        args=(args.world_size, args),
        nprocs=args.world_size,
        join=True
    )
