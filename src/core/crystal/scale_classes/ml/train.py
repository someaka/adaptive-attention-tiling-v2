import torch
from .models import HolographicNet
from .trainer import HolographicTrainer

def main():
    # Initialize model and trainer
    model = HolographicNet(
        dim=4,
        hidden_dim=32,
        n_layers=4,
        dtype=torch.complex64
    )
    trainer = HolographicTrainer(model)
    
    # Train model
    metrics = trainer.train(
        n_epochs=1000,
        batch_size=128,
        val_split=0.2,
        lr=1e-3,
        patience=50,
        min_delta=1e-4
    )
    
    return metrics

if __name__ == "__main__":
    main() 