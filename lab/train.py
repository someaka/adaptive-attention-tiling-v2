import torch
import torch.nn as nn
import json
import time
from pathlib import Path
import argparse
from typing import Dict, Any, Optional
from model import SimpleTransformer, load_tokenized_samples

class TrainingConfig:
    """Configuration for model training."""
    def __init__(
        self,
        train_time: int = 60,
        batch_size: int = 32,
        learning_rate: float = 3e-4,
        vocab_size: int = 10000,
        model_dim: int = 64,
        num_heads: int = 2,
        dropout: float = 0.1,
        print_every: float = 0.5,
        save_every: int = 100,
        gradient_clip: float = 1.0,
        checkpoint_path: str = "lab/model_weights.pt",
        samples_path: str = "lab/tokenized_samples.json"
    ):
        self.train_time = train_time
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.vocab_size = vocab_size
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.print_every = print_every
        self.save_every = save_every
        self.gradient_clip = gradient_clip
        self.checkpoint_path = Path(checkpoint_path)
        self.samples_path = Path(samples_path)

class MetricsTracker:
    """Track and log training metrics."""
    def __init__(self):
        self.flow_metrics: list[float] = []
        self.quantum_norms: list[float] = []
        self.pattern_metrics: list[Dict[str, Any]] = []
        self.losses: list[float] = []
        self.best_loss = float('inf')
        self.start_time = time.time()
        self.last_print = self.start_time
        self.start_iteration = 0
        self.iteration = 0

    def update(self, loss: float, attention_metrics: Dict[str, Any]) -> None:
        """Update metrics with new values."""
        self.losses.append(loss)
        self.flow_metrics.append(attention_metrics['geometric_flow'])
        self.quantum_norms.append(attention_metrics['quantum_norm'].item())
        
        if 'arithmetic_metrics' in attention_metrics:
            self.pattern_metrics.append({
                'layer_norms': [m['layer_norm'] for m in attention_metrics['arithmetic_metrics']],
                'heights': [m['height'] for m in attention_metrics['arithmetic_metrics']],
                'l_values': [m['l_value'] for m in attention_metrics['arithmetic_metrics']],
                'flow_magnitudes': [m['flow_magnitude'] for m in attention_metrics['arithmetic_metrics']]
            })

    def should_print(self, print_every: float) -> bool:
        """Check if it's time to print metrics."""
        return time.time() - self.last_print >= print_every

    def print_metrics(self, attention_metrics: Dict[str, Any]) -> None:
        """Print current training metrics."""
        current_loss = self.losses[-1]
        avg_loss = sum(self.losses) / len(self.losses)
        elapsed = time.time() - self.start_time
        iterations_per_sec = (self.iteration - self.start_iteration) / elapsed
        
        print(f"\nIteration {self.iteration} (Total: {self.iteration - self.start_iteration}):")
        print(f"Time: {elapsed:.1f}s ({iterations_per_sec:.2f} it/s)")
        print(f"Loss: {current_loss:.4f} (Avg: {avg_loss:.4f}, Best: {self.best_loss:.4f})")
        print(f"Geometric Flow Stability: {attention_metrics['geometric_flow']:.4f}")
        print(f"Quantum State Norm: {attention_metrics['quantum_norm']:.4f}")
        
        # Print pattern metrics if available
        if self.pattern_metrics:
            latest = self.pattern_metrics[-1]
            print("\nPattern Evolution:")
            print(f"Layer Norms: {[f'{x:.3f}' for x in latest['layer_norms']]}")
            print(f"Heights: {[f'{x:.3f}' for x in latest['heights']]}")
            print(f"L-values: {[f'{x:.3f}' for x in latest['l_values']]}")
            print(f"Flow Magnitudes: {[f'{x:.3f}' for x in latest['flow_magnitudes']]}")
        
        # Print attention statistics
        attn = attention_metrics['attention']
        print(f"\nAttention Stats:")
        print(f"Mean: {attn.mean():.4f}, Std: {attn.std():.4f}")
        print(f"Range: [{attn.min():.4f}, {attn.max():.4f}]")
        
        # Print loss improvement rate
        if len(self.losses) > 1:
            loss_change = self.losses[-1] - self.losses[-2]
            print(f"\nLoss Change: {loss_change:.4f}")
            if len(self.losses) > 10:
                avg_loss_change = (self.losses[-1] - self.losses[-10]) / 10
                print(f"Avg Loss Change (10 steps): {avg_loss_change:.4f}")
        
        # Print quantum metrics
        if 'quantum_metrics' in attention_metrics:
            qm = attention_metrics['quantum_metrics']
            print("\nQuantum Metrics:")
            print(f"State Purity: {qm.get('purity', 0.0):.4f}")
            print(f"Entanglement: {qm.get('entanglement', 0.0):.4f}")
            print(f"Coherence: {qm.get('coherence', 0.0):.4f}")
        
        self.last_print = time.time()

    def print_summary(self) -> None:
        """Print training summary."""
        avg_loss = sum(self.losses) / len(self.losses)
        total_time = time.time() - self.start_time
        iterations = self.iteration - self.start_iteration
        
        print(f"\nTraining Summary:")
        print(f"Total Time: {total_time:.1f}s")
        print(f"Total Iterations: {iterations}")
        print(f"Iterations/second: {iterations / total_time:.1f}")
        print(f"Best loss: {self.best_loss:.4f}")
        print(f"Final loss: {self.losses[-1]:.4f}")
        print(f"Average loss: {avg_loss:.4f}")
        
        if len(self.losses) > 1:
            total_improvement = self.losses[0] - self.losses[-1]
            print(f"\nTotal Loss Improvement: {total_improvement:.4f}")
            print(f"Average Improvement/iteration: {total_improvement / iterations:.4f}")
        
        print("\nMetrics Evolution:")
        print(f"Geometric Flow Stability Range: [{min(self.flow_metrics):.4f}, {max(self.flow_metrics):.4f}]")
        print(f"Quantum Norm Range: [{min(self.quantum_norms):.4f}, {max(self.quantum_norms):.4f}]")
        
        if self.pattern_metrics:
            print("\nPattern Evolution:")
            first = self.pattern_metrics[0]
            last = self.pattern_metrics[-1]
            print("Layer Norms (Start -> End):")
            for i, (start, end) in enumerate(zip(first['layer_norms'], last['layer_norms'])):
                change = end - start
                print(f"  Layer {i}: {start:.3f} -> {end:.3f} (Δ: {change:+.3f})")
            print("Heights (Start -> End):")
            for i, (start, end) in enumerate(zip(first['heights'], last['heights'])):
                change = end - start
                print(f"  Layer {i}: {start:.3f} -> {end:.3f} (Δ: {change:+.3f})")
            
            # Calculate and print rate of change for key metrics
            if len(self.pattern_metrics) > 1:
                print("\nMetric Change Rates (per iteration):")
                total_iters = len(self.pattern_metrics) - 1
                for metric in ['layer_norms', 'heights', 'l_values', 'flow_magnitudes']:
                    changes = []
                    for i in range(len(first[metric])):
                        start = first[metric][i]
                        end = last[metric][i]
                        rate = (end - start) / total_iters
                        changes.append(rate)
                    print(f"{metric}: {[f'{x:.6f}' for x in changes]}")

def load_or_create_model(config: TrainingConfig) -> tuple[SimpleTransformer, int, float]:
    """Load existing model or create a new one."""
    model = SimpleTransformer(
        vocab_size=config.vocab_size,
        d_model=config.model_dim,
        n_heads=config.num_heads,
        dropout=config.dropout
    )
    
    start_iteration = 0
    best_loss = float('inf')
    
    if config.checkpoint_path.exists():
        print("Loading previous weights...")
        checkpoint = torch.load(config.checkpoint_path)
        if not hasattr(model.quantum_attention, '_return_metrics_buffer'):
            model.quantum_attention.register_buffer('_return_metrics_buffer', torch.tensor(True))
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        start_iteration = checkpoint.get('iteration', 0)
        best_loss = checkpoint.get('loss', float('inf'))
        print(f"Resuming from iteration {start_iteration} with loss {best_loss:.4f}")
    else:
        print("Starting fresh model training")
    
    return model, start_iteration, best_loss

def prepare_data(samples: list[Dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare training data from samples."""
    print("Pre-processing samples...")
    all_input_ids = []
    all_target_ids = []
    
    # Find max length first
    max_len = max(len(sample['input_ids']) for sample in samples if len(sample['input_ids']) > 2)
    print(f"Max sequence length: {max_len}")
    
    # Pre-pad all sequences and remap token IDs
    for sample in samples:
        ids = sample['input_ids']
        if len(ids) > 2:
            input_ids = [id % 10000 for id in ids[:-1]]
            target_ids = [id % 10000 for id in ids[1:]]
            
            input_padded = torch.zeros(max_len - 1, dtype=torch.long)
            target_padded = torch.zeros(max_len - 1, dtype=torch.long)
            
            input_padded[:len(input_ids)] = torch.tensor(input_ids)
            target_padded[:len(target_ids)] = torch.tensor(target_ids)
            
            all_input_ids.append(input_padded)
            all_target_ids.append(target_padded)
    
    # Stack all tensors
    all_input_ids = torch.stack(all_input_ids)
    all_target_ids = torch.stack(all_target_ids)
    
    print(f"Input shape: {all_input_ids.shape}")
    print(f"Target shape: {all_target_ids.shape}")
    
    return all_input_ids, all_target_ids

def save_checkpoint(
    model: SimpleTransformer,
    metrics: MetricsTracker,
    path: Path,
    config: TrainingConfig
) -> None:
    """Save model checkpoint with metrics."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': config.vocab_size,
        'iteration': metrics.iteration,
        'loss': metrics.losses[-1] if metrics.losses else float('inf'),
        'avg_loss': sum(metrics.losses) / len(metrics.losses) if metrics.losses else float('inf'),
        'flow_metrics': metrics.flow_metrics,
        'quantum_norms': metrics.quantum_norms,
        'pattern_metrics': metrics.pattern_metrics
    }, path)
    print(f"Model saved to {path}")

def train_model(config: TrainingConfig) -> SimpleTransformer:
    """Train the model with the given configuration."""
    # Setup model and optimizer
    model, start_iteration, best_loss = load_or_create_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Load and prepare data
    print("\nLoading samples...")
    samples = load_tokenized_samples(str(config.samples_path))
    all_input_ids, all_target_ids = prepare_data(samples)
    
    # Setup metrics tracking
    metrics = MetricsTracker()
    metrics.start_iteration = start_iteration
    metrics.iteration = start_iteration
    metrics.best_loss = best_loss
    
    print(f"\nStarting training with {len(all_input_ids)} samples...")
    print(f"Training time: {config.train_time} seconds")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    
    while (time.time() - metrics.start_time) < config.train_time:
        # Get random batch
        indices = torch.randperm(len(all_input_ids))[:config.batch_size]
        input_batch = all_input_ids[indices]
        target_batch = all_target_ids[indices]
        
        # Forward pass with metrics
        outputs, attention_metrics = model(input_batch, return_metrics=True)
        loss = criterion(outputs.view(-1, outputs.size(-1)), target_batch.view(-1))
        
        # Update metrics
        metrics.update(loss.item(), attention_metrics)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step()
        
        # Print progress
        if metrics.should_print(config.print_every):
            metrics.print_metrics(attention_metrics)
        
        # Save if best loss
        if metrics.losses[-1] < metrics.best_loss:
            metrics.best_loss = metrics.losses[-1]
            save_checkpoint(model, metrics, config.checkpoint_path.with_name("model_weights_best.pt"), config)
        
        # Regular checkpoint save
        if metrics.iteration % config.save_every == 0:
            save_checkpoint(model, metrics, config.checkpoint_path, config)
        
        metrics.iteration += 1
    
    # Print final summary
    metrics.print_summary()
    
    # Save final model
    save_checkpoint(model, metrics, config.checkpoint_path, config)
    
    return model

def main():
    parser = argparse.ArgumentParser(description='Train the SimpleTransformer model')
    parser.add_argument('--train-time', type=int, default=60,
                      help='Training time in seconds')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                      help='Learning rate')
    parser.add_argument('--print-every', type=float, default=0.5,
                      help='Print metrics every N seconds')
    parser.add_argument('--save-every', type=int, default=100,
                      help='Save checkpoint every N iterations')
    parser.add_argument('--checkpoint', type=str, default='lab/model_weights.pt',
                      help='Path to save/load model checkpoint')
    args = parser.parse_args()
    
    config = TrainingConfig(
        train_time=args.train_time,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        print_every=args.print_every,
        save_every=args.save_every,
        checkpoint_path=args.checkpoint
    )
    
    train_model(config)

if __name__ == "__main__":
    main() 