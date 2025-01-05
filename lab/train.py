import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import time
from pathlib import Path
import argparse
from typing import Dict, Any, Optional

from src.core.initialization import InitializationConfig, InitializationSystem
from src.core.tiling.state_manager import StateType
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
        samples_path: str = "lab/tokenized_samples.json",
        # Initialization system parameters
        state_type: StateType = StateType.PURE,
        max_entanglement: float = 1.0,
        epsilon: float = 1e-6,
        min_scale: float = 0.25,
        max_scale: float = 4.0,
        num_scales: int = 4,
        motive_rank: int = 4,
        num_primes: int = 8
    ):
        # Training parameters
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
        
        # Initialization system parameters
        self.state_type = state_type
        self.max_entanglement = max_entanglement
        self.epsilon = epsilon
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.num_scales = num_scales
        self.motive_rank = motive_rank
        self.num_primes = num_primes
    
    def get_init_config(self) -> InitializationConfig:
        """Get initialization system configuration."""
        return InitializationConfig(
            hidden_dim=self.model_dim,
            num_heads=self.num_heads,
            state_type=self.state_type,
            max_entanglement=self.max_entanglement,
            epsilon=self.epsilon,
            min_scale=self.min_scale,
            max_scale=self.max_scale,
            num_scales=self.num_scales,
            motive_rank=self.motive_rank,
            num_primes=self.num_primes
        )

class MetricsTracker:
    """Track and log training metrics."""
    def __init__(self):
        self.component_states: list[Dict[str, Any]] = []
        self.initialization_validations: list[Dict[str, bool]] = []
        self.losses: list[float] = []
        self.best_loss = float('inf')
        self.start_time = time.time()
        self.last_print = self.start_time
        self.start_iteration = 0
        self.iteration = 0

    def update(self, loss: float, metrics: Dict[str, Any]) -> None:
        """Update metrics with new values."""
        self.losses.append(loss)
        self.component_states.append(metrics['component_states'])
        self.initialization_validations.append(metrics['initialization_validation'])

    def should_print(self, print_every: float) -> bool:
        """Check if it's time to print metrics."""
        return time.time() - self.last_print >= print_every

    def print_metrics(self, metrics: Dict[str, Any]) -> None:
        """Print current training metrics."""
        current_loss = self.losses[-1]
        avg_loss = sum(self.losses) / len(self.losses)
        elapsed = time.time() - self.start_time
        iterations_per_sec = (self.iteration - self.start_iteration) / elapsed
        
        print(f"\nIteration {self.iteration} (Total: {self.iteration - self.start_iteration}):")
        print(f"Time: {elapsed:.1f}s ({iterations_per_sec:.2f} it/s)")
        print(f"Loss: {current_loss:.4f} (Avg: {avg_loss:.4f}, Best: {self.best_loss:.4f})")
        
        # Print component states
        states = metrics['component_states']
        print("\nComponent States:")
        
        # State Manager
        print("State Manager:")
        for state_name, state in states['state_manager'].items():
            print(f"  {state_name}: norm={torch.norm(state).item():.4f}")
        
        # Pattern Processor
        print("\nPattern Processor:")
        print(f"  Quantum State Norm: {torch.norm(states['pattern_processor']['quantum_state']).item():.4f}")
        print(f"  Geometric State Norm: {torch.norm(states['pattern_processor']['geometric_state']).item():.4f}")
        
        # Quantum Tile
        print("\nQuantum Tile:")
        print(f"  Resolution: {states['quantum_tile']['resolution']:.4f}")
        print(f"  Load Profile: {states['quantum_tile']['load_profile']}")
        
        # Scale Transition
        print("\nScale Transition:")
        for scale, entanglement in states['scale_transition'].items():
            print(f"  Scale {scale}: entanglement={entanglement:.4f}")
        
        # Print initialization validation
        validation = metrics['initialization_validation']
        print("\nInitialization Validation:")
        for component, is_valid in validation.items():
            print(f"  {component}: {'✓' if is_valid else '✗'}")
        
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
        
        # Print component state evolution
        print("\nComponent State Evolution:")
        first_states = self.component_states[0]
        last_states = self.component_states[-1]
        
        for component in ['state_manager', 'pattern_processor', 'quantum_tile', 'scale_transition']:
            print(f"\n{component} Evolution:")
            if isinstance(first_states[component], dict):
                for key in first_states[component].keys():
                    if torch.is_tensor(first_states[component][key]):
                        start_norm = torch.norm(first_states[component][key]).item()
                        end_norm = torch.norm(last_states[component][key]).item()
                        print(f"  {key}: {start_norm:.4f} -> {end_norm:.4f}")
                    else:
                        print(f"  {key}: {first_states[component][key]} -> {last_states[component][key]}")
        
        # Print initialization validation evolution
        print("\nInitialization Validation Evolution:")
        first_validation = self.initialization_validations[0]
        last_validation = self.initialization_validations[-1]
        
        for component in first_validation.keys():
            start_valid = first_validation[component]
            end_valid = last_validation[component]
            print(f"  {component}: {start_valid} -> {end_valid}")

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
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        start_iteration = checkpoint.get('iteration', 0)
        best_loss = checkpoint.get('loss', float('inf'))
        
        # Validate initialization system after loading
        init_validation = model.init_system.validate_initialization()
        all_valid = all(init_validation.values())
        print(f"Initialization validation after loading: {'✓' if all_valid else '✗'}")
        if not all_valid:
            print("Warning: Some components failed validation:")
            for component, is_valid in init_validation.items():
                if not is_valid:
                    print(f"  {component}: ✗")
        
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
    # Get final component states and validation
    final_states = model.init_system.get_component_states()
    final_validation = model.init_system.validate_initialization()
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': config.vocab_size,
        'iteration': metrics.iteration,
        'loss': metrics.losses[-1] if metrics.losses else float('inf'),
        'avg_loss': sum(metrics.losses) / len(metrics.losses) if metrics.losses else float('inf'),
        'final_component_states': final_states,
        'final_validation': final_validation,
        'component_states_history': metrics.component_states[-10:],  # Save last 10 states
        'validation_history': metrics.initialization_validations[-10:]  # Save last 10 validations
    }, path)
    print(f"Model saved to {path}")
    
    # Print validation status
    print("\nFinal Component Validation:")
    for component, is_valid in final_validation.items():
        print(f"  {component}: {'✓' if is_valid else '✗'}")

def train_model(config: TrainingConfig) -> SimpleTransformer:
    """Train the model with the given configuration."""
    # Setup model and optimizer
    model, start_iteration, best_loss = load_or_create_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # Load training data
    samples = load_tokenized_samples(str(config.samples_path))
    input_ids, target_ids = prepare_data(samples)
    
    # Initialize metrics tracker
    metrics = MetricsTracker()
    metrics.start_iteration = start_iteration
    metrics.iteration = start_iteration
    metrics.best_loss = best_loss
    
    # Training loop
    start_time = time.time()
    print("\nStarting training loop...")
    
    try:
        while time.time() - start_time < config.train_time:
            # Get random batch
            batch_indices = torch.randint(0, len(input_ids), (config.batch_size,))
            batch_input = input_ids[batch_indices]
            batch_target = target_ids[batch_indices]
            
            # Forward pass with metrics
            output, attention_metrics = model(batch_input, return_metrics=True)
            
            # Calculate loss
            loss = torch.nn.functional.cross_entropy(
                output.view(-1, config.vocab_size),
                batch_target.view(-1)
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            
            # Update metrics
            metrics.update(loss.item(), attention_metrics)
            metrics.iteration += 1
            
            # Update best loss
            if loss.item() < metrics.best_loss:
                metrics.best_loss = loss.item()
                save_checkpoint(model, metrics, config.checkpoint_path.with_suffix('.best.pt'), config)
            
            # Print progress
            if metrics.should_print(config.print_every):
                metrics.print_metrics(attention_metrics)
            
            # Save checkpoint
            if metrics.iteration % config.save_every == 0:
                save_checkpoint(model, metrics, config.checkpoint_path, config)
                
            # Validate initialization system periodically
            if metrics.iteration % 100 == 0:
                init_validation = model.init_system.validate_initialization()
                all_valid = all(init_validation.values())
                if not all_valid:
                    print("\nWarning: Some components failed validation:")
                    for component, is_valid in init_validation.items():
                        if not is_valid:
                            print(f"  {component}: ✗")
                    
                    # Try to reinitialize invalid components
                    print("Attempting to reinitialize invalid components...")
                    model.init_system = InitializationSystem(config.get_init_config())
                    
                    # Validate again
                    init_validation = model.init_system.validate_initialization()
                    all_valid = all(init_validation.values())
                    print(f"Reinitialization {'successful' if all_valid else 'failed'}")
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    # Final save
    save_checkpoint(model, metrics, config.checkpoint_path, config)
    metrics.print_summary()
    
    return model

def main():
    parser = argparse.ArgumentParser(description="Train the model")
    
    # Training parameters
    parser.add_argument("--train-time", type=int, default=60,
                      help="Training time in seconds")
    parser.add_argument("--batch-size", type=int, default=32,
                      help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                      help="Learning rate")
    parser.add_argument("--vocab-size", type=int, default=10000,
                      help="Vocabulary size")
    parser.add_argument("--model-dim", type=int, default=64,
                      help="Model dimension")
    parser.add_argument("--num-heads", type=int, default=2,
                      help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.1,
                      help="Dropout rate")
    parser.add_argument("--gradient-clip", type=float, default=1.0,
                      help="Gradient clipping value")
    
    # Initialization system parameters
    parser.add_argument("--state-type", type=str, default="PURE",
                      choices=["PURE", "MIXED"],
                      help="Quantum state type")
    parser.add_argument("--max-entanglement", type=float, default=1.0,
                      help="Maximum entanglement value")
    parser.add_argument("--epsilon", type=float, default=1e-6,
                      help="Epsilon for numerical stability")
    parser.add_argument("--min-scale", type=float, default=0.25,
                      help="Minimum scale for transitions")
    parser.add_argument("--max-scale", type=float, default=4.0,
                      help="Maximum scale for transitions")
    parser.add_argument("--num-scales", type=int, default=4,
                      help="Number of scale levels")
    parser.add_argument("--motive-rank", type=int, default=4,
                      help="Rank of the motive")
    parser.add_argument("--num-primes", type=int, default=8,
                      help="Number of prime factors")
    
    # File paths
    parser.add_argument("--checkpoint-path", type=str,
                      default="lab/model_weights.pt",
                      help="Path to save model checkpoints")
    parser.add_argument("--samples-path", type=str,
                      default="lab/tokenized_samples.json",
                      help="Path to tokenized samples")
    
    args = parser.parse_args()
    
    # Create config with all parameters
    config = TrainingConfig(
        train_time=args.train_time,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        vocab_size=args.vocab_size,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        dropout=args.dropout,
        gradient_clip=args.gradient_clip,
        checkpoint_path=args.checkpoint_path,
        samples_path=args.samples_path,
        state_type=StateType[args.state_type],
        max_entanglement=args.max_entanglement,
        epsilon=args.epsilon,
        min_scale=args.min_scale,
        max_scale=args.max_scale,
        num_scales=args.num_scales,
        motive_rank=args.motive_rank,
        num_primes=args.num_primes
    )
    
    # Train model
    model = train_model(config)
    print("\nTraining completed successfully")

if __name__ == "__main__":
    main() 