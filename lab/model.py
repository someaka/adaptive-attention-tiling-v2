import torch
import torch.nn as nn
import json
import time
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2Config
from src.core.attention.compute import AttentionCompute
from src.core.attention.geometric import (
    HyperbolicExponential,
    HyperbolicLogarithm,
    EuclideanExponential,
    EuclideanLogarithm,
    ParallelTransport
)
from src.core.tiling.quantum_geometric_attention import QuantumGeometricAttention as CoreQuantumGeometricAttention
from src.core.tiling.geometric_flow import GeometricFlow
from src.core.patterns.arithmetic_dynamics import ArithmeticPattern
from src.core.tiling.quantum_attention_tile import QuantumMotivicTile
import tiktoken
import torch.nn.functional as F
from src.core.initialization import InitializationConfig, InitializationSystem
from src.core.tiling.state_manager import StateType

class AdaptiveAttention(nn.Module):
    def __init__(self, 
                 d_model=512,      # Model dimension
                 n_heads=8,        # Number of attention heads
                 dropout=0.1):     # Dropout rate
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # Project queries, keys, values
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_head))
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        
        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.out_proj(out)
        
        return out

class QuantumGeometricAttention(nn.Module):
    def __init__(self, dim, heads, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        
        # Quantum state preparation
        self.to_qkv = nn.Linear(dim, dim * 3)
        
        # Geometric flow components
        self.flow = GeometricFlow(
            hidden_dim=self.head_dim,  # Per head dimension
            manifold_dim=self.head_dim,
            motive_rank=4,
            num_charts=4,
            integration_steps=10,
            dt=0.1,
            stability_threshold=1e-6
        )
        
        # Phase encoding with stability
        self.phase_net = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim * 2),  # Per head dimension
            nn.Tanh(),
            nn.Linear(self.head_dim * 2, self.head_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def prepare_quantum_state(self, x):
        # x shape: [batch, heads, seq_len, head_dim]
        B, H, N, D = x.shape
        
        # Project to quantum state space (per head)
        x_flat = x.reshape(B * H * N, D)
        phase = self.phase_net(x_flat)
        phase = phase.view(B, H, N, D)
        
        amplitude = F.normalize(x, dim=-1)
        
        # Add stability through phase encoding
        phase = torch.tanh(phase) * torch.pi  # Bounded phase
        return amplitude * torch.exp(1j * phase)
        
    def geometric_flow(self, state):
        # state shape: [batch, heads, seq_len, head_dim]
        B, H, N, D = state.shape
        
        # Apply geometric flow with stability (per head)
        state_flat = state.reshape(B * H * N, D)
        state_evolved, metrics = self.flow(state_flat)
        state_evolved = state_evolved.view(B, H, N, D)
        
        # Monitor stability
        if 'stability' in metrics and metrics['stability'] < 0.5:
            # Fall back to identity if unstable
            return F.normalize(state, dim=-1)
            
        return F.normalize(state_evolved, dim=-1)
        
    def parallel_transport(self, state, key_state):
        # Compute geodesic distance
        dist = torch.sum((state - key_state) ** 2, dim=-1, keepdim=True).sqrt()
        
        # Adaptive transport based on distance
        transport_factor = torch.sigmoid(-dist * 10)  # Smooth cutoff
        transported = state * transport_factor + key_state * (1 - transport_factor)
        
        return F.normalize(transported, dim=-1)
    
    def forward(self, x, mask=None, return_metrics=False):
        B, N, C = x.shape
        
        # Project to QKV with stability
        qkv = self.to_qkv(x)
        qkv = F.normalize(qkv, dim=-1)  # Normalize for stability
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape to multi-head
        q = q.view(B, N, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.heads, self.head_dim).transpose(1, 2)
        
        # Prepare quantum states with phase stability
        q = self.prepare_quantum_state(q)
        k = self.prepare_quantum_state(k)
        v = self.prepare_quantum_state(v)
        
        # Apply geometric flow with stability monitoring
        q = self.geometric_flow(q)
        k = self.geometric_flow(k)
        
        # Parallel transport with adaptive factor
        k_transported = self.parallel_transport(k, q)
        
        # Compute attention with stability
        dots = torch.matmul(q, k_transported.transpose(-2, -1)) * self.scale
        dots = dots.real  # Use real part for stability
        
        if mask is not None:
            dots = dots.masked_fill(mask == 0, float('-inf'))
            
        attn = F.softmax(dots, dim=-1)
        attn = self.dropout(attn)
        
        # Convert v to real before attention
        v = v.abs()  # Take magnitude of complex values
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, N, C)
        
        if return_metrics:
            metrics = {
                'attention': attn.detach(),  # Detach for metrics
                'quantum_norm': torch.norm(q, dim=-1).mean().item(),
                'geometric_flow_stability': 1.0  # Default stability value
            }
            return out, metrics
        return out

class SimpleTransformer(nn.Module):
    def __init__(self,
                 vocab_size,       # Size of vocabulary
                 d_model=64,       # Tiny model dimension
                 n_heads=2,        # Minimal number of heads
                 dropout=0.1):     # Dropout rate
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 128, d_model))
        
        # Initialize the central initialization system
        init_config = InitializationConfig(
            hidden_dim=d_model,
            num_heads=n_heads,
            state_type=StateType.PURE,
            max_entanglement=1.0,
            epsilon=1e-6,
            min_scale=0.25,
            max_scale=4.0,
            num_scales=4,
            motive_rank=4,
            num_primes=8
        )
        self.init_system = InitializationSystem(init_config)
        
        # Add quantum geometric attention
        self.quantum_attention = QuantumGeometricAttention(
            dim=d_model,
            heads=n_heads,
            dropout=dropout
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Minimal feed-forward
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.final = nn.Linear(d_model, vocab_size)
        
    def forward(self, x, mask=None, return_metrics=False):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        
        batch_size, seq_len = x.shape
        
        # Embedding + positional encoding
        x = self.embedding(x) + self.pos_encoding[:, :seq_len, :]
        
        # Process through initialization system
        attention_out = self.init_system(x)
        
        # Post-attention processing
        x = self.norm1(x + attention_out)
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        
        # Final projection
        x = self.final(x)
        
        if return_metrics:
            metrics = {
                'component_states': self.init_system.get_component_states(),
                'initialization_validation': self.init_system.validate_initialization()
            }
            return x, metrics
        return x
    
    def predict_next(self, input_ids, top_k=5):
        """Predict the top_k most likely next tokens."""
        with torch.no_grad():
            # Get model output
            outputs = self(input_ids)
            
            # Get predictions for the last position
            next_token_logits = outputs[0, -1, :]
            
            # Get top k predictions
            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
            
            # Convert to probabilities
            top_k_probs = torch.softmax(top_k_logits, dim=-1)
            
            return list(zip(top_k_indices.tolist(), top_k_probs.tolist()))

class PreTrainedTransformer(nn.Module):
    def __init__(self, model_name='distilgpt2'):
        super().__init__()
        print(f"Loading {model_name}...")
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.config = self.model.config
        
    def forward(self, x, mask=None):
        outputs = self.model(x, attention_mask=mask)
        return outputs.logits
    
    def predict_next(self, input_ids, top_k=5):
        """Predict the top_k most likely next tokens."""
        with torch.no_grad():
            # Get model output
            outputs = self(input_ids.unsqueeze(0))
            
            # Get predictions for the last position
            next_token_logits = outputs[0, -1, :]
            
            # Get top k predictions
            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
            
            # Convert to probabilities
            top_k_probs = torch.softmax(top_k_logits, dim=-1)
            
            return list(zip(top_k_indices.tolist(), top_k_probs.tolist()))

def load_tokenized_samples(file_path="lab/tokenized_samples.json"):
    with open(file_path, 'r') as f:
        return json.load(f)

def train_simple_model(vocab_size, train_time=30):
    print("\nCreating simple model...")
    model = SimpleTransformer(vocab_size=10000)
    
    # Load previous weights
    model_path = Path("lab/model_weights_30s.pt")
    if model_path.exists():
        print("Loading previous weights...")
        checkpoint = torch.load(model_path)
        if not hasattr(model.quantum_attention, '_return_metrics_buffer'):
            model.quantum_attention.register_buffer('_return_metrics_buffer', torch.tensor(True))
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        start_iteration = checkpoint.get('iteration', 0)
        best_loss = checkpoint.get('loss', float('inf'))
        print(f"Resuming from iteration {start_iteration} with loss {best_loss:.4f}")
    else:
        start_iteration = 0
        best_loss = float('inf')
        print("No previous weights found, starting fresh")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()
    
    print("\nLoading samples...")
    samples = load_tokenized_samples()
    
    # Pre-process all samples to tensors
    print("Pre-processing samples...")
    all_input_ids = []
    all_target_ids = []
    
    # Find max length first
    max_len = 0
    for sample in samples:
        ids = sample['input_ids']
        if len(ids) > 2:
            max_len = max(max_len, len(ids))
    
    print(f"Max sequence length: {max_len}")
    
    # Pre-pad all sequences and remap token IDs
    for sample in samples:
        ids = sample['input_ids']
        if len(ids) > 2:
            # Remap token IDs to fit in reduced vocabulary
            input_ids = [id % 10000 for id in ids[:-1]]
            target_ids = [id % 10000 for id in ids[1:]]
            
            # Pad sequences
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
    
    # Training loop
    print(f"\nStarting training with {len(all_input_ids)} samples...")
    start_time = time.time()
    batch_size = 32
    iteration = start_iteration
    running_loss = 0.0
    last_print = start_time
    
    # Metrics tracking
    flow_metrics_history = []
    quantum_norms_history = []
    pattern_metrics_history = []
    
    while (time.time() - start_time) < train_time:
        # Get random batch indices
        indices = torch.randperm(len(all_input_ids))[:batch_size]
        input_batch = all_input_ids[indices]
        target_batch = all_target_ids[indices]
        
        # Forward pass with metrics
        outputs, attention_metrics = model(input_batch, return_metrics=True)
        loss = criterion(outputs.view(-1, outputs.size(-1)), target_batch.view(-1))
        
        # Track detailed metrics
        flow_metrics_history.append(attention_metrics['geometric_flow'])
        quantum_norms_history.append(attention_metrics['quantum_norm'].item())
        if 'arithmetic_metrics' in attention_metrics:
            pattern_metrics_history.append({
                'layer_norms': [m['layer_norm'] for m in attention_metrics['arithmetic_metrics']],
                'heights': [m['height'] for m in attention_metrics['arithmetic_metrics']],
                'l_values': [m['l_value'] for m in attention_metrics['arithmetic_metrics']],
                'flow_magnitudes': [m['flow_magnitude'] for m in attention_metrics['arithmetic_metrics']]
            })
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Update running statistics
        current_loss = loss.item()
        running_loss += current_loss
        elapsed = time.time() - start_time
        
        # Print progress every 0.5 seconds
        if time.time() - last_print >= 0.5:
            avg_loss = running_loss / (iteration - start_iteration + 1)
            print(f"\nIteration {iteration} (Total: {iteration - start_iteration}):")
            print(f"Loss: {current_loss:.4f}, Avg Loss: {avg_loss:.4f}, Time: {elapsed:.1f}s")
            print(f"Geometric Flow Stability: {attention_metrics['geometric_flow']:.4f}")
            print(f"Quantum State Norm: {attention_metrics['quantum_norm']:.4f}")
            
            # Print pattern metrics if available
            if pattern_metrics_history:
                latest = pattern_metrics_history[-1]
                print("\nPattern Metrics:")
                print(f"Layer Norms: {[f'{x:.3f}' for x in latest['layer_norms']]}")
                print(f"Heights: {[f'{x:.3f}' for x in latest['heights']]}")
                print(f"L-values: {[f'{x:.3f}' for x in latest['l_values']]}")
                print(f"Flow Magnitudes: {[f'{x:.3f}' for x in latest['flow_magnitudes']]}")
            
            # Print attention statistics
            attn = attention_metrics['attention']
            print(f"\nAttention Stats:")
            print(f"Mean: {attn.mean():.4f}, Std: {attn.std():.4f}")
            print(f"Range: [{attn.min():.4f}, {attn.max():.4f}]")
            last_print = time.time()
        
        # Save if best loss
        if current_loss < best_loss:
            best_loss = current_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'vocab_size': 10000,
                'iteration': iteration,
                'loss': current_loss,
                'avg_loss': running_loss / (iteration - start_iteration + 1),
                'flow_metrics': flow_metrics_history,
                'quantum_norms': quantum_norms_history,
                'pattern_metrics': pattern_metrics_history
            }, Path("lab/model_weights_best.pt"))
        
        iteration += 1
    
    avg_loss = running_loss / (iteration - start_iteration)
    print(f"\nTraining complete! Total iterations: {iteration - start_iteration}")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Average loss: {avg_loss:.4f}")
    print(f"Iterations per second: {(iteration - start_iteration) / train_time:.1f}")
    
    # Print final metrics summary
    print("\nMetrics Summary:")
    print(f"Geometric Flow Stability Range: [{min(flow_metrics_history):.4f}, {max(flow_metrics_history):.4f}]")
    print(f"Quantum Norm Range: [{min(quantum_norms_history):.4f}, {max(quantum_norms_history):.4f}]")
    
    if pattern_metrics_history:
        print("\nPattern Evolution:")
        first = pattern_metrics_history[0]
        last = pattern_metrics_history[-1]
        print("Layer Norms (Start -> End):")
        for i, (start, end) in enumerate(zip(first['layer_norms'], last['layer_norms'])):
            print(f"  Layer {i}: {start:.3f} -> {end:.3f}")
        print("Heights (Start -> End):")
        for i, (start, end) in enumerate(zip(first['heights'], last['heights'])):
            print(f"  Layer {i}: {start:.3f} -> {end:.3f}")
    
    # Save final model with metrics
    save_path = Path("lab/model_weights_30s.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': 10000,
        'iteration': iteration,
        'loss': current_loss,
        'avg_loss': avg_loss,
        'flow_metrics': flow_metrics_history,
        'quantum_norms': quantum_norms_history,
        'pattern_metrics': pattern_metrics_history
    }, save_path)
    print(f"Model saved to {save_path}")
    
    return model

class PatternFlow(nn.Module):
    def __init__(self, input_dim, hidden_dim, manifold_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.manifold_dim = manifold_dim
        self._metrics = {'stability': 1.0}  # Default stability
        
        # Chart embedding
        self.chart_embedding = nn.Parameter(torch.randn(1, manifold_dim))
        
        # Metric components
        self.metric_factors = nn.Parameter(torch.ones(manifold_dim))
        self.transitions = nn.Parameter(torch.eye(manifold_dim))
        
        # Arithmetic components
        self.coupling = nn.Parameter(torch.randn(hidden_dim))
        self.prime_bases = nn.Parameter(torch.arange(2, 2 + hidden_dim, dtype=torch.float))
        self.height_map = nn.Linear(input_dim, hidden_dim)
        self.flow_map = nn.Linear(hidden_dim, hidden_dim)
        self.l_function = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.adelic_proj = nn.Linear(hidden_dim, manifold_dim)
        self.output_proj = nn.Linear(manifold_dim, input_dim)
        
        # Flow components
        self.flow_field = nn.Sequential(
            nn.Linear(manifold_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, manifold_dim)
        )
        
        self.hamiltonian = nn.Sequential(
            nn.Linear(manifold_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, manifold_dim)
        )
        
        self.manifold_proj = nn.Sequential(
            nn.Linear(manifold_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, manifold_dim)
        )
        
        # Flow networks
        self.flow_net = nn.Sequential(
            nn.Linear(manifold_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, manifold_dim)
        )
        
        self.energy_net = nn.Sequential(
            nn.Linear(manifold_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, manifold_dim)
        )
        
        self.output_proj = nn.Sequential(
            nn.Linear(manifold_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Metric networks
        self.metric_net = nn.Sequential(
            nn.Linear(manifold_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, manifold_dim)
        )
        
        self.ricci_net = nn.Sequential(
            nn.Linear(manifold_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, manifold_dim)
        )
        
    def forward(self, x):
        # x shape: [batch * heads * seq_len, head_dim]
        batch_size = x.size(0)
        
        # Convert complex input to real by taking magnitude
        if x.is_complex():
            x = x.abs()
        
        # Embed in chart
        chart = self.chart_embedding.expand(batch_size, -1)
        
        # Apply metric
        metric = self.transitions @ torch.diag(self.metric_factors)
        
        # Arithmetic flow
        height = self.height_map(x)
        flow = self.flow_map(height)
        l_values = self.l_function(flow)
        adelic = self.adelic_proj(l_values)
        
        # Flow evolution
        flow_field = self.flow_field(chart)
        hamiltonian = self.hamiltonian(flow_field)
        manifold = self.manifold_proj(hamiltonian)
        
        # Energy conservation
        energy = self.energy_net(manifold)
        flow = self.flow_net(energy)
        
        # Metric evolution
        metric_evolved = self.metric_net(flow)
        ricci = self.ricci_net(metric_evolved)
        
        # Monitor stability
        stability = torch.mean(torch.abs(metric_evolved - ricci))
        self._metrics['stability'] = stability.item()
        
        # Project back
        out = self.output_proj(flow)
        
        # Convert back to complex if input was complex
        if x.is_complex():
            out = out * (1 + 0j)
        
        return out, self._metrics

if __name__ == "__main__":
    print("Loading vocabulary info...")
    with open('lab/vocab_info.json', 'r') as f:
        vocab_info = json.load(f)
    print(f"Vocabulary size: {vocab_info['vocab_size']}")
    
    # Train our simple model
    simple_model = train_simple_model(vocab_info['vocab_size'], train_time=30)
    
    # Load pre-trained model
    pretrained_model = PreTrainedTransformer()
    
    # Test sentences to complete
    test_sentences = [
        "The quick brown fox",
        "Once upon a time there was",
        "The meaning of life is",
        "In the beginning of the",
        "She looked at the sky and saw"
    ]
    
    print("\nGenerating completions...")
    print("-" * 50)
    
    for sentence in test_sentences:
        print(f"\nPrompt: {sentence}")
        
        # Get tokens
        with torch.no_grad():
            tokens = torch.tensor([t % 10000 for t in tiktoken.get_encoding("cl100k_base").encode(sentence)], dtype=torch.long)
            
            # Generate 10 more tokens
            for _ in range(10):
                # Get predictions
                predictions = simple_model.predict_next(tokens)
                
                # Sample from top 5 with temperature
                probs = torch.tensor([p for _, p in predictions])
                probs = torch.softmax(probs / 0.7, dim=0)  # temperature = 0.7
                next_token_idx = torch.multinomial(probs, 1)[0]
                next_token = predictions[next_token_idx][0]
                
                # Append to sequence
                tokens = torch.cat([tokens, torch.tensor([next_token], dtype=torch.long)])
            
            # Decode
            completed = sentence + " " + tiktoken.get_encoding("cl100k_base").decode([t % 10000 for t in tokens[len(tokens)-10:].tolist()])
            print(f"Completed: {completed}")
            
            # Show token probabilities for last prediction
            print("\nFinal token probabilities:")
            for token_id, prob in predictions:
                print(f"Token {token_id}: {prob:.4f}")
        
        print("-" * 50)
    
    print("\n=== Generation Complete ===")