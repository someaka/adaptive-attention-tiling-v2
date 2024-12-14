import torch
from torch import nn

from src.neural.attention.pattern import PatternDynamics


class InformationRouter(nn.Module):
    """Routes information based on pattern dynamics and information geometry."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_patterns: int = 64,
        bottleneck_threshold: float = 0.7,
        exploration_rate: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.bottleneck_threshold = bottleneck_threshold
        self.exploration_rate = exploration_rate

        # Pattern dynamics system
        self.pattern_dynamics = PatternDynamics(
            grid_size=dim,
            space_dim=num_heads,
            hidden_dim=num_patterns,
            boundary='periodic',
            dt=0.01,
            num_modes=8
        )

        # Routing networks
        self.route_scorer = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.ReLU(), nn.Linear(dim * 2, 1)
        )

        self.path_predictor = nn.GRU(
            input_size=dim, hidden_size=dim, num_layers=2, batch_first=True
        )

        # Bottleneck detection
        self.bottleneck_detector = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.ReLU(), nn.Linear(dim, 1), nn.Sigmoid()
        )

    def detect_bottlenecks(
        self, states: torch.Tensor, routing_scores: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Detect information bottlenecks in the routing graph."""
        # states: [batch, heads, seq_len, dim]
        # routing_scores: [batch, heads, seq_len]

        # Compute flow divergence
        flow = torch.gradient(routing_scores, dim=-1)[0]

        # Detect high-gradient regions
        batch, heads, seq_len, _ = states.shape
        bottlenecks = torch.zeros(batch, heads, seq_len - 1, device=states.device)

        for i in range(seq_len - 1):
            # Concatenate adjacent states
            state_pairs = torch.cat(
                [states[..., i : i + 1, :], states[..., i + 1 : i + 2, :]], dim=-1
            )

            # Check for bottleneck
            bottleneck_score = self.bottleneck_detector(state_pairs)
            bottlenecks[..., i] = bottleneck_score.squeeze(-1)

        # Compute bottleneck impact
        impact = torch.where(
            bottlenecks > self.bottleneck_threshold,
            flow[..., :-1],
            torch.zeros_like(flow[..., :-1]),
        )

        return bottlenecks, impact

    def predict_optimal_path(
        self, states: torch.Tensor, start_idx: torch.Tensor
    ) -> torch.Tensor:
        """Predict optimal path through the state sequence."""
        # Initialize path
        batch, heads, seq_len, _ = states.shape
        device = states.device

        current_idx = start_idx
        path = torch.zeros(batch, heads, seq_len, dtype=torch.long, device=device)
        path[..., 0] = current_idx

        # Hidden state for path predictor
        hidden = None

        # Generate path
        for t in range(seq_len - 1):
            # Current state
            current_state = torch.gather(
                states,
                dim=-2,
                index=current_idx.unsqueeze(-1).expand(-1, -1, -1, self.dim),
            )

            # Predict next position
            out, hidden = self.path_predictor(
                current_state.view(-1, 1, self.dim), hidden
            )

            # Compute scores for all positions
            scores = self.route_scorer(states.view(-1, seq_len, self.dim)).view(
                batch, heads, seq_len
            )

            # Mask already visited positions
            mask = torch.zeros_like(scores, dtype=torch.bool)
            mask.scatter_(-1, path[..., : t + 1], True)
            scores = scores.masked_fill(mask, float("-inf"))

            # Add exploration noise
            if self.training:
                noise = torch.randn_like(scores) * self.exploration_rate
                scores = scores + noise

            # Select next position
            current_idx = scores.argmax(dim=-1)
            path[..., t + 1] = current_idx

        return path

    def optimize_bottlenecks(
        self, states: torch.Tensor, bottlenecks: torch.Tensor, impact: torch.Tensor
    ) -> torch.Tensor:
        """Optimize states to reduce bottleneck impact."""
        # Find significant bottlenecks
        significant = (impact.abs() > self.bottleneck_threshold).nonzero()

        if len(significant) == 0:
            return states

        # Optimize each bottleneck
        optimized_states = states.clone()
        for b, h, t in significant:
            # Get adjacent states
            state_1 = states[b, h, t]
            state_2 = states[b, h, t + 1]

            # Interpolate to reduce bottleneck
            alpha = impact[b, h, t].abs()
            optimized_states[b, h, t + 1] = (1 - alpha) * state_2 + alpha * state_1

        return optimized_states

    def forward(
        self, states: torch.Tensor, return_diagnostics: bool = False
    ) -> dict[str, torch.Tensor]:
        """Process states through the information routing system."""
        # Get pattern dynamics
        pattern_info = self.pattern_dynamics(states, return_patterns=True)

        routing_scores = pattern_info["routing_scores"]

        # Detect and handle bottlenecks
        bottlenecks, impact = self.detect_bottlenecks(states, routing_scores)

        # Optimize states
        optimized_states = self.optimize_bottlenecks(states, bottlenecks, impact)

        # Predict optimal paths
        start_idx = torch.zeros(
            states.shape[0], states.shape[1], 1, dtype=torch.long, device=states.device
        )
        optimal_paths = self.predict_optimal_path(optimized_states, start_idx)

        results = {
            "states": optimized_states,
            "paths": optimal_paths,
            "routing_scores": routing_scores,
        }

        if return_diagnostics:
            results.update(
                {
                    "bottlenecks": bottlenecks,
                    "impact": impact,
                    "patterns": pattern_info["patterns"],
                    "pattern_scores": pattern_info["pattern_scores"],
                }
            )

        return results
