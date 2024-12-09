"""Advanced metrics for adaptive attention tiling.

This module implements the following metrics:
1. Information Flow Quality (IFQ)
2. Computational Efficiency Ratio (CER)
3. Adaptation Effectiveness (AE)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

from src.core.common.constants import (
    AE_ACCURACY_WEIGHT,
    AE_HISTORY_WINDOW,
    AE_MAX_OSCILLATION,
    AE_SPEED_WEIGHT,
    AE_STABILITY_WEIGHT,
    CER_MEMORY_FACTOR,
    CER_MIN_COMPUTE,
    CER_RESOLUTION_FACTOR,
    IFQ_DENSITY_WEIGHT,
    IFQ_EDGE_WEIGHT,
    IFQ_FLOW_WEIGHT,
    IFQ_PATTERN_WEIGHT,
)

logger = logging.getLogger(__name__)


class AdvancedMetricsAnalyzer:
    """Analyzer for advanced attention metrics."""

    def __init__(self) -> None:
        """Initialize analyzer."""
        self._history: list[dict[str, Any]] = []
        self._adaptive_weights = {
            'ifq': {
                'pattern': IFQ_PATTERN_WEIGHT,
                'flow': IFQ_FLOW_WEIGHT,
                'edge': IFQ_EDGE_WEIGHT,
                'density': IFQ_DENSITY_WEIGHT
            },
            'cer': {
                'memory': CER_MEMORY_FACTOR,
                'resolution': CER_RESOLUTION_FACTOR
            },
            'ae': {
                'stability': AE_STABILITY_WEIGHT,
                'speed': AE_SPEED_WEIGHT,
                'accuracy': AE_ACCURACY_WEIGHT
            }
        }
        self._learning_rate = 0.01
        self._weight_history = []

    def _update_weights(self, metric_performance: dict[str, float]) -> None:
        """Update weights based on their effectiveness.
        
        Args:
            metric_performance: Dictionary of metric performances used to adjust weights
        """
        if len(self._history) < 2:
            return
            
        # Calculate metric improvements
        prev_metrics = self._history[-2]
        curr_metrics = self._history[-1]
        
        # Update weights based on performance improvements
        for metric, components in self._adaptive_weights.items():
            if metric in prev_metrics and metric in curr_metrics:
                improvement = curr_metrics[metric] - prev_metrics[metric]
                
                # Skip if no improvement data
                if metric not in metric_performance:
                    continue
                    
                # Update weights based on contribution to improvement
                total_weight = sum(components.values())
                if total_weight <= 0:
                    continue
                    
                for component, weight in components.items():
                    contribution = weight / total_weight
                    new_weight = weight + self._learning_rate * improvement * contribution
                    
                    # Normalize to ensure weights sum to 1
                    self._adaptive_weights[metric][component] = max(0.1, min(0.5, new_weight))
                    
        # Store weight history
        self._weight_history.append(self._adaptive_weights.copy())
        
        # Normalize weights
        for metric, components in self._adaptive_weights.items():
            total = sum(components.values())
            if total > 0:
                for component in components:
                    components[component] /= total

    def compute_ifq(
        self,
        pattern_stability: float,
        cross_tile_flow: float,
        edge_utilization: float,
        info_density: float,
    ) -> float:
        """Compute Information Flow Quality metric with adaptive weights."""
        pattern_stability = np.clip(pattern_stability, 0, 1)
        cross_tile_flow = np.clip(cross_tile_flow, 0, 1)
        edge_utilization = np.clip(edge_utilization, 0, 1)
        info_density = np.clip(info_density, 0, 1)

        weights = self._adaptive_weights['ifq']
        ifq = (
            weights['pattern'] * pattern_stability
            + weights['flow'] * cross_tile_flow
            + weights['edge'] * edge_utilization
            + weights['density'] * info_density
        )

        return float(ifq)

    def compute_cer(
        self,
        information_transferred: float,
        compute_cost: float,
        memory_usage: float,
        resolution: float,
    ) -> float:
        """Compute Computational Efficiency Ratio with adaptive weights."""
        compute_cost = max(compute_cost, CER_MIN_COMPUTE)
        
        weights = self._adaptive_weights['cer']
        memory_factor = 1.0 + (weights['memory'] * (memory_usage / (1024 * 1024)))
        resolution_factor = 1.0 + (weights['resolution'] * (1.0 - resolution))

        cer = information_transferred / (compute_cost * memory_factor * resolution_factor)
        return float(cer)

    def compute_ae(
        self,
        resolution_history: list[float],
        load_variance_history: list[float],
        target_resolution: float | None = None,
    ) -> float:
        """Compute Adaptation Effectiveness with adaptive weights.
        
        For empty history, returns 0.0 as there's no evidence of adaptation.
        For single entries, returns 0.5 as a neutral score.
        For longer histories, computes a weighted score of stability, speed, and accuracy.
        """
        try:
            if not resolution_history or not load_variance_history:
                return 0.0  # No adaptation history available

            # Ensure we have matching history lengths
            min_len = min(len(resolution_history), len(load_variance_history))
            resolution_history = resolution_history[-min_len:]
            load_variance_history = load_variance_history[-min_len:]

            if min_len == 1:
                return 0.5  # Neutral score for single entry - can't assess adaptation

            # Compute component scores
            stability_score = self._compute_stability_score(resolution_history)
            speed_score = self._compute_speed_score(load_variance_history, resolution_history)
            accuracy_score = self._compute_accuracy_score(resolution_history, load_variance_history, target_resolution)

            # Apply weights from self._adaptive_weights
            weights = self._adaptive_weights['ae']
            ae = (
                weights['stability'] * stability_score
                + weights['speed'] * speed_score
                + weights['accuracy'] * accuracy_score
            )

            # Add oscillation penalty if present
            if min_len > 2:
                oscillation = np.mean(np.abs(np.diff(np.diff(resolution_history))))
                if oscillation > AE_MAX_OSCILLATION:
                    ae *= (1.0 - min(oscillation, 0.5))  # Reduce score for excessive oscillation

            # Ensure the result is between 0 and 1
            return float(np.clip(ae, 0, 1))

        except Exception as e:
            logger.warning("Error computing AE metric: %s", e)
            return 0.0  # Return lowest score on error

    def _compute_stability_score(self, resolution_history: list[float]) -> float:
        """Compute stability score based on resolution changes.
        
        Returns a score between 0 and 1, with higher scores for more stable patterns.
        Perfect stability (no changes) gets a moderate score since some adaptivity is desired.
        """
        if len(resolution_history) <= 1:
            return 0.5  # Neutral score for insufficient history

        # Calculate normalized changes
        resolution_changes = np.diff(resolution_history)
        max_possible_change = 1.0  # Since resolution is between 0 and 1
        normalized_changes = np.abs(resolution_changes) / max_possible_change
        
        # Calculate stability metrics
        changes_mean = float(np.mean(normalized_changes))
        changes_std = float(np.std(normalized_changes)) if len(normalized_changes) > 1 else 0.0
        
        # Penalize both too much stability (no adaptation) and too much change
        stability_score = 1.0 - (changes_mean * 2.0 + changes_std)  # More aggressive penalty
        
        # Cap at 0.8 for perfect stability to encourage some adaptation
        if changes_mean < 1e-6:
            stability_score = 0.8
            
        return float(np.clip(stability_score, 0.0, 0.8))

    def _compute_speed_score(self, load_variance_history: list[float], resolution_history: list[float]) -> float:
        """Compute speed score based on load variance and resolution changes.
        
        Higher scores indicate better adaptation speed - quick response to load changes
        but without overshooting.
        """
        if len(load_variance_history) <= 1 or len(resolution_history) <= 1:
            return 0.5  # Neutral score for insufficient history

        # Calculate load variance changes and resolution responses
        load_changes = np.diff(load_variance_history)
        resolution_responses = np.diff(resolution_history)
        
        if np.all(np.abs(load_changes) < 1e-6):
            return 0.6  # Moderate score when no load changes needed response
            
        # Calculate response ratio and timing
        significant_load_changes = np.abs(load_changes) > 0.05  # More sensitive threshold
        significant_responses = np.abs(resolution_responses) > 0.01
        
        if not any(significant_load_changes):
            return 0.6  # Moderate score when no significant load changes
            
        response_ratio = np.sum(significant_responses & significant_load_changes) / np.sum(significant_load_changes)
        
        # Penalize overshooting - responses should be proportional to load changes
        response_proportionality = np.mean(
            np.clip(1.0 - np.abs(np.abs(resolution_responses) - np.abs(load_changes)), 0, 1)
        )
        
        # Combine scores with emphasis on proportionality
        speed_score = 0.4 * response_ratio + 0.6 * response_proportionality
        
        return float(np.clip(speed_score, 0.0, 0.9))  # Cap at 0.9 to encourage improvement

    def _compute_accuracy_score(self, resolution_history: list[float], load_variance_history: list[float], target_resolution: float | None = None) -> float:
        """Compute accuracy score based on resolution history and target.
        
        Higher scores indicate better accuracy in meeting targets while maintaining
        reasonable load variance.
        """
        if len(resolution_history) <= 1:
            return 0.5  # Neutral score for insufficient history
            
        # Calculate base accuracy from target or load variance
        if target_resolution is not None:
            # Calculate deviation from target resolution with increasing penalty
            deviations = np.abs(np.array(resolution_history) - target_resolution)
            accuracy_score = 1.0 - float(np.mean(np.power(np.clip(deviations, 0, 1), 1.5)))
        else:
            # Use load variance as proxy for accuracy when no target
            # Exponential penalty for high variance
            mean_variance = float(np.mean(np.clip(load_variance_history, 0, 1)))
            accuracy_score = np.exp(-2.0 * mean_variance)
        
        # Add penalty for oscillation around target/optimal point
        if len(resolution_history) > 2:
            oscillation = np.mean(np.abs(np.diff(np.diff(resolution_history))))
            oscillation_penalty = np.clip(oscillation * 2.0, 0, 0.3)
            accuracy_score -= oscillation_penalty
            
        return float(np.clip(accuracy_score, 0.0, 0.9))  # Cap at 0.9 to encourage improvement

    def add_metrics(self, metrics: dict[str, Any]) -> None:
        """Add metrics snapshot to history.

        Args:
            metrics: Dictionary of current metrics
        """
        self._history.append(metrics)
        self._update_weights(metrics)

    def get_history(self) -> list[dict[str, Any]]:
        """Get metrics history.

        Returns:
            List of metrics snapshots
        """
        return self._history

    def clear_history(self) -> None:
        """Clear metrics history."""
        self._history.clear()

    def compute_metrics(
        self,
        pattern_stability: float,
        cross_tile_flow: float,
        edge_utilization: float,
        info_density: float,
        compute_cost: float,
        memory_usage: float,
        resolution: float,
        resolution_history: list[float],
        load_variance_history: list[float],
        target_resolution: float | None = None,
        load_distribution: torch.Tensor | list[float] | None = None,
    ) -> dict[str, float]:
        """Compute all advanced metrics.

        Args:
            pattern_stability: Stability of attention patterns (0-1)
            cross_tile_flow: Information flow between tiles (0-1)
            edge_utilization: Edge token utilization (0-1)
            info_density: Information density (0-1)
            compute_cost: Computational cost
            memory_usage: Memory usage in bytes
            resolution: Current resolution
            resolution_history: History of resolution values
            load_variance_history: History of load variance values
            target_resolution: Optional target resolution for accuracy
            load_distribution: Optional tensor or list representing load distribution across tiles

        Returns:
            Dictionary containing all computed metrics
        """
        # Compute Information Flow Quality
        ifq = self.compute_ifq(
            pattern_stability,
            cross_tile_flow,
            edge_utilization,
            info_density,
        )

        # Compute Computational Efficiency Ratio using IFQ as information_transferred
        cer = self.compute_cer(
            information_transferred=ifq,
            compute_cost=compute_cost,
            memory_usage=memory_usage,
            resolution=resolution,
        )

        # Compute Adaptation Effectiveness
        ae = self.compute_ae(
            resolution_history=resolution_history,
            load_variance_history=load_variance_history,
            target_resolution=target_resolution,
        )

        # Create metrics dictionary
        metrics = {
            'ifq': ifq,
            'cer': cer,
            'ae': ae,
            'pattern_stability': pattern_stability,
            'cross_tile_flow': cross_tile_flow,
            'edge_utilization': edge_utilization,
            'info_density': info_density,
            'compute_cost': compute_cost,
            'memory_usage': memory_usage,
            'resolution': resolution,
            'load_distribution': load_distribution if isinstance(load_distribution, list) else 
                               load_distribution.detach().cpu().numpy().tolist() if load_distribution is not None else None,
        }

        # Add metrics to history
        self.add_metrics(metrics)

        return metrics
