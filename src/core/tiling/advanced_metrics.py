"""Advanced Metrics Analyzer.

This module implements advanced metrics computation for the quantum geometric attention framework.
It provides metrics for analyzing information flow, computational efficiency, and adaptation quality.
"""

import torch
import numpy as np
from typing import List, Optional

class AdvancedMetricsAnalyzer:
    """Analyzer for computing advanced metrics in quantum geometric attention."""
    
    def __init__(self):
        """Initialize the metrics analyzer."""
        self.history = {
            "ifq": [],  # Information Flow Quality
            "cer": [],  # Compute-to-Efficiency Ratio
            "ae": []    # Adaptation Efficiency
        }
        
    def compute_ifq(
        self,
        pattern_stability: float,
        cross_tile_flow: float,
        edge_utilization: float,
        info_density: float,
        alpha: float = 0.25  # Weight for each component
    ) -> float:
        """Compute Information Flow Quality (IFQ).
        
        IFQ measures the quality of information processing by combining:
        1. Pattern stability over time
        2. Cross-tile information flow
        3. Edge attention utilization
        4. Information density
        
        Args:
            pattern_stability: Stability measure of attention patterns
            cross_tile_flow: Amount of information flow between tiles
            edge_utilization: Utilization of edge attention mechanisms
            info_density: Density of information in attention patterns
            alpha: Weight for each component
            
        Returns:
            IFQ score between 0 and 1
        """
        components = [
            pattern_stability,
            cross_tile_flow,
            edge_utilization,
            info_density
        ]
        
        # Normalize components
        components = [max(0.0, min(1.0, c)) for c in components]
        
        # Weighted sum
        ifq = sum(alpha * c for c in components)
        
        # Store in history
        self.history["ifq"].append(ifq)
        
        return ifq
        
    def compute_cer(
        self,
        information_transferred: float,
        compute_cost: float,
        memory_usage: float,
        resolution: float,
        beta: float = 0.5  # Balance between compute and memory
    ) -> float:
        """Compute Compute-to-Efficiency Ratio (CER).
        
        CER measures computational efficiency by relating:
        1. Amount of information transferred
        2. Computational cost
        3. Memory usage
        4. Resolution of processing
        
        Args:
            information_transferred: Amount of information processed
            compute_cost: Computational cost metric
            memory_usage: Memory usage metric
            resolution: Current resolution of processing
            beta: Weight between compute and memory costs
            
        Returns:
            CER score (higher is better)
        """
        # Normalize inputs
        info = max(1e-6, information_transferred)
        compute = max(1e-6, compute_cost)
        memory = max(1e-6, memory_usage)
        res = max(1e-6, resolution)
        
        # Compute efficiency ratio
        resource_cost = beta * compute + (1 - beta) * memory
        cer = (info * res) / resource_cost
        
        # Store in history
        self.history["cer"].append(cer)
        
        return cer
        
    def compute_ae(
        self,
        resolution_history: List[float],
        load_variance_history: List[float],
        window_size: int = 10
    ) -> float:
        """Compute Adaptation Efficiency (AE).
        
        AE measures how well the system adapts by analyzing:
        1. Resolution changes over time
        2. Load balancing effectiveness
        3. Adaptation smoothness
        
        Args:
            resolution_history: History of resolution values
            load_variance_history: History of load variance values
            window_size: Window size for smoothness computation
            
        Returns:
            AE score between 0 and 1
        """
        if not resolution_history or not load_variance_history:
            return 1.0
            
        # Compute resolution adaptation smoothness
        res_diffs = [abs(resolution_history[i+1] - resolution_history[i])
                    for i in range(len(resolution_history)-1)]
        smoothness = 1.0 / (1.0 + np.mean(res_diffs) if res_diffs else 1.0)
        
        # Compute load balancing effectiveness
        load_balance = 1.0 / (1.0 + np.mean(load_variance_history))
        
        # Combine metrics
        ae = 0.5 * (smoothness + load_balance)
        
        # Store in history
        self.history["ae"].append(ae)
        
        return ae
        
    def get_history(self, metric: str) -> List[float]:
        """Get history of a specific metric."""
        return self.history.get(metric, [])
