"""Pattern formation validation implementation.

This module validates pattern formation:
- Pattern emergence
- Spatial organization
- Temporal evolution
"""

from typing import Optional, List, Tuple, Dict, Any, Union, cast, Mapping, TypeVar, Protocol
from dataclasses import dataclass
import torch
import numpy as np
from scipy import signal, ndimage
from scipy.fft import fft2, ifft2
from numpy.typing import NDArray, ArrayLike
from torch import Tensor

from src.neural.attention.pattern.dynamics import PatternDynamics
from src.neural.flow.geometric_flow import GeometricFlow
from src.neural.flow.hamiltonian import HamiltonianSystem
from src.validation.base import ValidationResult

T = TypeVar('T', bound=Union[float, bool, torch.Tensor])

class MetricsDict(Protocol[T]):
    def __getitem__(self, key: str) -> T: ...
    def __setitem__(self, key: str, value: T) -> None: ...
    def update(self, other: Mapping[str, T]) -> None: ...

def _ensure_tensor(x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    """Convert numpy array to tensor if needed."""
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    return x

def _ensure_numpy(x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Convert tensor to numpy array if needed."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x

def _convert_to_float(x: Union[torch.Tensor, np.ndarray, float, int]) -> float:
    """Convert tensor/array to float."""
    if isinstance(x, (torch.Tensor, np.ndarray)):
        return float(x.item()) if x.size == 1 else float(x)
    return float(x)

@dataclass
class SpatialMetrics:
    """Metrics for spatial pattern properties."""
    
    def __init__(
        self,
        threshold: float = 0.1,
        min_size: int = 5
    ):
        """Initialize spatial metrics.
        
        Args:
            threshold: Threshold for pattern detection
            min_size: Minimum size for pattern features
        """
        self.threshold = threshold
        self.min_size = min_size
        
    def compute_spatial_statistics(
        self,
        data: torch.Tensor
    ) -> Dict[str, Union[float, Tensor]]:
        """Compute spatial statistics of pattern.
        
        Args:
            data: Input data tensor
            
        Returns:
            Dictionary with spatial statistics
        """
        # Convert to numpy
        data_np = _ensure_numpy(data)
        
        # Threshold data
        binary = data_np > self.threshold
        
        # Create structure for labeling (cross-shaped for 2D)
        structure = np.array([
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]
        ], dtype=np.int32)
        
        # Ensure binary array is proper type
        binary = binary.astype(np.int32)
        
        # Label connected components
        labels_arr, num_features = cast(
            Tuple[NDArray[np.int32], int],
            ndimage.label(binary, structure=structure)
        )
        
        if num_features == 0:
            return {
                "num_features": 0,
                "mean_area": 0.0,
                "std_area": 0.0,
                "total_area": 0.0,
                "density": 0.0
            }
        
        # Create index array for feature properties
        feature_indices = np.arange(1, num_features + 1, dtype=np.int32)
        
        # Compute feature properties using binary array
        areas = ndimage.sum(
            binary.astype(np.float32),  # Ensure float type for sum
            labels_arr,
            index=feature_indices
        )
        
        centers = ndimage.center_of_mass(
            data_np,
            labels_arr,
            index=feature_indices
        )
        
        # Filter small features
        valid = areas >= self.min_size
        areas = areas[valid]
        centers = [c for i, c in enumerate(centers) if valid[i]]
        
        # Compute statistics
        stats = {
            "num_features": len(areas),
            "mean_area": float(np.mean(areas)) if len(areas) > 0 else 0,
            "std_area": float(np.std(areas)) if len(areas) > 0 else 0,
            "total_area": float(np.sum(areas)),
            "density": float(np.sum(areas)) / data_np.size
        }
        
        if len(centers) > 1:
            # Compute distances between centers
            distances = []
            for i in range(len(centers)):
                for j in range(i + 1, len(centers)):
                    dist = np.sqrt(
                        sum((a - b) ** 2 for a, b in zip(centers[i], centers[j]))
                    )
                    distances.append(dist)
                    
            stats.update({
                "mean_spacing": float(np.mean(distances)),
                "min_spacing": float(np.min(distances)),
                "max_spacing": float(np.max(distances))
            })
            
        return stats
        
    def compute_spatial_correlations(
        self,
        data: torch.Tensor,
        max_distance: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute spatial correlations."""
        data_np = data.detach().cpu().numpy()
        height, width = data_np.shape[-2:]
        max_dist = max_distance if max_distance is not None else min(height, width) // 2
            
        # Initialize arrays with explicit types
        distances: NDArray[np.int32] = np.arange(max_dist, dtype=np.int32)
        correlations: NDArray[np.float32] = np.zeros_like(distances, dtype=np.float32)
        
        # Compute correlations at each distance
        for d in distances:
            # Create shifted arrays
            shifted_h = np.roll(data_np, d, axis=-2)
            shifted_w = np.roll(data_np, d, axis=-1)
            
            # Compute correlations
            corr_h = np.corrcoef(
                data_np.reshape(-1),
                shifted_h.reshape(-1)
            )[0, 1]
            corr_w = np.corrcoef(
                data_np.reshape(-1),
                shifted_w.reshape(-1)
            )[0, 1]
            
            # Average correlations
            correlations[d] = (corr_h + corr_w) / 2
            
        return (
            torch.tensor(distances, device=data.device),
            torch.tensor(correlations, device=data.device)
        )
        
    def analyze_spatial_structure(
        self,
        data: torch.Tensor
    ) -> Dict[str, Union[torch.Tensor, float]]:
        """Analyze spatial structure of pattern."""
        # Compute statistics
        stats = self.compute_spatial_statistics(data)
        
        # Compute correlations
        distances, correlations = self.compute_spatial_correlations(data)
        
        # Find characteristic length
        char_length = 0.0
        if len(correlations) > 1:
            # Find first minimum in correlations
            mins = signal.find_peaks(-correlations.cpu().numpy())[0]
            if len(mins) > 0:
                char_length = float(distances[mins[0]].item())
            else:
                char_length = float(distances[-1].item())
            
        return {
            **stats,
            "distances": distances,
            "correlations": correlations,
            "characteristic_length": char_length
        }


@dataclass 
class TemporalMetrics:
    """Metrics for temporal pattern evolution."""
    
    def __init__(
        self,
        window_size: int = 10,
        overlap: int = 5
    ):
        """Initialize temporal metrics."""
        self.window_size = window_size
        self.overlap = overlap
        
    def compute_temporal_statistics(
        self,
        data: torch.Tensor,
        dt: float = 1.0
    ) -> Dict[str, Union[float, Tensor]]:
        """Compute temporal statistics."""
        # Convert to numpy for gradient calculation
        data_np = data.detach().cpu().numpy()
        grad = np.gradient(data_np, dt, axis=0)
        
        # Convert back to tensor for operations
        grad_tensor = torch.from_numpy(grad).to(data.device)
        
        stats: Dict[str, Union[float, Tensor]] = {
            "mean_rate": float(torch.mean(torch.abs(grad_tensor)).item()),
            "max_rate": float(torch.max(torch.abs(grad_tensor)).item()),
            "std_rate": float(torch.std(grad_tensor).item()),
            "mean_amplitude": float(torch.mean(torch.abs(data)).item()),
            "max_amplitude": float(torch.max(torch.abs(data)).item()),
            "std_amplitude": float(torch.std(data).item())
        }
        
        return stats
        
    def compute_temporal_correlations(
        self,
        data: torch.Tensor,
        dt: float = 1.0,
        max_lag: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute temporal autocorrelations."""
        if max_lag is None:
            max_lag = data.shape[0] // 4
            
        # Initialize arrays with explicit types
        lags: NDArray[np.int_] = np.arange(max_lag, dtype=np.int_)
        correlations = torch.zeros(max_lag, device=data.device)
        
        # Compute correlations at each lag
        data_flat = data.reshape(data.shape[0], -1)
        mean = torch.mean(data_flat, dim=1, keepdim=True)
        std = torch.std(data_flat, dim=1, keepdim=True)
        normalized = (data_flat - mean) / (std + 1e-8)
        
        for lag in range(max_lag):
            if lag < data.shape[0]:
                corr = torch.mean(
                    normalized[:-lag if lag > 0 else None] * 
                    normalized[lag:],
                    dim=1
                ).mean()
                correlations[lag] = corr
                
        return torch.tensor(lags, device=data.device), correlations
        
    def analyze_temporal_structure(
        self,
        data: torch.Tensor,
        dt: float = 1.0
    ) -> Dict[str, Union[float, Tensor, bool]]:
        """Analyze temporal structure of pattern."""
        # Compute statistics
        stats = self.compute_temporal_statistics(data)
        
        # Compute correlations
        lags, correlations = self.compute_temporal_correlations(data, dt)
        
        # Find characteristic time
        char_time = 0.0
        if len(correlations) > 1:
            # Find first minimum in correlations
            corr_np = correlations.cpu().numpy()
            mins = signal.find_peaks(-corr_np)[0]
            if len(mins) > 0:
                char_time = float(lags[mins[0]].item()) * dt
            else:
                char_time = float(lags[-1].item()) * dt
                
        # Detect oscillations
        is_oscillatory = False
        if len(correlations) > 1:
            # Look for sign changes in derivative
            corr_grad = torch.gradient(correlations)[0]
            sign_changes = torch.where(corr_grad[:-1] * corr_grad[1:] < 0)[0]
            is_oscillatory = len(sign_changes) > 1
            
        return {
            **stats,
            "lags": lags * dt,
            "correlations": correlations,
            "characteristic_time": char_time,
            "is_oscillatory": bool(is_oscillatory)
        }


@dataclass
class PatternModeAnalyzer:
    """Analyze patterns into spatial modes."""
    
    def __init__(
        self,
        num_modes: int = 10,
        threshold: float = 0.1
    ):
        """Initialize mode analyzer."""
        self.num_modes = num_modes
        self.threshold = threshold
        
    def compute_spatial_modes(
        self,
        data: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute spatial modes using FFT."""
        # Convert to numpy for FFT
        data_np: NDArray[np.float64] = data.detach().cpu().numpy()
        
        # Compute 2D FFT
        fft_data = np.asarray(fft2(data_np), dtype=np.complex128)
        
        # Get magnitudes
        magnitudes = np.abs(fft_data)
        
        # Find peak frequencies
        peaks = []
        for i in range(self.num_modes):
            # Find maximum magnitude
            max_idx = np.unravel_index(
                np.argmax(magnitudes),
                magnitudes.shape
            )
            peaks.append(max_idx)
            
            # Zero out peak and surrounding region
            magnitudes[
                max_idx[0]-1:max_idx[0]+2,
                max_idx[1]-1:max_idx[1]+2
            ] = 0
            
        # Extract modes and amplitudes
        modes = []
        amplitudes = []
        for peak in peaks:
            # Get complex mode
            mode = np.zeros_like(fft_data)
            mode[peak] = fft_data[peak]
            
            # Convert back to spatial domain
            spatial_mode = np.asarray(ifft2(mode), dtype=np.complex128).real
            
            # Compute amplitude
            amplitude = np.abs(fft_data[peak])
            
            if amplitude > self.threshold:
                modes.append(spatial_mode)
                amplitudes.append(amplitude)
                
        # Convert to tensors
        modes_tensor = torch.tensor(
            np.stack(modes),
            device=data.device
        )
        amplitudes_tensor = torch.tensor(
            amplitudes,
            device=data.device
        )
        
        return modes_tensor, amplitudes_tensor
        
    def reconstruct_pattern(
        self,
        modes: torch.Tensor,
        amplitudes: torch.Tensor
    ) -> torch.Tensor:
        """Reconstruct pattern from modes.
        
        Args:
            modes: Spatial modes
            amplitudes: Mode amplitudes
            
        Returns:
            Reconstructed pattern
        """
        # Weight modes by amplitudes
        weighted = modes * amplitudes.view(-1, 1, 1)
        
        # Sum all modes
        return torch.sum(weighted, dim=0)
        
    def compute_mode_correlations(
        self,
        modes: torch.Tensor
    ) -> torch.Tensor:
        """Compute correlations between modes.
        
        Args:
            modes: Spatial modes
            
        Returns:
            Correlation matrix
        """
        # Flatten modes
        flat_modes = modes.reshape(modes.shape[0], -1)
        
        # Compute correlation matrix
        correlations = torch.corrcoef(flat_modes)
        
        return correlations
        
    def analyze_modes(
        self,
        data: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Analyze spatial modes in pattern.
        
        Args:
            data: Input data tensor
            
        Returns:
            Dictionary with analysis results
        """
        # Compute modes
        modes, amplitudes = self.compute_spatial_modes(data)
        
        # Compute correlations
        correlations = self.compute_mode_correlations(modes)
        
        # Reconstruct pattern
        reconstruction = self.reconstruct_pattern(modes, amplitudes)
        
        # Compute reconstruction error
        error = torch.mean((data - reconstruction) ** 2)
        
        return {
            "modes": modes,
            "amplitudes": amplitudes,
            "correlations": correlations,
            "reconstruction": reconstruction,
            "error": error
        }


@dataclass
class EmergenceMetrics:
    """Metrics for measuring pattern emergence."""

    def __init__(
        self,
        threshold: float = 0.1,
        window_size: int = 10,
        min_peaks: int = 2
    ):
        """Initialize emergence metrics.
        
        Args:
            threshold: Threshold for pattern detection
            window_size: Window size for temporal analysis
            min_peaks: Minimum number of peaks for pattern
        """
        self.threshold = threshold
        self.window_size = window_size
        self.min_peaks = min_peaks
        
    def detect_pattern(
        self,
        data: torch.Tensor,
        time_axis: int = -1
    ) -> bool:
        """Detect if pattern has emerged in data."""
        # Compute temporal average
        mean = torch.mean(data, dim=time_axis, keepdim=True)
        
        # Compute temporal variance
        var = torch.var(data, dim=time_axis, keepdim=True)
        
        # Check if variance exceeds threshold
        return bool(torch.any(var > self.threshold * mean).item())
        
    def find_peaks(
        self,
        data: torch.Tensor,
        spatial_axis: int = -1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Find peaks in spatial pattern.
        
        Args:
            data: Input data tensor
            spatial_axis: Axis representing spatial dimension
            
        Returns:
            Tuple of (peak_positions, peak_heights)
        """
        # Convert to numpy for peak finding
        data_np = data.detach().cpu().numpy()
        
        # Find peaks along spatial axis
        peaks, properties = signal.find_peaks(
            data_np,
            height=self.threshold,
            distance=self.window_size
        )
        
        # Convert back to torch tensors
        peak_positions = torch.tensor(peaks, device=data.device)
        peak_heights = torch.tensor(
            properties["peak_heights"],
            device=data.device
        )
        
        return peak_positions, peak_heights
        
    def compute_emergence_time(
        self,
        data: torch.Tensor,
        time_points: torch.Tensor,
        time_axis: int = -1
    ) -> Optional[float]:
        """Compute time when pattern emerges.
        
        Args:
            data: Input data tensor
            time_points: Time points corresponding to data
            time_axis: Axis representing time dimension
            
        Returns:
            Time of pattern emergence, or None if no pattern
        """
        # Compute temporal variance
        var = torch.var(data, dim=time_axis, keepdim=True)
        
        # Find first time variance exceeds threshold
        emergence_idx = torch.where(var > self.threshold)[0]
        
        if len(emergence_idx) > 0:
            return float(time_points[emergence_idx[0]].item())
        return None
        
    def validate_emergence(
        self,
        data: torch.Tensor,
        time_points: torch.Tensor
    ) -> Dict[str, Union[bool, Optional[float]]]:
        """Validate pattern emergence."""
        # Check for pattern presence
        has_pattern = self.detect_pattern(data)
        
        # Find peaks if pattern present
        if has_pattern:
            peaks, heights = self.find_peaks(data)
            sufficient_peaks = len(peaks) >= self.min_peaks
        else:
            sufficient_peaks = False
            
        # Compute emergence time
        emergence_time = self.compute_emergence_time(data, time_points)
        
        return {
            "has_pattern": has_pattern,
            "sufficient_peaks": sufficient_peaks,
            "emergence_time": emergence_time,
            "is_valid": has_pattern and sufficient_peaks
        }


@dataclass
class EmergenceValidation:
    """Results of pattern emergence validation."""

    emerged: bool  # Pattern emergence
    formation_time: float  # Time to form
    coherence: float  # Pattern coherence
    stability: float  # Formation stability


@dataclass
class SpatialValidation:
    """Results of spatial pattern validation."""

    wavelength: float  # Pattern wavelength
    symmetry: str  # Symmetry type
    defects: int  # Number of defects
    correlation: float  # Spatial correlation


@dataclass
class TemporalValidation:
    """Results of temporal pattern validation."""

    frequency: float  # Oscillation frequency
    phase_locked: bool  # Phase locking
    drift_rate: float  # Pattern drift
    persistence: float  # Temporal persistence


@dataclass
class BifurcationPoint:
    """Bifurcation point data."""
    
    parameter_value: float  # Value of bifurcation parameter
    pattern_type: str  # Type of pattern after bifurcation
    eigenvalues: torch.Tensor  # Eigenvalues at bifurcation
    eigenvectors: torch.Tensor  # Eigenvectors at bifurcation


class EmergenceValidator:
    """Validates pattern emergence properties."""

    def __init__(self, tolerance: float = 1e-6, coherence_threshold: float = 0.8):
        """Initialize emergence validator.
        
        Args:
            tolerance: Numerical tolerance for computations
            coherence_threshold: Threshold for determining pattern emergence
        """
        self.tolerance = tolerance
        self.threshold = coherence_threshold

    def validate_emergence(self, trajectory: torch.Tensor) -> EmergenceValidation:
        """Validate pattern emergence from trajectory.
        
        Args:
            trajectory: Tensor of shape (time_steps, height, width) containing system states
            
        Returns:
            EmergenceValidation containing emergence metrics
        """
        # Get initial and final states
        initial_state = trajectory[0]
        final_state = trajectory[-1]
        
        # Compute emergence metrics
        coherence = self._compute_coherence(final_state)
        stability = self._compute_stability(trajectory)
        emerged = coherence > self.threshold
        
        # Compute formation time if emerged
        formation_time = 0.0
        if emerged:
            formation_time = self._compute_formation_time(trajectory)

        return EmergenceValidation(
            emerged=emerged,
            formation_time=formation_time,
            coherence=coherence,
            stability=stability
        )

    def _compute_coherence(self, state: torch.Tensor) -> float:
        """Compute coherence of pattern.

        Args:
            state: Pattern state tensor

        Returns:
            float: Coherence score
        """
        # Ensure we have a 2D tensor by taking the first batch and channel if present
        if state.ndim > 2:
            state = state.squeeze()  # Remove singleton dimensions
            if state.ndim > 2:
                state = state[0]  # Take first batch if still > 2D
                if state.ndim > 2:
                    state = state[0]  # Take first channel if still > 2D
        
        padding = 2
        height, width = state.shape
        # Add padding for circular boundary conditions
        # For 2D input, pad needs to be specified as (left, right, top, bottom)
        padded = torch.nn.functional.pad(
            state.unsqueeze(0).unsqueeze(0),  # Add batch and channel dims for padding
            (padding, padding, padding, padding),
            mode="circular"
        ).squeeze()  # Remove the extra dimensions
        
        # Compute local coherence using circular cross-correlation
        coherence = 0.0
        for i in range(padding * 2 + 1):
            for j in range(padding * 2 + 1):
                if i == padding and j == padding:
                    continue
                shifted = torch.roll(padded, shifts=(i - padding, j - padding), dims=(0, 1))
                
                # Compute correlation by stacking tensors
                stacked = torch.stack([padded[padding:-padding, padding:-padding].flatten(), shifted[padding:-padding, padding:-padding].flatten()])
                correlation = torch.corrcoef(stacked)[0, 1]
                coherence += correlation
                
        return float(coherence / ((padding * 2 + 1) ** 2 - 1))

    def _compute_stability(self, trajectory: torch.Tensor) -> float:
        """Compute temporal stability of pattern."""
        # Use normalized variance of final states as stability measure
        final_states = trajectory[-10:]  # Look at last 10 states
        variance = float(torch.var(final_states).item())
        max_val = float(torch.max(torch.abs(final_states)).item())
        if max_val > 0:
            stability = 1.0 - min(1.0, variance / max_val)
        else:
            stability = 1.0
        return float(stability)

    def _compute_formation_time(self, trajectory: torch.Tensor) -> float:
        """Compute time taken for pattern to emerge."""
        # Find first time coherence exceeds threshold
        for t in range(len(trajectory)):
            coherence = self._compute_coherence(trajectory[t])
            if coherence > self.threshold:
                return float(t) / len(trajectory)
        return 1.0


class SpatialValidator:
    """Validation of spatial pattern properties."""

    def __init__(self, symmetry_threshold: float = 0.9, defect_threshold: float = 0.1):
        self.symmetry_threshold = symmetry_threshold
        self.defect_threshold = defect_threshold

    def validate_spatial(self, pattern: torch.Tensor) -> SpatialValidation:
        """Validate spatial pattern properties."""
        # Compute wavelength
        wavelength = self._compute_wavelength(pattern)

        # Analyze symmetry
        symmetry = self._analyze_symmetry(pattern)

        # Count defects
        defects = self._count_defects(pattern)

        # Compute correlation
        correlation = self._compute_correlation(pattern)

        return SpatialValidation(
            wavelength=wavelength.item(),
            symmetry=symmetry,
            defects=defects,
            correlation=correlation.item(),
        )

    def _compute_wavelength(self, pattern: torch.Tensor) -> torch.Tensor:
        """Compute dominant wavelength from pattern.

        Args:
            pattern: Pattern tensor of shape (batch_size, channels, height, width)

        Returns:
            Wavelength tensor of shape (batch_size, channels)
        """
        # Get size and create frequency grid
        N = pattern.shape[-1]
        freqs = torch.fft.fftfreq(N, dtype=torch.float32, device=pattern.device)
        
        # Create 2D frequency grids for x and y separately
        freqs_x = freqs[None, :].expand(N, N)
        freqs_y = freqs[:, None].expand(N, N)
        
        # Compute power spectrum
        fft = torch.fft.fft2(pattern)
        power = torch.abs(fft).pow(2)
        
        # Create mask for valid frequencies (exclude DC and above Nyquist)
        nyquist = 0.5
        mask_x = (freqs_x.abs() > 0) & (freqs_x.abs() <= nyquist)  # Explicitly exclude DC
        mask_y = (freqs_y.abs() > 0) & (freqs_y.abs() <= nyquist)  # Explicitly exclude DC
        mask = mask_x | mask_y  # Use OR to capture peaks in either direction
        
        # Get valid frequencies and reshape power
        batch_shape = power.shape[:-2]  # (batch_size, channels)
        power_valid = power.reshape(*batch_shape, -1)[..., mask.reshape(-1)]
        freqs_x_valid = freqs_x[mask]
        freqs_y_valid = freqs_y[mask]
        
        # Find peak frequency for each batch and channel
        peak_idx = torch.argmax(power_valid, dim=-1)  # Shape: (batch_size, channels)
        peak_freq_x = freqs_x_valid[peak_idx]
        peak_freq_y = freqs_y_valid[peak_idx]
        
        # For 2D patterns, we need to consider that peaks can appear in either x or y direction
        # The wavelength is determined by the highest frequency component
        peak_freqs = torch.maximum(peak_freq_x.abs(), peak_freq_y.abs())
        
        # Convert to wavelength in pixels
        # fftfreq gives frequencies in cycles per N samples, normalized by N
        # To get wavelength in pixels: wavelength = 1 / f_fft
        wavelength = 1.0 / peak_freqs  # Since freqs are already normalized
        
        # Ensure output shape is correct for batched input
        if len(pattern.shape) == 4:  # Batch case
            wavelength = wavelength.reshape(-1, 1, 1)  # Make it (batch_size, 1, 1)

        return wavelength

    def _analyze_symmetry(self, pattern: torch.Tensor) -> str:
        """Analyze pattern symmetry."""
        # Check different symmetries
        symmetries = {
            "translation": self._check_translation(pattern),
            "rotation": self._check_rotation(pattern),
            "reflection": self._check_reflection(pattern),
        }

        # Return dominant symmetry
        return max(symmetries.items(), key=lambda x: x[1])[0]

    def _check_translation(self, pattern: torch.Tensor) -> float:
        """Check translational symmetry."""
        shifted = torch.roll(pattern, shifts=1, dims=0)
        x = torch.stack([pattern.reshape(-1), shifted.reshape(-1)])
        correlation = torch.corrcoef(x)[0, 1]
        return float(correlation.item())

    def _check_rotation(self, pattern: torch.Tensor) -> float:
        """Check rotational symmetry."""
        rotated = torch.rot90(pattern, k=1)
        x = torch.stack([pattern.reshape(-1), rotated.reshape(-1)])
        correlation = torch.corrcoef(x)[0, 1]
        return float(correlation.item())

    def _check_reflection(self, pattern: torch.Tensor) -> float:
        """Check reflection symmetry."""
        reflected = torch.flip(pattern, dims=[0])
        x = torch.stack([pattern.reshape(-1), reflected.reshape(-1)])
        correlation = torch.corrcoef(x)[0, 1]
        return float(correlation.item())

    def _count_defects(self, pattern: torch.Tensor) -> int:
        """Count number of defects in pattern."""
        # Ensure pattern is 2D
        if pattern.ndim > 2:
            pattern = pattern.squeeze()  # Remove singleton dimensions
            if pattern.ndim > 2:
                pattern = pattern[0]  # Take first batch if still > 2D
                if pattern.ndim > 2:
                    pattern = pattern[0]  # Take first channel if still > 2D

        # Use gradient magnitude
        dx = pattern[1:, :] - pattern[:-1, :]  # Shape: [H-1, W]
        dy = pattern[:, 1:] - pattern[:, :-1]  # Shape: [H, W-1]

        # Pad gradients to match original size
        dx = torch.nn.functional.pad(dx, (0, 0, 0, 1), mode='constant', value=0)  # Pad height
        dy = torch.nn.functional.pad(dy, (0, 1, 0, 0), mode='constant', value=0)  # Pad width

        # Now dx and dy should both be [H, W]
        gradient_mag = torch.sqrt(dx ** 2 + dy ** 2)

        # Count high gradient points
        defects = torch.sum(gradient_mag > self.defect_threshold)
        return int(defects.item())

    def _compute_correlation(self, pattern: torch.Tensor) -> torch.Tensor:
        """Compute spatial correlation."""
        # Ensure pattern is 2D
        if pattern.ndim > 2:
            pattern = pattern.squeeze()  # Remove singleton dimensions
            if pattern.ndim > 2:
                pattern = pattern[0]  # Take first batch if still > 2D
                if pattern.ndim > 2:
                    pattern = pattern[0]  # Take first channel if still > 2D
                    
        # Use average local correlation
        padding = 2
        height, width = pattern.shape
        correlations = []

        # Compute correlations at different offsets
        for i in range(-padding, padding + 1):
            for j in range(-padding, padding + 1):
                if i == 0 and j == 0:
                    continue

                # Shift pattern
                shifted = torch.roll(pattern, shifts=(i, j), dims=(0, 1))
                
                # Compute correlation by stacking tensors
                stacked = torch.stack([pattern[padding:-padding, padding:-padding].flatten(), shifted[padding:-padding, padding:-padding].flatten()])
                correlation = torch.corrcoef(stacked)[0, 1]
                correlations.append(correlation.item())

        # Return average correlation
        return torch.tensor(sum(correlations) / len(correlations))


class TemporalValidator:
    """Validation of temporal pattern properties."""

    def __init__(self, frequency_threshold: float = 0.1, phase_threshold: float = 0.1):
        self.frequency_threshold = frequency_threshold
        self.phase_threshold = phase_threshold

    def validate_temporal(self, trajectory: torch.Tensor) -> TemporalValidation:
        """Validate temporal pattern properties."""
        # Compute frequency
        frequency = self._compute_frequency(trajectory)

        # Check phase locking
        phase_locked = self._check_phase_locking(trajectory)

        # Compute drift
        drift = self._compute_drift(trajectory)

        # Measure persistence
        persistence = self._measure_persistence(trajectory)

        return TemporalValidation(
            frequency=frequency.item(),
            phase_locked=phase_locked,
            drift_rate=drift.item(),
            persistence=persistence.item(),
        )

    def _compute_frequency(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Compute pattern oscillation frequency."""
        # Use temporal Fourier transform
        fft = torch.fft.fft(trajectory, dim=0)
        power = torch.mean(torch.abs(fft) ** 2, dim=(1, 2))
        
        # Get frequencies and find dominant frequency
        freqs = torch.fft.fftfreq(trajectory.size(0))
        # Only look at positive frequencies (excluding DC)
        positive_freqs = freqs[1:len(freqs)//2]  # Exclude DC and negative frequencies
        positive_power = power[1:len(freqs)//2]   # Corresponding power spectrum
        
        # Find dominant frequency index
        dominant_idx = torch.argmax(positive_power)
        
        return torch.abs(positive_freqs[dominant_idx])

    def _check_phase_locking(self, trajectory: torch.Tensor) -> bool:
        """Check if pattern is phase locked."""
        # Compute phase using Hilbert transform
        analytic = torch.view_as_real(torch.fft.fft(trajectory, dim=0))
        phase = torch.atan2(analytic[..., 1], analytic[..., 0])

        # Check phase coherence
        phase_diff = phase[1:] - phase[:-1]
        coherence = torch.std(phase_diff)

        return bool((coherence < self.phase_threshold).item())

    def _compute_drift(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Compute pattern drift rate."""
        # Use center of mass motion
        com = torch.mean(trajectory, dim=(1, 2))
        drift = torch.mean(torch.abs(com[1:] - com[:-1]))

        return drift

    def _measure_persistence(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Measure pattern persistence."""
        # Use temporal autocorrelation
        mean = torch.mean(trajectory, dim=0)
        centered = trajectory - mean

        norm = torch.sum(centered[0] ** 2)
        if norm == 0:
            return torch.tensor(0.0)

        correlations = []
        for t in range(len(trajectory)):
            corr = torch.sum(centered[0] * centered[t]) / norm
            correlations.append(corr)

        return torch.mean(torch.tensor(correlations))


class BifurcationAnalyzer:
    """Analysis of pattern bifurcations."""

    def __init__(
        self,
        parameter_range: Tuple[float, float],
        num_points: int = 100,
        tolerance: float = 1e-6
    ):
        """Initialize bifurcation analyzer."""
        self.parameter_range = parameter_range
        self.num_points = num_points
        self.tolerance = tolerance
        self.bifurcation_points: List[BifurcationPoint] = []

    def analyze_bifurcations(
        self,
        dynamics: Optional[PatternDynamics],
        initial_state: torch.Tensor,
        parameter_name: Optional[str] = None,
    ) -> List[BifurcationPoint]:
        """Analyze bifurcations in pattern dynamics."""
        if dynamics is None or parameter_name is None:
            return []

        # Sample parameter values
        param_values = torch.linspace(
            self.parameter_range[0], self.parameter_range[1], self.num_points
        )

        # Track states and eigenvalues
        prev_state = None
        prev_eigenvals = None
        current_param = getattr(dynamics, parameter_name)

        for param_value in param_values:
            # Update parameter
            setattr(dynamics, parameter_name, param_value)

            # Evolve to steady state
            state = self._evolve_to_steady_state(dynamics, initial_state)

            # Compute stability
            eigenvals, eigenvecs = self._compute_stability(dynamics, state)

            # Check for bifurcation
            if prev_state is not None and prev_eigenvals is not None:
                if self._detect_bifurcation(
                    state, prev_state, eigenvals, prev_eigenvals
                ):
                    pattern_type = self._classify_pattern(state)
                    self.bifurcation_points.append(
                        BifurcationPoint(
                            parameter_value=float(param_value),
                            pattern_type=pattern_type,
                            eigenvalues=eigenvals,
                            eigenvectors=eigenvecs,
                        )
                    )

            prev_state = state
            prev_eigenvals = eigenvals

        # Restore original parameter
        setattr(dynamics, parameter_name, current_param)

        return self.bifurcation_points

    def _evolve_to_steady_state(
        self,
        dynamics: PatternDynamics,
        initial_state: torch.Tensor,
        max_steps: int = 1000
    ) -> torch.Tensor:
        """Evolve system to steady state.
        
        Args:
            dynamics: Pattern dynamics
            initial_state: Initial state
            max_steps: Maximum evolution steps
            
        Returns:
            Steady state pattern
        """
        state = initial_state
        
        for _ in range(max_steps):
            new_state = dynamics.step(state)
            
            # Check convergence
            if torch.allclose(new_state, state, rtol=self.tolerance):
                return new_state
                
            state = new_state
            
        return state

    def _compute_stability(
        self,
        dynamics: PatternDynamics,
        state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute stability analysis at state.
        
        Args:
            dynamics: Pattern dynamics
            state: Pattern state
            
        Returns:
            Tuple of (eigenvalues, eigenvectors)
        """
        # Compute Jacobian
        jacobian = dynamics.compute_jacobian(state)
        
        # Compute eigendecomposition
        eigenvals, eigenvecs = torch.linalg.eigh(jacobian)
        
        return eigenvals, eigenvecs

    def _detect_bifurcation(
        self,
        state: torch.Tensor,
        prev_state: torch.Tensor,
        eigenvals: torch.Tensor,
        prev_eigenvals: torch.Tensor
    ) -> bool:
        """Detect if bifurcation occurred."""
        # Check for qualitative changes
        state_change = not torch.allclose(
            state, prev_state,
            rtol=self.tolerance
        )
        
        # Check eigenvalue crossing
        stability_change = torch.any(
            torch.sign(eigenvals.real) != torch.sign(prev_eigenvals.real)
        )
        
        return bool(state_change or stability_change.item())

    def _classify_pattern(self, state: torch.Tensor) -> str:
        """Classify pattern type based on spatial structure.
        
        Args:
            state: Pattern state tensor
            
        Returns:
            String describing pattern type
        """
        # Compute basic statistics
        mean = torch.mean(state)
        std = torch.std(state)
        
        # Compute spatial FFT
        fft = torch.fft.fft2(state)
        power = torch.abs(fft)
        
        # Analyze symmetry
        rotated = torch.rot90(state)
        reflected = torch.flip(state, [0])
        rot_sim = torch.mean((state - rotated) ** 2)
        ref_sim = torch.mean((state - reflected) ** 2)
        
        # Classify based on properties
        if std < 0.1 * mean:
            return "homogeneous"
        elif rot_sim < 0.1 and ref_sim < 0.1:
            return "symmetric"
        elif torch.max(power[1:]) > 2 * torch.mean(power[1:]):
            return "periodic"
        else:
            return "irregular"


@dataclass
class FormationValidationResult(ValidationResult[Dict[str, Any]]):
    """Validation result for pattern formation that wraps spatial and temporal metrics."""
    
    def __init__(self, is_valid: bool, message: str, data: Optional[Dict[str, Any]] = None):
        """Initialize formation validation result.
        
        Args:
            is_valid: Whether validation passed
            message: Description of validation result
            data: Optional validation data or metrics
        """
        super().__init__(is_valid, message, data)
    
    def merge(self, other: ValidationResult) -> 'FormationValidationResult':
        """Merge with another validation result."""
        if not isinstance(other, FormationValidationResult):
            raise TypeError(f"Cannot merge with {type(other)}")
            
        # Merge metrics dictionaries carefully
        merged_data = {**(self.data or {})}
        other_data = other.data or {}
        
        # Special handling for metrics
        for key, value in other_data.items():
            if key in merged_data and isinstance(value, dict):
                merged_data[key].update(value)
            else:
                merged_data[key] = value
        
        return FormationValidationResult(
            is_valid=bool(self.is_valid and other.is_valid),
            message=f"{self.message}; {other.message}",
            data=merged_data
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "is_valid": bool(self.is_valid),
            "message": self.message,
            "data": self.data or {}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FormationValidationResult':
        """Create from dictionary representation."""
        return cls(
            is_valid=bool(data["is_valid"]),
            message=data["message"],
            data=data.get("data", {})
        )

@dataclass
class FormationValidator:
    """Main validator that combines spatial and temporal metrics with the validation interface."""
    
    spatial_metrics: SpatialMetrics
    temporal_metrics: TemporalMetrics
    
    def __init__(
        self,
        spatial_threshold: float = 0.1,
        min_size: int = 5,
        window_size: int = 10,
        overlap: int = 5
    ):
        """Initialize formation validator with both metrics systems."""
        self.spatial_metrics = SpatialMetrics(
            threshold=spatial_threshold,
            min_size=min_size
        )
        self.temporal_metrics = TemporalMetrics(
            window_size=window_size,
            overlap=overlap
        )
    
    def validate_formation(
        self,
        pattern: torch.Tensor,
        dt: float = 1.0
    ) -> ValidationResult:
        """Validate pattern formation using both spatial and temporal metrics."""
        try:
            # Compute spatial metrics
            spatial_stats = self.spatial_metrics.analyze_spatial_structure(pattern)
            
            # Compute temporal metrics if pattern has time dimension
            temporal_stats = {}
            if len(pattern.shape) > 2:  # Has time dimension
                temporal_stats = self.temporal_metrics.compute_temporal_statistics(pattern, dt)
            
            # Determine validity based on metrics
            has_features = bool(spatial_stats.get("num_features", 0) > 0)
            has_structure = bool(spatial_stats.get("characteristic_length", 0) > 0)
            
            # Prepare validation message
            if has_features and has_structure:
                msg = (
                    f"Valid pattern formation with {spatial_stats['num_features']} features "
                    f"and characteristic length {spatial_stats['characteristic_length']:.2f}"
                )
            else:
                msg = "Pattern lacks sufficient spatial organization"
            
            # Return validation result with all metrics
            return FormationValidationResult(
                is_valid=bool(has_features and has_structure),
                message=msg,
                data={
                    "spatial_metrics": spatial_stats,
                    "temporal_metrics": temporal_stats
                }
            )
            
        except Exception as e:
            return FormationValidationResult(
                is_valid=False,
                message=f"Formation validation failed: {str(e)}",
                data={}
            )
    
    def validate_spatial(self, pattern: torch.Tensor) -> ValidationResult:
        """Validate only spatial aspects of pattern formation."""
        try:
            stats = self.spatial_metrics.analyze_spatial_structure(pattern)
            has_features = bool(stats.get("num_features", 0) > 0)
            
            return FormationValidationResult(
                is_valid=has_features,
                message=f"Pattern has {stats['num_features']} features",
                data={"spatial_metrics": stats}
            )
            
        except Exception as e:
            return FormationValidationResult(
                is_valid=False,
                message=f"Spatial validation failed: {str(e)}",
                data={}
            )
    
    def validate_temporal(
        self,
        pattern: torch.Tensor,
        dt: float = 1.0
    ) -> ValidationResult:
        """Validate only temporal aspects of pattern formation."""
        try:
            stats = self.temporal_metrics.compute_temporal_statistics(pattern, dt)
            has_dynamics = bool(stats.get("mean_rate", 0) > 0)
            
            return FormationValidationResult(
                is_valid=has_dynamics,
                message=f"Pattern has mean rate {stats['mean_rate']:.3f}",
                data={"temporal_metrics": stats}
            )
            
        except Exception as e:
            return FormationValidationResult(
                is_valid=False,
                message=f"Temporal validation failed: {str(e)}",
                data={}
            )

class PatternFormationValidator:
    """Complete pattern formation validation system."""
    
    def __init__(
        self,
        tolerance: float = 1e-6,
        coherence_threshold: float = 0.8,
        symmetry_threshold: float = 0.9,
        defect_threshold: float = 0.1,
        frequency_threshold: float = 0.1,
        phase_threshold: float = 0.1,
    ):
        """Initialize validator.
        
        Args:
            tolerance: Numerical tolerance
            coherence_threshold: Threshold for pattern coherence
            symmetry_threshold: Threshold for pattern symmetry
            defect_threshold: Threshold for pattern defects
            frequency_threshold: Threshold for temporal frequency
            phase_threshold: Threshold for phase locking
        """
        self.emergence_validator = EmergenceValidator(tolerance, coherence_threshold)
        self.spatial_validator = SpatialValidator(symmetry_threshold, defect_threshold)
        self.temporal_validator = TemporalValidator(
            frequency_threshold, phase_threshold
        )
    
    def validate(
        self,
        dynamics: Optional[PatternDynamics],
        initial: torch.Tensor,
        time_steps: int = 1000,
    ) -> ValidationResult:
        """Perform complete pattern formation validation.
        
        Args:
            dynamics: Pattern dynamics system (optional)
            initial: Initial pattern state
            time_steps: Number of time steps to simulate
            
        Returns:
            ValidationResult with is_valid=True if pattern formation is valid
        """
        # Initialize trajectory with initial state
        current = initial
        trajectory = [current]

        # Evolve system if dynamics provided
        if dynamics is not None:
            for _ in range(time_steps - 1):
                current = dynamics.step(current)
                trajectory.append(current)
        else:
            # If no dynamics, use initial state for validation
            trajectory = [initial] * time_steps

        # Convert to tensor
        trajectory = torch.stack(trajectory)

        # Validate emergence
        emergence = self.emergence_validator.validate_emergence(trajectory)

        # Validate spatial organization  
        spatial = self.spatial_validator.validate_spatial(trajectory[-1])

        # Validate temporal evolution
        temporal = self.temporal_validator.validate_temporal(trajectory)
        
        # Combine validation results
        is_valid = (
            emergence.emerged and
            spatial.correlation > self.spatial_validator.symmetry_threshold and
            temporal.persistence > self.temporal_validator.phase_threshold
        )
        
        return FormationValidationResult(
            is_valid=is_valid,
            message=f"Pattern formation validation: {'passed' if is_valid else 'failed'}",
            data={
                "emergence": {
                    "formation_time": emergence.formation_time,
                    "coherence": emergence.coherence,
                    "stability": emergence.stability
                },
                "spatial": {
                    "wavelength": spatial.wavelength,
                    "symmetry": spatial.symmetry,
                    "defects": spatial.defects,
                    "correlation": spatial.correlation
                },
                "temporal": {
                    "frequency": temporal.frequency,
                    "phase_locked": temporal.phase_locked,
                    "drift_rate": temporal.drift_rate,
                    "persistence": temporal.persistence
                }
            }
        )

def _detect_bifurcation(
    eigenvals: torch.Tensor,
    prev_eigenvals: torch.Tensor,
    threshold: float = 0.1
) -> bool:
    """Detect bifurcation from eigenvalue changes."""
    if prev_eigenvals is None:
        return False
        
    # Convert to real values and compute changes
    real_curr = eigenvals.real
    real_prev = prev_eigenvals.real
    
    # Compute relative changes
    changes = torch.abs(real_curr - real_prev) / (torch.abs(real_prev) + 1e-8)
    
    # Detect significant changes
    return bool((changes > threshold).any().item())

def _compute_pattern_metrics(
    data: torch.Tensor,
    threshold: float = 0.1
) -> Dict[str, Union[float, bool]]:
    """Compute pattern formation metrics."""
    # Convert to numpy for processing
    data_np = data.detach().cpu().numpy()
    
    # Compute basic statistics
    mean_val = float(np.mean(data_np))
    std_val = float(np.std(data_np))
    max_val = float(np.max(data_np))
    min_val = float(np.min(data_np))
    
    # Detect pattern formation
    has_pattern = bool(std_val > threshold * (max_val - min_val))
    
    return {
        "mean": mean_val,
        "std": std_val,
        "max": max_val,
        "min": min_val,
        "has_pattern": has_pattern
    }

def _convert_metrics_to_float(metrics: Dict[str, Any]) -> Dict[str, Union[float, bool]]:
    """Convert metric values to float or bool."""
    result = {}
    for key, value in metrics.items():
        if isinstance(value, (torch.Tensor, np.ndarray)):
            if value.size == 1:  # Single value
                result[key] = float(value.item())
            else:
                continue  # Skip non-scalar tensors
        elif isinstance(value, (bool, float)):
            result[key] = value
        elif isinstance(value, (int, np.number)):
            result[key] = float(value)
    return result

def _merge_metrics(
    base: Dict[str, Any],
    update: Dict[str, Any]
) -> Dict[str, Union[float, bool, torch.Tensor]]:
    """Merge metric dictionaries with type conversion."""
    result = dict(base)  # Make a copy
    
    for key, value in update.items():
        if isinstance(value, (torch.Tensor, np.ndarray)):
            if value.size == 1:  # Single value
                result[key] = float(value.item())
            else:
                result[key] = torch.as_tensor(value) if isinstance(value, np.ndarray) else value
        elif isinstance(value, (bool, float)):
            result[key] = value
        elif isinstance(value, (int, np.number)):
            result[key] = float(value)
            
    return result

def analyze_pattern_formation(
    data: torch.Tensor,
    dt: float = 1.0,
    threshold: float = 0.1
) -> Dict[str, Union[float, bool, torch.Tensor]]:
    """Analyze pattern formation in data."""
    # Initialize metrics with type hints
    metrics: Dict[str, Union[float, bool, torch.Tensor]] = {}
    
    # Compute basic metrics
    basic_metrics = _compute_pattern_metrics(data, threshold)
    metrics.update(basic_metrics)
    
    # Compute temporal evolution
    if data.dim() > 3:  # Has time dimension
        temporal = TemporalMetrics()
        time_metrics = temporal.analyze_temporal_structure(data, dt)
        for key, value in time_metrics.items():
            if isinstance(value, (torch.Tensor, float, bool)):
                metrics[key] = value
    
    # Compute spatial structure
    if data.dim() > 1:  # Has spatial dimensions
        spatial = SpatialMetrics(threshold=threshold)
        space_metrics = spatial.analyze_spatial_structure(data)
        for key, value in space_metrics.items():
            if isinstance(value, (torch.Tensor, float, bool)):
                metrics[key] = value
    
    return metrics

def _detect_pattern_coherence(
    data: torch.Tensor,
    threshold: float = 0.1
) -> bool:
    """Detect pattern coherence."""
    # Use torch operations for better type safety
    data_flat = data.reshape(-1)
    mean = torch.mean(data_flat)
    std = torch.std(data_flat)
    normalized = (data_flat - mean) / (std + 1e-8)
    coherence = torch.max(torch.abs(normalized))
    
    return bool(coherence > threshold)

def _compute_eigenvalue_metrics(
    eigenvals: torch.Tensor,
    prev_eigenvals: Optional[torch.Tensor] = None,
    threshold: float = 0.1
) -> Dict[str, Union[float, bool]]:
    """Compute eigenvalue-based metrics."""
    # Convert to numpy for processing
    eig_np = eigenvals.detach().cpu().numpy()
    
    # Compute basic statistics using numpy for complex operations
    max_real = float(np.max(np.real(eig_np)))
    min_real = float(np.min(np.real(eig_np)))
    max_imag = float(np.max(np.abs(np.imag(eig_np))))
    
    # Detect bifurcation if previous eigenvalues available
    has_bifurcation = False
    if prev_eigenvals is not None:
        prev_np = prev_eigenvals.detach().cpu().numpy()
        real_change = np.abs(np.real(eig_np) - np.real(prev_np))
        has_bifurcation = bool(np.any(real_change > threshold))
    
    return {
        "max_real": max_real,
        "min_real": min_real,
        "max_imag": max_imag,
        "has_bifurcation": has_bifurcation
    }

def _compute_energy_metrics(
    data: torch.Tensor,
    dt: float = 1.0
) -> Dict[str, float]:
    """Compute energy-based metrics."""
    # Convert to numpy for processing
    data_np = data.detach().cpu().numpy()
    
    # Compute energy metrics
    energy = np.sum(data_np**2, axis=tuple(range(1, data_np.ndim)))
    grad = np.gradient(energy, dt)
    
    return {
        "mean_energy": float(np.mean(energy)),
        "max_energy": float(np.max(energy)),
        "energy_rate": float(np.mean(np.abs(grad)))
    }

def analyze_pattern_stability(
    data: torch.Tensor,
    eigenvals: Optional[torch.Tensor] = None,
    prev_eigenvals: Optional[torch.Tensor] = None,
    dt: float = 1.0,
    threshold: float = 0.1
) -> Dict[str, Union[float, bool]]:
    """Analyze pattern stability."""
    # Initialize metrics with type hints
    metrics: Dict[str, Union[float, bool]] = {}
    
    # Compute pattern coherence
    metrics["is_coherent"] = _detect_pattern_coherence(data, threshold)
    
    # Compute energy metrics
    energy_metrics = _compute_energy_metrics(data, dt)
    for key, value in energy_metrics.items():
        metrics[key] = float(value)
    
    # Compute eigenvalue metrics if available
    if eigenvals is not None:
        eig_metrics = _compute_eigenvalue_metrics(eigenvals, prev_eigenvals, threshold)
        for key, value in eig_metrics.items():
            if isinstance(value, bool):
                metrics[key] = value
            else:
                metrics[key] = float(value)
    
    return metrics
