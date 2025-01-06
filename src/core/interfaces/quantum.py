"""
Core quantum interfaces defining the protocol for quantum state management and operations.
These interfaces will serve as the foundation for unifying quantum implementations across layers.
"""

from __future__ import annotations  # Enable forward references
from typing import Protocol, TypeVar, Tuple, List, Optional, Dict, Any, Callable, Union, Sequence, Generic
import torch
from dataclasses import dataclass
from enum import Enum

T = TypeVar('T', bound=torch.Tensor)

@dataclass
class EntanglementMetrics:
    """Metrics for quantum entanglement properties."""
    entropy: float
    concurrence: Optional[float]
    negativity: float
    witness_value: float
    purity: float

@dataclass
class MeasurementResult(Generic[T]):
    """Result of a quantum measurement."""
    expectation: torch.Tensor
    variance: torch.Tensor
    collapsed_state: Optional[IQuantumState[T]]
    probability_distribution: torch.Tensor

@dataclass
class BlochVector(Generic[T]):
    """Bloch sphere representation of a qubit state."""
    x: float
    y: float
    z: float
    phase: float = 0.0
    purity: float = 1.0

class EvolutionType(Enum):
    """Types of quantum evolution."""
    UNITARY = "unitary"
    DISSIPATIVE = "dissipative"
    ADIABATIC = "adiabatic"
    GEOMETRIC = "geometric"

class IQuantumState(Protocol[T]):
    """Protocol defining the interface for quantum states.
    
    This interface unifies quantum state implementations across different layers:
    - Core quantum operations (core.quantum.types.QuantumState)
    - Neural quantum states ( src.neural.attention.pattern.quantum.QuantumState)
    - Pattern-specific states (core.tiling.quantum_attention_tile.QuantumState)
    """
    
    @property
    def dimension(self) -> int:
        """The dimension of the quantum state space."""
        ...
    
    @property
    def is_pure(self) -> bool:
        """Whether the state is a pure state."""
        ...
    
    @property
    def hilbert_space(self) -> HilbertSpace[T]:
        """The Hilbert space this state belongs to."""
        ...
    
    def to_bloch_vector(self) -> T:
        """Convert state to Bloch sphere representation for qubit systems."""
        ...
    
    def from_bloch_vector(self, vector: T) -> IQuantumState[T]:
        """Create quantum state from Bloch vector representation."""
        ...
    
    def compute_expectation(self, observable: T) -> float:
        """Compute expectation value of an observable directly."""
        ...
    
    def apply_unitary(self, unitary: T) -> IQuantumState[T]:
        """Apply unitary transformation to the state."""
        ...
    
    def evolve(self, 
               hamiltonian: T, 
               time: float,
               evolution_type: EvolutionType = EvolutionType.UNITARY,
               **kwargs) -> IQuantumState[T]:
        """Evolve the quantum state according to the given Hamiltonian.
        
        Args:
            hamiltonian: The Hamiltonian operator driving the evolution
            time: Evolution time
            evolution_type: Type of evolution to perform
            **kwargs: Additional evolution parameters
            
        Returns:
            The evolved quantum state
        """
        ...
    
    def measure(self, 
                observable: T,
                collapse: bool = False) -> MeasurementResult[T]:
        """Measure an observable on the quantum state.
        
        Args:
            observable: The observable to measure
            collapse: Whether to collapse the state after measurement
            
        Returns:
            MeasurementResult containing measurement statistics
        """
        ...
    
    def to_density_matrix(self) -> T:
        """Convert the state to its density matrix representation."""
        ...
    
    def compute_entropy(self) -> float:
        """Compute the von Neumann entropy of the state."""
        ...
    
    def entanglement_metrics(self, 
                           partition: Optional[Tuple[Sequence[int], Sequence[int]]] = None) -> EntanglementMetrics:
        """Compute various entanglement metrics for the state.
        
        Args:
            partition: Optional bipartition of the system for computing entanglement
            
        Returns:
            EntanglementMetrics containing various entanglement measures
        """
        ...
    
    def parallel_transport(self, 
                         connection: T,
                         path: T) -> IQuantumState[T]:
        """Parallel transport the state along a path using the given connection.
        
        Args:
            connection: The connection defining parallel transport
            path: The path along which to transport
            
        Returns:
            The parallel transported state
        """
        ...
    
    def tensor_product(self, other: IQuantumState[T]) -> IQuantumState[T]:
        """Compute tensor product with another quantum state."""
        ...
    
    def partial_trace(self, subsystem: Sequence[int]) -> IQuantumState[T]:
        """Compute partial trace over specified subsystem."""
        ...
    
    def apply_channel(self, 
                     kraus_operators: Sequence[T]) -> IQuantumState[T]:
        """Apply a quantum channel defined by Kraus operators."""
        ...
    
    def fidelity(self, other: IQuantumState[T]) -> float:
        """Compute quantum fidelity with another state."""
        ...
    
    def to_device(self, device: torch.device) -> IQuantumState[T]:
        """Move the state to specified device."""
        ...
    
    def clone(self) -> IQuantumState[T]:
        """Create a copy of the quantum state."""
        ...
    
    # New Quantum Field Theory Methods
    def to_field_configuration(self) -> T:
        """Convert state to field configuration representation."""
        ...
    
    def compute_action(self, lagrangian: Callable[[T], float]) -> float:
        """Compute action functional for the state using given Lagrangian."""
        ...
    
    def propagate(self, 
                 time: float,
                 hamiltonian_density: Callable[[T, T], T],
                 boundary_conditions: Optional[Dict[str, T]] = None) -> IQuantumState[T]:
        """Propagate state using field equations of motion."""
        ...
    
    def compute_correlation(self,
                          other: IQuantumState[T],
                          distance: float) -> T:
        """Compute field correlation function at given distance."""
        ...
    
    def to_momentum_space(self) -> IQuantumState[T]:
        """Transform state to momentum space representation."""
        ...
    
    def apply_field_operator(self,
                           operator: Callable[[T], T],
                           position: Optional[T] = None) -> IQuantumState[T]:
        """Apply field operator at given position (or all positions if None)."""
        ...

    # Geometric Methods
    def compute_berry_phase(self, 
                          path: T,
                          connection: Optional[T] = None) -> float:
        """Compute Berry phase along given path."""
        ...
    
    def to_geometric_tensor(self) -> T:
        """Convert to quantum geometric tensor representation."""
        ...
    
    def compute_metric_tensor(self, 
                            parameter_space: Optional[T] = None) -> T:
        """Compute quantum metric tensor in parameter space."""
        ...

    # Pattern Integration Methods
    def to_pattern_state(self) -> PatternState[T]:
        """Convert to pattern-compatible state representation."""
        ...
    
    def apply_pattern_flow(self, 
                          flow: GeometricFlow[T],
                          time: float) -> IQuantumState[T]:
        """Evolve state according to pattern-based geometric flow."""
        ...
    
    def compute_pattern_stability(self) -> Dict[str, float]:
        """Compute stability metrics for pattern formation."""
        ...

class IQuantumRegister(Protocol[T]):
    """Protocol for quantum registers managing multiple quantum states."""
    
    @property
    def size(self) -> int:
        """Number of qubits in the register."""
        ...
    
    @property
    def state(self) -> IQuantumState[T]:
        """Current state of the entire register."""
        ...
    
    @property
    def hilbert_space(self) -> HilbertSpace[T]:
        """The Hilbert space of the register."""
        ...
    
    def initialize(self, 
                  state: Union[str, Sequence[int], T]) -> None:
        """Initialize the register to a given state."""
        ...
    
    def apply_gate(self,
                  gate: T,
                  target_qubits: Sequence[int],
                  control_qubits: Optional[Sequence[int]] = None) -> None:
        """Apply a quantum gate to specified qubits."""
        ...
    
    def measure_qubit(self, 
                     qubit: int,
                     basis: Optional[T] = None) -> Tuple[int, float]:
        """Measure a specific qubit."""
        ...
    
    def reset(self) -> None:
        """Reset register to initial state."""
        ...

    # Field Theory Integration
    def to_field_register(self) -> FieldRegister[T]:
        """Convert to field theory register representation."""
        ...
    
    def apply_field_operator(self,
                           operator: Callable[[T], T],
                           qubits: Optional[Sequence[int]] = None) -> None:
        """Apply field operator to specified qubits."""
        ...
    
    def compute_correlation_matrix(self) -> T:
        """Compute correlation matrix between qubits."""
        ...

    # Pattern Integration
    def to_pattern_register(self) -> PatternRegister[T]:
        """Convert to pattern-based register representation."""
        ...
    
    def apply_geometric_operation(self,
                                metric: T,
                                connection: T,
                                qubits: Sequence[int]) -> None:
        """Apply geometric operation using metric and connection."""
        ...
    
    def compute_entanglement_structure(self) -> Dict[str, Any]:
        """Analyze entanglement structure of register."""
        ...

    # Advanced Operations
    def apply_channel_to_qubits(self,
                               channel: IQuantumChannel[T],
                               qubits: Sequence[int]) -> None:
        """Apply quantum channel to specific qubits."""
        ...
    
    def create_bell_pair(self,
                        qubit1: int,
                        qubit2: int) -> None:
        """Create Bell pair between two qubits."""
        ...
    
    def apply_error_correction(self,
                             code_type: str,
                             logical_qubits: Sequence[int],
                             physical_qubits: Sequence[int]) -> None:
        """Apply quantum error correction encoding."""
        ...
    
    def measure_stabilizers(self,
                          stabilizer_generators: Sequence[T]) -> Dict[str, float]:
        """Measure stabilizer operators."""
        ...
    
    def compute_reduced_density_matrix(self,
                                     qubits: Sequence[int]) -> T:
        """Compute reduced density matrix for subset of qubits."""
        ...

@dataclass
class CircuitMetrics:
    """Metrics for quantum circuit analysis."""
    depth: int
    gate_count: Dict[str, int]
    entangling_operations: int
    classical_operations: int
    measurement_count: int

class CircuitOptimization(Enum):
    """Types of circuit optimizations."""
    GATE_CANCELLATION = "gate_cancellation"
    COMMUTATION = "commutation"
    TEMPLATE_MATCHING = "template_matching"
    QUBIT_MAPPING = "qubit_mapping"

class IQuantumCircuit(Protocol[T]):
    """Protocol for quantum circuits composing quantum operations."""
    
    @property
    def depth(self) -> int:
        """Circuit depth."""
        ...
    
    @property
    def width(self) -> int:
        """Number of qubits."""
        ...
    
    @property
    def hilbert_space(self) -> HilbertSpace[T]:
        """The Hilbert space of the circuit."""
        ...
    
    def add_gate(self,
                name: str,
                params: Optional[Sequence[float]] = None,
                target_qubits: Sequence[int] = (),
                control_qubits: Optional[Sequence[int]] = None) -> None:
        """Add a gate to the circuit."""
        ...
    
    def add_measurement(self,
                       qubits: Sequence[int],
                       basis: Optional[T] = None,
                       classical_registers: Optional[Sequence[int]] = None) -> None:
        """Add measurement operations."""
        ...
    
    def compose(self,
               other: IQuantumCircuit[T],
               qubits: Optional[Sequence[int]] = None) -> IQuantumCircuit[T]:
        """Compose with another circuit."""
        ...
    
    def optimize(self,
                methods: Sequence[CircuitOptimization] = (),
                **kwargs) -> IQuantumCircuit[T]:
        """Optimize the circuit."""
        ...
    
    def to_matrix(self) -> T:
        """Convert circuit to matrix representation."""
        ...
    
    def simulate(self,
                initial_state: Optional[IQuantumState[T]] = None,
                shots: int = 1000) -> Dict[str, Any]:
        """Simulate the circuit."""
        ...
    
    def get_metrics(self) -> CircuitMetrics:
        """Get circuit metrics."""
        ...

    # Field Theory Integration
    def add_field_operation(self,
                          operator: Callable[[T], T],
                          region: Optional[Sequence[int]] = None) -> None:
        """Add field theory operation to circuit."""
        ...
    
    def add_propagator(self,
                      hamiltonian_density: Callable[[T, T], T],
                      time: float,
                      qubits: Sequence[int]) -> None:
        """Add field propagator operation."""
        ...
    
    def add_correlation_measurement(self,
                                  qubits1: Sequence[int],
                                  qubits2: Sequence[int]) -> None:
        """Add correlation function measurement."""
        ...

    # Pattern Integration
    def to_attention_circuit(self) -> QuantumGeometricAttention[T]:
        """Convert to attention-based circuit representation."""
        ...
    
    def add_geometric_operation(self,
                              metric: T,
                              connection: T,
                              qubits: Sequence[int]) -> None:
        """Add geometric operation using metric and connection."""
        ...
    
    def add_pattern_evolution(self,
                            flow: GeometricFlow[T],
                            time: float,
                            qubits: Sequence[int]) -> None:
        """Add pattern evolution operation."""
        ...

    # Advanced Operations
    def add_error_correction(self,
                           code_type: str,
                           logical_qubits: Sequence[int],
                           physical_qubits: Sequence[int]) -> None:
        """Add quantum error correction encoding."""
        ...
    
    def add_stabilizer_measurement(self,
                                 stabilizer_generators: Sequence[T]) -> None:
        """Add stabilizer measurements."""
        ...
    
    def add_tomography_sequence(self,
                              bases: Sequence[MeasurementBasis[T]],
                              qubits: Sequence[int]) -> None:
        """Add state tomography measurement sequence."""
        ...
    
    def to_parametric_circuit(self) -> 'ParametricCircuit[T]':
        """Convert to parametric circuit form for optimization."""
        ...
    
    def compute_resource_requirements(self) -> Dict[str, int]:
        """Compute required quantum resources."""
        ...

@dataclass
class MeasurementBasis(Generic[T]):
    """Specification of measurement basis."""
    operators: Sequence[T]
    labels: Sequence[str]
    transformation: Optional[T] = None

@dataclass
class MeasurementProtocol(Generic[T]):
    """Specification of measurement protocol."""
    basis: MeasurementBasis[T]
    target_qubits: Sequence[int]
    conditional_operations: Optional[Dict[str, Callable[..., Any]]] = None

class IQuantumMeasurement(Protocol[T]):
    """Protocol for quantum measurements."""
    
    def setup_basis(self, basis: MeasurementBasis[T]) -> None:
        """Setup measurement basis."""
        ...
    
    def project(self,
               state: IQuantumState[T],
               protocol: MeasurementProtocol[T]) -> MeasurementResult[T]:
        """Perform projective measurement."""
        ...
    
    def weak_measure(self,
                    state: IQuantumState[T],
                    protocol: MeasurementProtocol[T],
                    strength: float) -> MeasurementResult[T]:
        """Perform weak measurement."""
        ...
    
    def tomography(self,
                  state: IQuantumState[T],
                  bases: Sequence[MeasurementBasis[T]] = (),
                  shots: int = 1000) -> Dict[str, Any]:
        """Perform quantum state tomography."""
        ...

    # Field Theory Measurements
    def measure_field_observable(self,
                               state: IQuantumState[T],
                               protocol: FieldMeasurementProtocol[T]) -> MeasurementResult[T]:
        """Measure field theory observable."""
        ...
    
    def measure_correlation_function(self,
                                   state: IQuantumState[T],
                                   operator1: Callable[[T], T],
                                   operator2: Callable[[T], T],
                                   separation: float) -> T:
        """Measure correlation function between two field operators."""
        ...
    
    def measure_field_momentum(self,
                             state: IQuantumState[T],
                             region: Optional[Sequence[int]] = None) -> T:
        """Measure field momentum in specified region."""
        ...

    # Pattern-Specific Measurements
    def measure_pattern_observable(self,
                                 state: IQuantumState[T],
                                 protocol: PatternMeasurementProtocol[T]) -> MeasurementResult[T]:
        """Measure pattern-specific observable."""
        ...
    
    def measure_geometric_phase(self,
                              state: IQuantumState[T],
                              path: T,
                              connection: Optional[T] = None) -> float:
        """Measure geometric phase along path."""
        ...
    
    def measure_pattern_stability(self,
                                state: IQuantumState[T],
                                metric: T,
                                threshold: float) -> Dict[str, float]:
        """Measure pattern stability metrics."""
        ...

    # Advanced Measurements
    def measure_entanglement_witness(self,
                                   state: IQuantumState[T],
                                   partition: Optional[Tuple[Sequence[int], Sequence[int]]] = None) -> float:
        """Measure entanglement witness."""
        ...
    
    def measure_purity(self,
                      state: IQuantumState[T],
                      subsystem: Optional[Sequence[int]] = None) -> float:
        """Measure state or subsystem purity."""
        ...
    
    def measure_coherence(self,
                         state: IQuantumState[T],
                         basis: MeasurementBasis[T]) -> float:
        """Measure quantum coherence in given basis."""
        ...
    
    def measure_fidelity_to_target(self,
                                  state: IQuantumState[T],
                                  target: IQuantumState[T]) -> float:
        """Measure fidelity to target state."""
        ...

@dataclass
class ChannelProperties:
    """Properties of quantum channels."""
    is_unital: bool
    is_cptp: bool
    kraus_rank: int
    choi_rank: int
    channel_fidelity: float
    field_preserving: bool = False
    pattern_preserving: bool = False

class IQuantumChannel(Protocol[T]):
    """Protocol for quantum channels."""
    
    @property
    def dimension(self) -> Tuple[int, int]:
        """Input and output dimensions."""
        ...
    
    @property
    def hilbert_space(self) -> HilbertSpace[T]:
        """The Hilbert space of the channel."""
        ...
    
    def apply(self, state: IQuantumState[T]) -> IQuantumState[T]:
        """Apply channel to quantum state."""
        ...
    
    def compose(self, other: IQuantumChannel[T]) -> IQuantumChannel[T]:
        """Compose with another channel."""
        ...
    
    def tensor(self, other: IQuantumChannel[T]) -> IQuantumChannel[T]:
        """Tensor product with another channel."""
        ...
    
    def to_kraus(self) -> Sequence[T]:
        """Get Kraus operator representation."""
        ...
    
    def to_choi(self) -> T:
        """Get Choi matrix representation."""
        ...
    
    def adjoint(self) -> IQuantumChannel[T]:
        """Get adjoint channel."""
        ...
    
    def get_properties(self) -> ChannelProperties:
        """Get channel properties."""
        ...
    
    def optimize_implementation(self,
                              method: str = "kraus",
                              **kwargs) -> IQuantumChannel[T]:
        """Optimize channel implementation."""
        ...

    # Field Theory Integration
    def to_field_channel(self) -> FieldChannel[T]:
        """Convert to field theory channel representation."""
        ...
    
    def apply_to_field_configuration(self,
                                   field_state: T,
                                   region: Optional[Sequence[int]] = None) -> T:
        """Apply channel to field configuration."""
        ...
    
    def preserve_field_observables(self,
                                 observables: Sequence[Callable[[T], T]]) -> bool:
        """Check if channel preserves given field observables."""
        ...

    # Pattern Integration
    def to_pattern_channel(self) -> PatternChannel[T]:
        """Convert to pattern-compatible channel representation."""
        ...
    
    def apply_with_metric(self,
                         state: IQuantumState[T],
                         metric: T) -> IQuantumState[T]:
        """Apply channel using geometric metric."""
        ...
    
    def preserve_geometric_structure(self,
                                   metric: T,
                                   connection: Optional[T] = None) -> bool:
        """Check if channel preserves geometric structure."""
        ...

    # Advanced Operations
    def apply_selectively(self,
                         state: IQuantumState[T],
                         region: Sequence[int],
                         strength: float = 1.0) -> IQuantumState[T]:
        """Apply channel selectively to region with given strength."""
        ...
    
    def to_parametric_channel(self) -> ParametricChannel[T]:
        """Convert to parametric form for optimization."""
        ...
    
    def compute_channel_capacity(self,
                               input_ensemble: Optional[Sequence[Tuple[float, IQuantumState[T]]]] = None) -> float:
        """Compute quantum channel capacity."""
        ...
    
    def compute_noise_parameters(self) -> Dict[str, float]:
        """Compute channel noise characteristics."""
        ...

# Base Protocols
class HilbertSpace(Protocol[T]):
    """Protocol for Hilbert space operations."""
    @property
    def dimension(self) -> int: ...
    def inner_product(self, state1: T, state2: T) -> complex: ...
    def tensor_product(self, other: HilbertSpace[T]) -> HilbertSpace[T]: ...

class GeometricFlow(Protocol[T]):
    """Protocol for geometric flow operations."""
    def compute_flow(self, metric: T, time: float) -> T: ...
    def evolve_state(self, state: T, time: float) -> T: ...
    def compute_stability(self, state: T) -> Dict[str, float]: ...

# Add new base protocols
class QuantumOperation(Protocol[T]):
    """Protocol for general quantum operations."""
    @property
    def dimension(self) -> Tuple[int, int]: ...
    def apply(self, state: IQuantumState[T]) -> IQuantumState[T]: ...
    def adjoint(self) -> 'QuantumOperation[T]': ...
    def compose(self, other: 'QuantumOperation[T]') -> 'QuantumOperation[T]': ...
    def tensor(self, other: 'QuantumOperation[T]') -> 'QuantumOperation[T]': ...

class QuantumObservable(Protocol[T]):
    """Protocol for quantum observables."""
    @property
    def is_hermitian(self) -> bool: ...
    def expectation(self, state: IQuantumState[T]) -> float: ...
    def variance(self, state: IQuantumState[T]) -> float: ...
    def eigensystem(self) -> Tuple[T, T]: ...
    def commutator(self, other: 'QuantumObservable[T]') -> 'QuantumObservable[T]': ...

class QuantumSymmetry(Protocol[T]):
    """Protocol for quantum symmetry operations."""
    def action(self, state: IQuantumState[T]) -> IQuantumState[T]: ...
    def infinitesimal_generator(self) -> QuantumObservable[T]: ...
    def is_unitary(self) -> bool: ...
    def compose(self, other: 'QuantumSymmetry[T]') -> 'QuantumSymmetry[T]': ...
    def inverse(self) -> 'QuantumSymmetry[T]': ...

class PatternState(Protocol[T]):
    """Protocol for pattern-based quantum states."""
    def to_quantum_state(self) -> IQuantumState[T]: ...
    def apply_pattern(self, pattern: T) -> PatternState[T]: ...
    def compute_stability(self) -> Dict[str, float]: ...

class FieldRegister(Protocol[T]):
    """Protocol for field theory registers."""
    def to_quantum_register(self) -> IQuantumRegister[T]: ...
    def apply_field(self, field: T) -> None: ...
    def measure_field(self, operator: Callable[[T], T]) -> T: ...

class PatternRegister(Protocol[T]):
    """Protocol for pattern-based registers."""
    def to_quantum_register(self) -> IQuantumRegister[T]: ...
    def apply_pattern(self, pattern: T) -> None: ...
    def measure_pattern(self, metric: T) -> Dict[str, float]: ...

class QuantumGeometricAttention(Protocol[T]):
    """Protocol for quantum geometric attention circuits."""
    def to_quantum_circuit(self) -> IQuantumCircuit[T]: ...
    def apply_attention(self, state: IQuantumState[T]) -> IQuantumState[T]: ...
    def compute_attention_pattern(self) -> T: ...

class ParametricCircuit(Protocol[T]):
    """Protocol for parametric quantum circuits."""
    def to_quantum_circuit(self) -> IQuantumCircuit[T]: ...
    def optimize_parameters(self, cost_function: Callable[[T], float]) -> None: ...
    def get_optimal_parameters(self) -> Dict[str, float]: ...

class FieldChannel(Protocol[T]):
    """Protocol for field theory channels."""
    def to_quantum_channel(self) -> IQuantumChannel[T]: ...
    def apply_to_field(self, field: T) -> T: ...
    def compute_field_properties(self) -> Dict[str, Any]: ...

class PatternChannel(Protocol[T]):
    """Protocol for pattern-based channels."""
    def to_quantum_channel(self) -> IQuantumChannel[T]: ...
    def apply_to_pattern(self, pattern: T) -> T: ...
    def compute_pattern_properties(self) -> Dict[str, Any]: ...

@dataclass
class FieldMeasurementProtocol(Generic[T]):
    """Protocol for field theory measurements."""
    operator: Callable[[T], T]
    region: Optional[Sequence[int]]
    boundary_conditions: Optional[Dict[str, T]] = None

@dataclass
class PatternMeasurementProtocol(Generic[T]):
    """Protocol for pattern-specific measurements."""
    metric: T
    connection: Optional[T]
    stability_parameters: Optional[Dict[str, float]] = None

class ParametricChannel(Protocol[T]):
    """Protocol for parametric quantum channels."""
    def to_quantum_channel(self) -> IQuantumChannel[T]: ...
    def optimize_parameters(self, cost_function: Callable[[T], float]) -> None: ...
    def get_optimal_parameters(self) -> Dict[str, float]: ...
    def apply_to_state(self, state: IQuantumState[T], parameters: Dict[str, float]) -> IQuantumState[T]: ...
    def compute_gradient(self, state: IQuantumState[T], target: IQuantumState[T]) -> Dict[str, float]: ...
 