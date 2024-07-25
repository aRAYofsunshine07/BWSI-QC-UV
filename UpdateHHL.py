import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import QFT
from qiskit.quantum_info import Statevector, Operator

# Custom import and other werid imports removed 

# Define matrix (A) and vector (b)
A = np.array([[1, -1], [1, 1]])
b = np.array([1, 0])

# Normalize the vector b
norm_b = np.linalg.norm(b)
b_normalized = b / norm_b

# Number of qubits
n = A.shape[0]
num_qubits = int(np.log2(n)) + 1

# Create registers
b_qubits = QuantumRegister(n, name='b')
clock_qubits = QuantumRegister(num_qubits, name='clock')
ancilla_qubit = QuantumRegister(1, name='ancilla')
classical_reg = ClassicalRegister(n, name='measure')

# Initialize circuit
qc = QuantumCircuit(b_qubits, clock_qubits, ancilla_qubit, classical_reg)

# prepare initial state to vector b
def prepare_initial_state(circuit, qubits, vector):
    norm = np.linalg.norm(vector)
    theta = 2 * np.arccos(vector[0] / norm)
    circuit.ry(theta, qubits[0])
    if len(vector) > 1 and vector[1] != 0:
        phi = np.angle(vector[1])
        circuit.rz(phi, qubits[0])

prepare_initial_state(qc, b_qubits, b_normalized)

def phase_estimate(b: QuantumRegister, clock: QuantumRegister, unitary: np.ndarray) -> QuantumCircuit:
    circuit = QuantumCircuit(b, clock)

    # Hadamard transform on clock qubits
    for i in range(clock.size):
        circuit.h(clock[i])

    # Turn U gate matrix into controlled U gate matrix where MSB is control
    dim = len(unitary)
    checked_unitary = np.zeros((2 * dim, 2 * dim), dtype=complex)
    for i in range(dim):
        checked_unitary[i, i] = 1
        for j in range(dim):
            checked_unitary[i + dim, j + dim] = unitary[i, j]

    # Create gate from matrix
    unitary_gate = Operator(checked_unitary)

    # Run controlled U gate the correct amount of times for each clock qubit
    for i in range(clock.size):
        control_qubit = clock[i]
        target_qubits = list(range(b.size))
        for _ in range(2 ** i):
            circuit.append(unitary_gate.control(), [control_qubit] + target_qubits)

    # Run QFT on the clock register
    qft = QFT(num_qubits=clock.size).to_gate()
    circuit.append(qft, clock)

    return circuit

def controlled_rotation(qc: QuantumCircuit, clock_qubits: QuantumRegister, ancilla_qubit: QuantumRegister) -> QuantumCircuit:
    # Crotating the ancilla qubit per clock-qubit
    for i in range(clock_qubits.size):
        angle = 2 * np.arcsin(i / (clock_qubits.size - 1))
        qc.cry(angle, clock_qubits[i], ancilla_qubit[0])
    return qc


# Updated 
def inverse_qpe(qc, clock_qubits):
    qft_inv = QFT(len(clock_qubits)).inverse()
    qc.append(qft_inv, clock_qubits)
    qc.h(clock_qubits)
    qc.barrier()

# phase Estimation
phase_estimate(qc, b_qubits, clock_qubits)

# controlled Rotation
controlled_rotation(qc, clock_qubits, ancilla_qubit)


# inverse QPE
inverse_qpe(qc, clock_qubits)

# Measure
qc.measure(b_qubits, classical_reg)


# Simulataion 
state = Statevector.from_label('0' * (n + num_qubits + 1))
state = state.evolve(qc)
probabilities = state.probabilities_dict()

print("Statevector probabilities:", probabilities)
