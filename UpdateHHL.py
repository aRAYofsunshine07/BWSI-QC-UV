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

# Prepare initial state to vector b
qc.initialize(b_normalized.tolist(), b_qubits)

def phase_estimate(b: QuantumRegister, clock: QuantumRegister, unitary: np.ndarray) -> QuantumCircuit:
    circuit = QuantumCircuit(b, clock)

    #Hadamard transform on clock qbits
    for i in range(clock.size):
        circuit.h(clock[i])
        
    # Turn U gate matrix into controlled U gate matrix where MSB is control
    checked_unitary = np.zeros([len(unitary) * 2, len(unitary) * 2])
    for i in range(len(unitary)):
        for j in range(len(unitary)):
            if i == j:
                checked_unitary[i][j] = 1
            checked_unitary[i + len(unitary)][j + len(unitary)] = unitary[i][j]

    # Create gate from matrix
    unitary_gate = Operator(checked_unitary)

    # Run controlled U gate the correct amount of times for each clock qubit
    for i in range(clock.size):
        control_array = list(range(b.size)) + [i + b.size]
        for _ in range(2 ** i):
            circuit.unitary(unitary_gate, control_array)

    # Run QFT
    qft = QFT(num_qubits=clock.size).to_gate()
    circuit.append(qft, np.arange(clock.size) + b.size)
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
phase_estimate(qc, b_qubits, clock_qubits, A)

# controlled Rotation
controlled_rotation(qc, clock_qubits, ancilla_qubit)

qc.measure(ancilla, 0)
result = AerSimulator().run(qc, shots=1, memory=True).result()
s = result.get_memory()[0]\

#conditional ancilla control
if s == '0':
    # phase Estimation
    phase_estimate(qc, b_qubits, clock_qubits, A)

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
