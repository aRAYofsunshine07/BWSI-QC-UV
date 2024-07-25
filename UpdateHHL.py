import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import QFT
from qiskit.quantum_info import Statevector, Operator

# Custom import and other werid imports removed 

# Define matrix (A) and vector (b)
A = (1/np.sqrt(2)) * np.array([[1, -1], [1, 1]])
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

def PhaseEstimate(b: QuantumRegister, clock: QuantumRegister, unitary: List[List[complex]]) -> QuantumCircuit:
    circuit = QuantumCircuit(b, clock)
    circuit.h(clock)
    #Turns U gate matrix into controlled U gate matrix where MSB is control
    #This implementation is really messy but it works at least
    checkedUnitary = []
    for _ in range(len(unitary) * 2):
        row = []
        for _ in range(len(unitary) * 2):
            row.append(complex(0, 0))
        checkedUnitary.append(row)
    for i in range(len(unitary)):
        for j in range(len(unitary)):
            if(i == j):
                checkedUnitary[i][j] = complex(1, 0)
            checkedUnitary[i + len(unitary)][j + len(unitary)] = unitary[i][j]
    #Creates gate from matrix
    unitaryGate = Operator(checkedUnitary)
    #Runs controlled U gate the correct amount of times for each clock qubit
    for i in range(clock.size):
        #array is the input for the controlled U gate
        array = []
        for j in range(b.size):
            array.append(j)
        array.append(i + b.size)
        for _ in range(int(2.0 ** float(i))):
            circuit.unitary(unitaryGate, qubits = array)
    #Runs QFT
    qft = QFT(inverse = True, num_qubits = clock.size).to_gate()
    circuit.append(qft, qargs = [i + b.size for i in range(clock.size)])
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

# phase Estimation, error here 
phase_estimate(b_qubits, clock_qubits, A)

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
