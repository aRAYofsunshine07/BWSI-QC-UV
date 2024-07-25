import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer
from qiskit.circuit.library import QFT
from qiskit.aqua.components.initial_states import Custom
from qiskit.quantum_info import Operator
from typing import List

# Define matrix (A) and vector (b)
A = np.array([[1, -1], [1, 1]])
b = np.array([1, 0])

# Normalize the vector b
norm_b = np.linalg.norm(b)
b_normalized = b / norm_b

# Define the number of qubits
num_qubits = int(np.log2(A.shape[0])) + 1
b_qubits = QuantumRegister(num_qubits, name='b')
clock_qubits = QuantumRegister(num_qubits, name='clock')

# Initial state circuit for |b>
initial_state = Custom(num_qubits=num_qubits, state_vector=b_normalized)

# Initialize circuit
qc = QuantumCircuit(b_qubits, clock_qubits)

# Prepare initial state to vector b
initial_state_circuit = initial_state.construct_circuit('circuit')
qc.append(initial_state_circuit, b_qubits)


class PhaseEstimation:
    @staticmethod
    def phase_estimate(b: QuantumRegister, clock: QuantumRegister, unitary: List[List[float]]) -> QuantumCircuit:
        circuit = QuantumCircuit(b, clock)

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


def rotate_ancilla(qc: QuantumCircuit, clock_qubits: QuantumRegister) -> List:
    # adds the ancill qbit and a CReg for measuring
    ancilla_qubit = QuantumRegister(1, name='ancilla')
    classical_reg = ClassicalRegister(1, name='measure')
    qc.add_register(ancilla_qubit)
    qc.add_register(classical_reg)

    # Crotating the ancilla qubit per clock-qubit
    for i in range(clock_qubits.size):
        angle = 2 * np.arcsin(i / (clock_qubits.size - 1))
        qc.cry(angle, clock_qubits[i], ancilla_qubit[0])

    qc.measure(ancilla_qubit[0], classical_reg[0])

    simulator = Aer.get_backend('aer_simulator')
    result = simulator.run(qc, shots=1, memory=True).result()
    s = result.get_memory()[0]

    return [qc, int(s)]


def inverse_qpe(qc: QuantumCircuit, clock_qubits: QuantumRegister) -> QuantumCircuit:
    qc.append(QFT(len(clock_qubits), do_swaps=False).inverse(), clock_qubits)
    return qc

# COMPLETE CIRCUIT!!
phase_estimation_circuit = PhaseEstimation.phase_estimate(b_qubits, clock_qubits, A)
qc.append(phase_estimation_circuit, b_qubits[:] + clock_qubits[:])
inverse_qpe(qc, clock_qubits)
result = rotate_ancilla(qc, clock_qubits)

# Display the final quantum circuit
print(result[0])
