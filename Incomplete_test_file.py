import unittest
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import QFT
from qiskit.quantum_info import Statevector, Operator
from typing import List
from UpdateHHL.py import prepare_initial_state, PhaseEstimate, controlled_rotation, inverse_qpe, main


class HHLTests(unittest.TestCase):
    def test_phase_estimate(self):
        A = (1 / np.sqrt(2)) * np.array([[1, -1], [1, 1]])
        b_qubits = QuantumRegister(2, name='b')
        clock_qubits = QuantumRegister(2, name='clock')
        phase_estimation_circuit = PhaseEstimate(b_qubits, clock_qubits, A)
        
        self.assertIsInstance(phase_estimation_circuit, QuantumCircuit)

    def test_controlled_rotation(self):
        clock_qubits = QuantumRegister(2, name='clock')
        ancilla_qubit = QuantumRegister(1, name='ancilla')
        qc = QuantumCircuit(clock_qubits, ancilla_qubit)
        controlled_rotation(qc, clock_qubits, ancilla_qubit)
        
        self.assertIsInstance(qc, QuantumCircuit)

    def test_inverse_qpe(self):
        clock_qubits = QuantumRegister(2, name='clock')
        qc = QuantumCircuit(clock_qubits)
        inverse_qpe(qc, clock_qubits)
        
        self.assertIsInstance(qc, QuantumCircuit)

