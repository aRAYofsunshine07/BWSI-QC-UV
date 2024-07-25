import numpy as np

from typing import List
from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.quantum_info.operators import Operator
from qiskit.circuit.library import QFT

class PhaseEstimation:
    def PhaseEstimate(b: QuantumRegister, clock: QuantumRegister, unitary: List[List[complex]]) -> QuantumCircuit:
        circuit = QuantumCircuit(b, clock)
        circuit.h(clock)
        #Turns U gate matrix into controlled U gate matrix where MSB is control
        #This implementation is super messy but it doesn't throw errors at least
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
            for k in range(int(2.0 ** float(i))):
                circuit.unitary(unitaryGate, qubits = array)
        #Runs QFT
        qft = QFT(inverse = True, num_qubits = clock.size).to_gate()
        circuit.append(qft, qargs = [i + b.size for i in range(clock.size)])
        return circuit
