import numpy as np

from typing import List
from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.quantum_info.operators import Operator
from qiskit.circuit.library import QFT

class PhaseEstimation:
    def PhaseEstimate(b: QuantumRegister, clock: QuantumRegister, unitary: List[List[float]]) -> QuantumCircuit:
        circuit = QuantumCircuit(b, clock)
        circuit.h(clock)
        #Turns U gate matrix into controlled U gate matrix where MSB is control
        checkedUnitary = np.zeros([len(unitary) * 2, len(unitary) * 2])
        for i in range(len(unitary)):
            for j in range(len(unitary)):
                if(i == j):
                    checkedUnitary[i][j] = 1
                checkedUnitary[i + 4][j + 4] = unitary[i][j]
        #Creates gate from matrix
        unitaryGate = Operator(checkedUnitary)
        #Runs controlled U gate the correct amount of times for each clock qubit
        for i in range(clock.size):
            #array is the input for the controlled U gate
            array = []
            for j in range(b.size):
                array.append(j)
            array.append(i + b.size)
            for k in range(2.0 ** float(i)):
                circuit.unitary(unitaryGate, array)
        #Runs QFT
        qft = QFT(num_qubits = clock.size).to_gate
        circuit.append(qft, (np.arange(clock.size) + b.size))
        return circuit
