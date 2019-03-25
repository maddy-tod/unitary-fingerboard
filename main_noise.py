# -*- coding: utf-8 -*-

# Copyright 2018 IBM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""
Based on code by James Wootton
https://github.com/Qiskit/qiskit-tutorials/blob/master/community/games/random_terrain_generation.ipynb

"""

import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
from qiskit import *

class RandomTerrain():
    def __init__(self):
        self.num_qubits = 9
        self.shots = 4 ** self.num_qubits
        self.Lx = int(2 ** np.ceil(self.num_qubits / 2))
        self.Ly = int(2 ** np.floor(self.num_qubits / 2))

        self.make_strings()

    def make_strings(self):

        strings = {}
        for y in range(self.Ly):
            for x in range(self.Lx):
                strings[(x, y)] = ''

        for (x, y) in strings:
            for j in range(self.num_qubits):
                if (j % 2) == 0:
                    xx = np.floor(x / 2 ** (j / 2))
                    strings[(x, y)] = str(int((xx + np.floor(xx / 2)) % 2)) + strings[(x, y)]
                else:
                    yy = np.floor(y / 2 ** ((j - 1) / 2))
                    strings[(x, y)] = str(int((yy + np.floor(yy / 2)) % 2)) + strings[(x, y)]

        # shuffle to add a bit more randomness
        order = [j for j in range(self.num_qubits)]
        random.shuffle(order)

        for (x, y) in strings:
            new_string = ''
            for j in order:
                new_string = strings[(x, y)][j] + new_string
            strings[(x, y)] = new_string

        # make 0000 the centre
        center = '0' * self.num_qubits
        current_center = strings[(int(np.floor(self.Lx / 2)), int(np.floor(self.Ly / 2)))]
        diff = ''
        for j in range(self.num_qubits):
            diff += '0' * (current_center[j] == center[j]) + '1' * (current_center[j] != center[j])
        for (x, y) in strings:
            newstring = ''
            for j in range(self.num_qubits):
                newstring += strings[(x, y)][j] * (diff[j] == '0') + (
                        '0' * (strings[(x, y)][j] == '1') + '1' * (strings[(x, y)][j] == '0')) * (diff[j] == '1')
            strings[(x, y)] = newstring

        # make string to pos dict
        pos = {}
        for y in range(self.Ly):
            for x in range(self.Lx):
                pos[strings[(x, y)]] = (x, y)

        self.strings = strings
        self.pos = pos

    def plot_terrain(self, probs, filename=None,log=True, normalize=True):
        Z = {}
        for node in probs:
            if log:
                Z[node] = np.log(probs[node])
            else:
                Z[node] = probs[node]

        minZ = min(Z.values())
        maxZ = max(Z.values())
        colors = {}
        for node in Z:
            if normalize:
                z = (Z[node] - minZ) / (maxZ - minZ)
            else:
                z = Z[node]
            colors[node] = (z, z, z, 1)

        fig = plt.figure()
        ax = fig.add_subplot(111)

        for node in self.pos:
            rect = matplotlib.patches.Rectangle(self.pos[node], 1, 1, color=colors[node])
            ax.add_patch(rect)
        plt.xlim([0, self.Lx])
        plt.ylim([0, self.Ly])
        plt.axis('off')
        if filename:
            plt.savefig(filename, dpi=1000)
        plt.show()

    def get_probs(self, job):
        counts = job.result().get_counts()
        probs = {}
        for string in self.pos:
            try:
                probs[string] = counts[string]/self.shots
            except:
                probs[string] = 1/self.shots
        return probs

    def _execute_circuit(self, circuit):

        backend = Aer.get_backend('qasm_simulator')

        job = execute(circuit, backend, shots=self.shots)

        probs = self.get_probs(job)
        self.plot_terrain(probs)

    def basic_circuit(self, start_state=None):

        q = QuantumRegister(self.num_qubits)
        c = ClassicalRegister(self.num_qubits)
        qc = QuantumCircuit(q, c)

        if start_state:
            qc.initialize(start_state, q)

        qc.measure(q, c)

        self._execute_circuit(qc)

    def noisy_circuit(self, start_state=None):
        from qiskit.providers.aer import noise
        IBMQ.load_accounts()

        backend_to_simulate = IBMQ.get_backend('ibmq_16_melbourne')
        noise_model = noise.device.basic_device_noise_model(backend_to_simulate.properties())
        q = QuantumRegister(self.num_qubits)
        c = ClassicalRegister(self.num_qubits)
        qc = QuantumCircuit(q, c)

        if start_state:
            qc.initialize(start_state, q)

        qc.measure(q, c)

        backend = Aer.get_backend('qasm_simulator')
        print("starting job")
        job = execute(qc, backend, shots=self.shots, noise_model=noise_model, basis_gates=noise_model.basis_gates)

        probs = self.get_probs(job)

        self.plot_terrain(probs)

    def ghz_circuit(self, start_state=None):
        q = QuantumRegister(self.num_qubits)
        c = ClassicalRegister(self.num_qubits)
        qc = QuantumCircuit(q, c)

        if start_state:
            qc.initialize(start_state, q)

        qc.h(q[0])
        for j in range(self.num_qubits - 1):
            qc.cx(q[j], q[j + 1])
        qc.measure(q, c)

        self._execute_circuit(qc)

    def rotation_y_circuit(self, fraction, start_state=None):
        q = QuantumRegister(self.num_qubits)
        c = ClassicalRegister(self.num_qubits)
        qc = QuantumCircuit(q, c)

        if start_state:
            qc.initialize(start_state, q)

        qc.h(q[0])
        for j in range(self.num_qubits - 1):
            qc.cx(q[j], q[j + 1])
        qc.ry(np.pi * fraction, q)
        qc.measure(q, c)

        self._execute_circuit(qc)

    def random_start_state(self):
        rho = 0.1
        N = int(rho * 2 ** self.num_qubits)
        state = [0] * (2 ** self.num_qubits)
        for j in range(N):
            state[int(random.choice(list(self.pos.keys())), 2)] = 1
        Z = sum(np.absolute(state) ** 2)
        return [amp / np.sqrt(Z) for amp in state]


if __name__ == '__main__':
    rt = RandomTerrain()
    s = rt.random_start_state()
    rt.basic_circuit(start_state=s)
