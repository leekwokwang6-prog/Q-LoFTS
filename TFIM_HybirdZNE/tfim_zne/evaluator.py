"""Evaluate parameterized TFIM circuits with the configured gate noise."""
import tensorcircuit as tc
import numpy as np
import tensorflow as tf
tc.set_dtype('complex64')
tc.set_backend('tensorflow')

class VqeEvaluator:

    def __init__(self, n_cir_parallel, n_qubit, hamiltonian, noise_param=None):
        self.K = tc.set_backend('tensorflow')
        self.n_qubit = n_qubit
        self.n_cir_parallel = n_cir_parallel
        self.hamiltonian_ = hamiltonian
        self.lattice = tc.templates.graphs.Line1D(self.n_qubit, pbc=self.hamiltonian_['pbc'])
        self.h = tc.quantum.heisenberg_hamiltonian(self.lattice, hzz=self.hamiltonian_['hzz'], hxx=self.hamiltonian_['hxx'], hyy=self.hamiltonian_['hyy'], hx=self.hamiltonian_['hx'], hy=self.hamiltonian_['hy'], hz=self.hamiltonian_['hz'], sparse=self.hamiltonian_['sparse'])
        if noise_param is None:
            self.noise = False
        else:
            self.noise = True
            self.two_qubit_channel_depolarizing_p = noise_param['two_qubit_channel_depolarizing_p']
            self.single_qubit_channel_depolarizing_p = noise_param['single_qubit_channel_depolarizing_p']
            self.bit_flip_p = noise_param['bit_flip_p']
            self.single_qubit_dep_channel = tc.channels.generaldepolarizingchannel(self.single_qubit_channel_depolarizing_p / 3, 1)
            self.two_qubit_dep_channel = tc.channels.generaldepolarizingchannel(self.two_qubit_channel_depolarizing_p / 3, 1)
            self.two_qubit_channel_T1 = noise_param['two_qubit_channel_T1']
            self.two_qubit_channel_T2 = noise_param['two_qubit_channel_T2']
            self.two_qubit_channel_gateTime = noise_param['two_qubit_channel_gateTime']
            self.two_qubit_channel_esp = noise_param['two_qubit_channel_esp']
            self.single_qubit_channel_T1 = noise_param['single_qubit_channel_T1']
            self.single_qubit_channel_T2 = noise_param['single_qubit_channel_T2']
            self.single_qubit_channel_gateTime = noise_param['single_qubit_channel_gateTime']
            self.single_qubit_channel_esp = noise_param['single_qubit_channel_esp']
            self.single_qubit_tm_channel = tc.channels.thermalrelaxationchannel(self.single_qubit_channel_T1, self.single_qubit_channel_T2, self.single_qubit_channel_gateTime, 'AUTO', self.single_qubit_channel_esp)
            self.two_qubit_tm_channel = tc.channels.thermalrelaxationchannel(self.two_qubit_channel_T1, self.two_qubit_channel_T2, self.two_qubit_channel_gateTime, 'AUTO', self.two_qubit_channel_esp)

    def compute_energy(self, structure):
        if self.noise:
            K0 = np.array([[1, 0], [0, 1]]) * np.sqrt(1 - self.bit_flip_p)
            K1 = np.array([[0, 1], [1, 0]]) * np.sqrt(self.bit_flip_p)
            c = tc.DMCircuit(self.n_qubit)
            a = -0.01
            b = 0.02
            for gate in structure:
                if gate[0] == 'h':
                    c.h(gate[1])
                    c.general_kraus(self.single_qubit_dep_channel, gate[1])
                    c.general_kraus(self.single_qubit_tm_channel, gate[1])
                elif gate[0] == 'cont':
                    c.cx(gate[1], gate[2])
                    c.general_kraus(self.two_qubit_dep_channel, gate[1])
                    c.general_kraus(self.two_qubit_dep_channel, gate[2])
                    c.general_kraus(self.two_qubit_tm_channel, gate[1])
                    c.general_kraus(self.two_qubit_tm_channel, gate[2])
                elif gate[0] == 'cz':
                    c.cz(gate[1], gate[2])
                    c.general_kraus(self.two_qubit_dep_channel, gate[1])
                    c.general_kraus(self.two_qubit_dep_channel, gate[2])
                    c.general_kraus(self.two_qubit_tm_channel, gate[1])
                    c.general_kraus(self.two_qubit_tm_channel, gate[2])
                elif gate[0] == 'ry':
                    delta = np.random.uniform(a * np.pi, b * np.pi)
                    c.ry(gate[1], theta=gate[2] + delta)
                    c.general_kraus(self.single_qubit_dep_channel, gate[1])
                    c.general_kraus(self.single_qubit_tm_channel, gate[1])
                elif gate[0] == 'rz':
                    delta = np.random.uniform(a * np.pi, b * np.pi)
                    c.rz(gate[1], theta=gate[2] + delta)
                    c.general_kraus(self.single_qubit_dep_channel, gate[1])
                    c.general_kraus(self.single_qubit_tm_channel, gate[1])
                elif gate[0] == 'rx':
                    delta = np.random.uniform(a * np.pi, b * np.pi)
                    c.rx(gate[1], theta=gate[2] + delta)
                    c.general_kraus(self.single_qubit_dep_channel, gate[1])
                    c.general_kraus(self.single_qubit_tm_channel, gate[1])
            for q in range(self.n_qubit):
                c.general_kraus([K0, K1], q)
            'Calculate energy'
            st = c.state()
            x = tf.matmul(st, self.h)
            e = tf.linalg.trace(x)
            e = self.K.real(e)
        else:
            c = tc.Circuit(self.n_qubit)
            for gate in structure:
                if gate[0] == 'h':
                    c.h(gate[1])
                elif gate[0] == 'cont':
                    c.cx(gate[1], gate[2])
                elif gate[0] == 'cz':
                    c.cz(gate[1], gate[2])
                elif gate[0] == 'ry':
                    c.ry(gate[1], theta=gate[2])
                elif gate[0] == 'rz':
                    c.rz(gate[1], theta=gate[2])
                elif gate[0] == 'rx':
                    c.rx(gate[1], theta=gate[2])
            e = tc.templates.measurements.operator_expectation(c, self.h)
        return e

    def evaluate_single(self, sample):
        cir_list = sample[0]
        e = self.compute_energy(cir_list)
        return e
