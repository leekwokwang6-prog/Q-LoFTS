"""Hybrid TFIM energy objective and circuit-parameter optimization."""
import tensorcircuit as tc
import numpy as np
import tensorflow as tf
from .model import QLoFTSModel
import torch
import scipy.optimize as optimize
from numpy import random
tc.set_dtype('complex128')
tc.set_backend('tensorflow')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f'Random seed set to: {seed}')

class VqeTrainerNew:

    def __init__(self, n_cir_parallel, n_runs, max_iteration, n_qubit, hamiltonian, noise_param=None):
        self.K = tc.set_backend('tensorflow')
        self.n_qubit = n_qubit
        self.max_iteration = max_iteration
        self.n_cir_parallel = n_cir_parallel
        self.n_runs = n_runs
        self.hamiltonian_ = hamiltonian
        self.lattice = tc.templates.graphs.Line1D(self.n_qubit, pbc=self.hamiltonian_['pbc'])
        self.h = tc.quantum.heisenberg_hamiltonian(self.lattice, hzz=self.hamiltonian_['hzz'], hxx=self.hamiltonian_['hxx'], hyy=self.hamiltonian_['hyy'], hx=self.hamiltonian_['hx'], hy=self.hamiltonian_['hy'], hz=self.hamiltonian_['hz'], sparse=self.hamiltonian_['sparse'])
        self.give_up_rest = False
        self.solution = None
        if noise_param is None:
            raise ValueError('Hybrid QLoFTS requires a configured noise model.')
        self.noise = True
        self.two_qubit_channel_depolarizing_p = None
        self.single_qubit_channel_depolarizing_p = None
        self.bit_flip_p = None
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

    def compute_energy_trotter(self, param, structure, loop_circuit, num_layer, num_qubit):
        """
        :param param: Circuit Parameters
        :param structure: Circuit
        :return:
        """
        a = -0.01
        b = 0.02
        K0 = np.array([[1, 0], [0, 1]]) * np.sqrt(1 - self.bit_flip_p)
        K1 = np.array([[0, 1], [1, 0]]) * np.sqrt(self.bit_flip_p)
        c = tc.DMCircuit(self.n_qubit)
        temp_loop = 0
        for i in range(loop_circuit):
            param_idx = 0
            if temp_loop % 2 == 0:
                for q in range(num_qubit):
                    c.h(q)
                    c.general_kraus(self.single_qubit_dep_channel, q)
                    c.general_kraus(self.single_qubit_tm_channel, q)
                for _ in range(num_layer):
                    for q in range(num_qubit):
                        theta_rx = param[param_idx] + np.random.uniform(a * np.pi, b * np.pi)
                        param_idx += 1
                        c.rx(q, theta=theta_rx)
                        c.general_kraus(self.single_qubit_dep_channel, q)
                        c.general_kraus(self.single_qubit_tm_channel, q)
                    for q in range(0, num_qubit - 1, 2):
                        theta_rz = param[param_idx] + np.random.uniform(a * np.pi, b * np.pi)
                        param_idx += 1
                        c.cnot(q, q + 1)
                        c.general_kraus(self.two_qubit_dep_channel, q)
                        c.general_kraus(self.two_qubit_dep_channel, q + 1)
                        c.general_kraus(self.two_qubit_tm_channel, q)
                        c.general_kraus(self.two_qubit_tm_channel, q + 1)
                        c.rz(q + 1, theta=theta_rz)
                        c.general_kraus(self.single_qubit_dep_channel, q + 1)
                        c.general_kraus(self.single_qubit_tm_channel, q + 1)
                        c.cnot(q, q + 1)
                        c.general_kraus(self.two_qubit_dep_channel, q)
                        c.general_kraus(self.two_qubit_dep_channel, q + 1)
                        c.general_kraus(self.two_qubit_tm_channel, q)
                        c.general_kraus(self.two_qubit_tm_channel, q + 1)
                    for q in range(1, num_qubit - 1, 2):
                        theta_rz = param[param_idx] + np.random.uniform(a * np.pi, b * np.pi)
                        param_idx += 1
                        c.cnot(q, q + 1)
                        c.general_kraus(self.two_qubit_dep_channel, q)
                        c.general_kraus(self.two_qubit_dep_channel, q + 1)
                        c.general_kraus(self.two_qubit_tm_channel, q)
                        c.general_kraus(self.two_qubit_tm_channel, q + 1)
                        c.rz(q + 1, theta=theta_rz)
                        c.general_kraus(self.single_qubit_dep_channel, q + 1)
                        c.general_kraus(self.single_qubit_tm_channel, q + 1)
                        c.cnot(q, q + 1)
                        c.general_kraus(self.two_qubit_dep_channel, q)
                        c.general_kraus(self.two_qubit_dep_channel, q + 1)
                        c.general_kraus(self.two_qubit_tm_channel, q)
                        c.general_kraus(self.two_qubit_tm_channel, q + 1)
            else:
                reversed_params = param[::-1]
                rev_idx = 0
                for _ in range(num_layer):
                    for q in range(1, num_qubit - 1, 2):
                        c.cnot(q, q + 1)
                        c.general_kraus(self.two_qubit_dep_channel, q)
                        c.general_kraus(self.two_qubit_dep_channel, q + 1)
                        c.general_kraus(self.two_qubit_tm_channel, q)
                        c.general_kraus(self.two_qubit_tm_channel, q + 1)
                    for q in reversed(range(1, num_qubit - 1, 2)):
                        theta_rz = -1 * reversed_params[rev_idx] + np.random.uniform(a * np.pi, b * np.pi)
                        rev_idx += 1
                        c.rz(q + 1, theta=theta_rz)
                        c.general_kraus(self.single_qubit_dep_channel, q + 1)
                        c.general_kraus(self.single_qubit_tm_channel, q + 1)
                    for q in range(1, num_qubit - 1, 2):
                        c.cnot(q, q + 1)
                        c.general_kraus(self.two_qubit_dep_channel, q)
                        c.general_kraus(self.two_qubit_dep_channel, q + 1)
                        c.general_kraus(self.two_qubit_tm_channel, q)
                        c.general_kraus(self.two_qubit_tm_channel, q + 1)
                    for q in range(0, num_qubit - 1, 2):
                        c.cnot(q, q + 1)
                        c.general_kraus(self.two_qubit_dep_channel, q)
                        c.general_kraus(self.two_qubit_dep_channel, q + 1)
                        c.general_kraus(self.two_qubit_tm_channel, q)
                        c.general_kraus(self.two_qubit_tm_channel, q + 1)
                    for q in reversed(range(0, num_qubit - 1, 2)):
                        theta_rz = -1 * reversed_params[rev_idx] + np.random.uniform(a * np.pi, b * np.pi)
                        rev_idx += 1
                        c.rz(q + 1, theta=theta_rz)
                        c.general_kraus(self.single_qubit_dep_channel, q + 1)
                        c.general_kraus(self.single_qubit_tm_channel, q + 1)
                    for q in range(0, num_qubit - 1, 2):
                        c.cnot(q, q + 1)
                        c.general_kraus(self.two_qubit_dep_channel, q)
                        c.general_kraus(self.two_qubit_dep_channel, q + 1)
                        c.general_kraus(self.two_qubit_tm_channel, q)
                        c.general_kraus(self.two_qubit_tm_channel, q + 1)
                    for q in reversed(range(num_qubit)):
                        theta_rx = -1 * reversed_params[rev_idx] + np.random.uniform(a * np.pi, b * np.pi)
                        rev_idx += 1
                        c.rx(q, theta=theta_rx)
                        c.general_kraus(self.single_qubit_dep_channel, q)
                        c.general_kraus(self.single_qubit_tm_channel, q)
                for q in range(num_qubit):
                    c.h(q)
                    c.general_kraus(self.single_qubit_dep_channel, q)
                    c.general_kraus(self.single_qubit_tm_channel, q)
            temp_loop = temp_loop + 1
        for q in range(self.n_qubit):
            c.general_kraus([K0, K1], q)
        'Calculate energy'
        st = c.state()
        x = tf.matmul(st, self.h)
        e = tf.linalg.trace(x)
        e = self.K.real(e)
        return e

    def get_param_num(self, cir):
        param_num = 0
        for i in range(len(cir)):
            if cir[i][0] == 'rx':
                param_num += 1
            if cir[i][0] == 'ry':
                param_num += 1
            if cir[i][0] == 'rz':
                param_num += 1
        return param_num

    def train_circuit(self, circuit_and_seed, run_id):
        single_circuit = circuit_and_seed[0]
        seed = circuit_and_seed[1]
        INPUT_DIM = 19
        HIDDEN_UNITS = 128
        MAX_FREQ = 1
        DROPOUT_RATE = 0
        models_dict = {}
        for noise_level in [5, 7, 9]:
            model = QLoFTSModel(input_dim=INPUT_DIM, hidden_units=HIDDEN_UNITS, max_freq=MAX_FREQ, dropout_rate=DROPOUT_RATE)
            checkpoint_path = f'frozen/models/training_{noise_level}/run_{run_id}/best_model.pth'
            model.load_state_dict(torch.load(checkpoint_path))
            model.eval()
            models_dict[noise_level] = model
        np.random.seed(seed)
        tf.random.set_seed(seed)
        param_num = self.get_param_num(single_circuit[0])
        if param_num != 19:
            raise ValueError('Hybrid QLoFTS requires exactly 19 circuit parameters.')
        L = single_circuit[1]
        par = np.random.normal(loc=0, scale=1 / (8 * (L + 2)), size=param_num)
        param_initial = par.copy()
        energy_epoch = []
        raw_energy_epoch = []

        def current_energy(p):
            return self.compute_energy_trotter(p, single_circuit[0], 1, 1, self.n_qubit)
        f_scipy = tc.interfaces.scipy_interface(current_energy, shape=[param_num], jit=False, gradient=False)

        def cost_wrapper(p_numpy):
            z = []
            e_val = f_scipy(p_numpy)
            z.append(e_val)
            print(f'Current circuit energy: {e_val}')
            raw_energy_epoch.append(float(e_val))
            z.append(self.compute_energy_trotter(p_numpy, single_circuit[0], 3, 1, self.n_qubit))
            p_tensor = torch.tensor(p_numpy, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                for noise_level in [5, 7, 9]:
                    prediction = models_dict[noise_level](p_tensor)
                    z.append(prediction.item())
            z = np.asarray(z, dtype=np.float64)
            mitigated_e = np.polyfit([1, 3, 5, 7, 9], z, 2)[-1]
            print(f'Extrapolated energy: {mitigated_e}')
            energy_epoch.append(float(mitigated_e))
            return mitigated_e
        res = optimize.minimize(cost_wrapper, par, method='COBYLA', options={'maxiter': self.max_iteration, 'tol': 0.0001})
        final_e = res.fun
        final_param = res.x
        return (final_e, final_param, energy_epoch, raw_energy_epoch)
