"""Generate parameterized TFIM circuits at five folding factors."""
import numpy as np

def generate_hea_sample_theta(num_qubit, num_layer, loop_circuit):
    base_params = []
    for _ in range(num_layer):
        for q in range(num_qubit):
            base_params.append(np.random.uniform(-np.pi, np.pi))
        for q in range(0, num_qubit - 1, 2):
            base_params.append(np.random.uniform(-np.pi, np.pi))
        for q in range(1, num_qubit - 1, 2):
            base_params.append(np.random.uniform(-np.pi, np.pi))
    sample1 = circuit_expend(base_params, 1, num_qubit, num_layer)
    sample3 = circuit_expend(base_params, 3, num_qubit, num_layer)
    sample5 = circuit_expend(base_params, 5, num_qubit, num_layer)
    sample7 = circuit_expend(base_params, 7, num_qubit, num_layer)
    sample9 = circuit_expend(base_params, 9, num_qubit, num_layer)
    return (sample1, sample3, sample5, sample7, sample9, base_params)

def circuit_expend(base_params, loop_circuit, num_qubit, num_layer):
    circuit = []
    temp_loop = 0
    for i in range(loop_circuit):
        param_idx = 0
        if temp_loop % 2 == 0:
            for q in range(num_qubit):
                circuit.append(['h', q])
            for _ in range(num_layer):
                for q in range(num_qubit):
                    theta_rx = base_params[param_idx]
                    param_idx += 1
                    circuit.append(['rx', q, theta_rx])
                for q in range(0, num_qubit - 1, 2):
                    theta_rz = base_params[param_idx]
                    param_idx += 1
                    circuit.append(['cont', q, q + 1])
                    circuit.append(['rz', q + 1, theta_rz])
                    circuit.append(['cont', q, q + 1])
                for q in range(1, num_qubit - 1, 2):
                    theta_rz = base_params[param_idx]
                    param_idx += 1
                    circuit.append(['cont', q, q + 1])
                    circuit.append(['rz', q + 1, theta_rz])
                    circuit.append(['cont', q, q + 1])
        else:
            reversed_params = base_params[::-1]
            rev_idx = 0
            for _ in range(num_layer):
                for q in range(1, num_qubit - 1, 2):
                    circuit.append(['cont', q, q + 1])
                for q in reversed(range(1, num_qubit - 1, 2)):
                    theta_rz = -1 * reversed_params[rev_idx]
                    rev_idx += 1
                    circuit.append(['rz', q + 1, theta_rz])
                for q in range(1, num_qubit - 1, 2):
                    circuit.append(['cont', q, q + 1])
                for q in range(0, num_qubit - 1, 2):
                    circuit.append(['cont', q, q + 1])
                for q in reversed(range(0, num_qubit - 1, 2)):
                    theta_rz = -1 * reversed_params[rev_idx]
                    rev_idx += 1
                    circuit.append(['rz', q + 1, theta_rz])
                for q in range(0, num_qubit - 1, 2):
                    circuit.append(['cont', q, q + 1])
                for q in reversed(range(num_qubit)):
                    theta_rx = -1 * reversed_params[rev_idx]
                    rev_idx += 1
                    circuit.append(['rx', q, theta_rx])
            for q in range(num_qubit):
                circuit.append(['h', q])
        temp_loop = temp_loop + 1
    (unique_cir, depth, num_gate, num_2_qb_gate) = make_it_unique(circuit, num_qubit)
    return [unique_cir, depth, num_gate, num_2_qb_gate]

def make_it_unique(circuit, num_qubit):
    bitmap = []
    for _ in range(0, num_qubit):
        bitmap.append([])
    (num_1_qb_gate, num_2_qb_gate) = (0, 0)
    single_qubit_gates = {'ry', 'rz', 'rx', 'h', 'id'}
    for gate in circuit:
        gate_name = gate[0]
        is_single_qubit = gate_name in single_qubit_gates
        if is_single_qubit:
            qubit_idx = gate[1]
            bitmap[qubit_idx].append(gate)
            num_1_qb_gate += 1
        else:
            (q1, q2) = (gate[1], gate[2])
            if len(bitmap[q1]) >= len(bitmap[q2]):
                (qb_long, qb_short) = (q1, q2)
            else:
                (qb_long, qb_short) = (q2, q1)
            while len(bitmap[qb_short]) < len(bitmap[qb_long]):
                bitmap[qb_short].append(None)
            if q1 > q2:
                (qb_low, qb_high) = (q2, q1)
            else:
                (qb_low, qb_high) = (q1, q2)
            bitmap[qb_low].append(gate)
            bitmap[qb_high].append(None)
            num_2_qb_gate += 1
    num_bit = []
    for i in range(0, num_qubit):
        num_bit.append(len(bitmap[i]))
    depth = max(num_bit) if num_bit else 0
    uni_cir = []
    for i in range(depth):
        for j in range(num_qubit):
            if num_bit[j] > i:
                if bitmap[j][i] is not None:
                    uni_cir.append(bitmap[j][i])
    return (uni_cir.copy(), depth, num_1_qb_gate + num_2_qb_gate, num_2_qb_gate)
