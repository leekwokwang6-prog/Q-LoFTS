"""Generate the original unfurled HEA circuit without changing random draws."""

import random

import numpy as np


def make_it_unique(circuit, num_qubit):
    bitmap = [[] for _ in range(num_qubit)]
    num_1_qb_gate, num_2_qb_gate = 0, 0
    single_qubit_gates = {"ry", "rz", "rx", "h", "id"}
    for gate in circuit:
        if gate[0] in single_qubit_gates:
            bitmap[gate[1]].append(gate)
            num_1_qb_gate += 1
        else:
            q1, q2 = gate[1], gate[2]
            if len(bitmap[q1]) >= len(bitmap[q2]):
                qb_long, qb_short = q1, q2
            else:
                qb_long, qb_short = q2, q1
            while len(bitmap[qb_short]) < len(bitmap[qb_long]):
                bitmap[qb_short].append(None)
            qb_low, qb_high = (q2, q1) if q1 > q2 else (q1, q2)
            bitmap[qb_low].append(gate)
            bitmap[qb_high].append(None)
            num_2_qb_gate += 1
    num_bit = [len(row) for row in bitmap]
    depth = max(num_bit) if num_bit else 0
    unique = []
    for i in range(depth):
        for j in range(num_qubit):
            if num_bit[j] > i and bitmap[j][i] is not None:
                unique.append(bitmap[j][i])
    return unique.copy(), depth, num_1_qb_gate + num_2_qb_gate, num_2_qb_gate


def circuit_from_parameters(base_params, num_qubit=10, num_layer=1):
    expected = num_layer * (2 * num_qubit - 1)
    if len(base_params) != expected:
        raise ValueError(f"Expected {expected} parameters, received {len(base_params)}")
    circuit = [["h", q] for q in range(num_qubit)]
    param_idx = 0
    for _ in range(num_layer):
        for q in range(num_qubit):
            circuit.append(["rx", q, base_params[param_idx]])
            param_idx += 1
        for start in (0, 1):
            for q in range(start, num_qubit - 1, 2):
                theta = base_params[param_idx]
                param_idx += 1
                circuit.append(["cont", q, q + 1])
                circuit.append(["rz", q + 1, theta])
                circuit.append(["cont", q, q + 1])
    return list(make_it_unique(circuit, num_qubit))


def generate_sample(num_qubit=10, num_layer=1):
    params = []
    for _ in range(num_layer):
        for q in range(num_qubit):
            params.append(np.random.uniform(-np.pi, np.pi))
        for q in range(0, num_qubit - 1, 2):
            params.append(np.random.uniform(-np.pi, np.pi))
        for q in range(1, num_qubit - 1, 2):
            params.append(np.random.uniform(-np.pi, np.pi))
    return circuit_from_parameters(params, num_qubit, num_layer), params


def sample_circuits(seed, count=5000, num_qubit=10, num_layer=1):
    if count < 1:
        raise ValueError("Sample count must be positive")
    random.seed(seed)
    np.random.seed(seed)
    samples, parameters = [], []
    for _ in range(count):
        sample, params = generate_sample(num_qubit, num_layer)
        samples.append(sample)
        parameters.append(params)
    return samples, parameters
