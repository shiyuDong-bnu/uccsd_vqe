import cirq


def get_energy(circuit, jw_hamiltonian, qubits):
    energy = 0
    simulator = cirq.Simulator()
    for key in jw_hamiltonian.terms:
        if len(key) == 0:
            energy += jw_hamiltonian.terms[key]
        else:
            measure_ansatz = circuit.copy()
            each_ops = [eval("cirq."+ops[1])(qubits[ops[0]]) for ops in key]
            total_op = 1
            for op in each_ops:
                total_op *= op
            ev_list = simulator.simulate_expectation_values(
                measure_ansatz, observables=[total_op]
            )
            energy += ev_list[0]*jw_hamiltonian.terms[key]
    return energy
