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
def get_energy_observable(jw_hamiltonian,qubits):
    energy_observable=cirq.PauliSum()
    for key,value in jw_hamiltonian.terms.items():
        if len(key)==0:
            energy_observable+=value
        else:
            total_op=cirq.PauliString()
            for i in key:
                op=eval("cirq.{}".format(i[1]))(qubits[i[0]])
                total_op*=op
            total_op*=value.real
            energy_observable+=total_op
    return energy_observable
