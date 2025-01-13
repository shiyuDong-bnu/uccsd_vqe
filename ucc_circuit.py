from ucc_operator import single_excitation_fix_ind,double_excitation_fix_ind
import cirq
def gen_uccsd_anstaz(n_occ,n_qubits,ansatz,qubits,params):
    single_params=params[0]
    double_params=params[1]
    t1_count=0
    for l in range(0,n_occ):
        for d in range(n_occ,n_qubits):
            ansatz.append(single_excitation_fix_ind([d,l],
                                        theta=single_params[t1_count],qubits=qubits))
            t1_count+=1
    t2_count=0
    for k in range(0,n_occ):
        for l in range(k+1,n_occ):
            for c in range(n_occ,n_qubits):
                for d in range(c+1,n_qubits):
                    ansatz.append(double_excitation_fix_ind([c,d,l,k],
                                                theta=double_params[t2_count],qubits=qubits))
                    t2_count+=1
    return ansatz
def resolve_uccsd_ansatz(my_qubits,n_occ,n_vir):
    # my_qubits=[cirq.GridQubit(i,0) for i in range(n_qubits)]
## initial occupation state into 1 ,
## thus the state is \phi_0\alpha \phi_0 \beta \phi_1\alpha\phi_1\beta \cdots \phi_n \alpha \phi_n\beta
    n_qubits=len(my_qubits)
    def initial_circuit(n_electron,qubits):
        for i in range(n_electron):
            yield cirq.X(qubits[i]) 
    
    import sympy
    c=cirq.Circuit()
    c.append(initial_circuit(n_occ,my_qubits))
    n_single_excitation=int(n_occ*n_vir)
    n_double_excitation=int(((n_occ-1)*n_occ)/2 *((n_vir-1)*n_vir)/2)
    single_param=sympy.symbols(f"ts_0:{n_single_excitation}")
    double_param=sympy.symbols(f"td_0:{n_double_excitation}")
    params=(single_param,double_param)
    ansatz=gen_uccsd_anstaz(n_occ,n_qubits,c,my_qubits,params)
    return ansatz