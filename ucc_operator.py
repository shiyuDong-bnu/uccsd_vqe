import cirq
def basis_trans(ind,type,qubits):
    assert type in "xyz"
    if type=="x":
        yield cirq.H(qubits[ind])
    if type=="y":
        yield cirq.XPowGate(exponent=-0.5)(qubits[ind])
def basis_backtrans(ind,type,qubits):
    assert type in "xyz"
    if type=="x":
        yield cirq.H(qubits[ind])
    if type=="y":
        yield cirq.XPowGate(exponent=0.5)(qubits[ind])  
def cnot_stair_2(inds,rot_ang,qubits):
    """
    cnot stari used in siglet 
    inds of order dl
    rot_ang signed value
    """
    occ_ind=inds[1]
    vir_ind=inds[0]
    for i in range(occ_ind,vir_ind):
        yield cirq.CNOT(qubits[i],qubits[i+1])
    yield cirq.ZPowGate(exponent=rot_ang)(qubits[vir_ind])
    for i in range(vir_ind,occ_ind,-1):
        yield cirq.CNOT(qubits[i-1],qubits[i]) 
def cnot_stair_4(inds,rot_ang,qubits):
    """inds with order cdlk"""
    occ_begin=inds[-1]
    occ_end=inds[-2]
    vir_begin=inds[0]
    vir_end=inds[1]
    for i in range(occ_begin,occ_end):
        yield cirq.CNOT(qubits[i],qubits[i+1])
    yield cirq.CNOT(qubits[occ_end],qubits[vir_begin]) 
    for i in range(vir_begin,vir_end):
        yield cirq.CNOT(qubits[i],qubits[i+1]) 
    yield cirq.ZPowGate(exponent=rot_ang)(qubits[vir_end])
    for i in range(vir_end,vir_begin,-1):
        yield cirq.CNOT(qubits[i-1],qubits[i]) 
    yield cirq.CNOT(qubits[occ_end],qubits[vir_begin])
    for i in range(occ_end,occ_begin,-1):
        yield cirq.CNOT(qubits[i-1],qubits[i]) 

def double_excitation_typed(inds,type,theta,qubits):
    """
    inds with order cdkl
    type sring "xxxy"
    theta singed value
    """
    for i,s in zip(inds,type):
        yield basis_trans(i,s,qubits)
    yield cnot_stair_4(inds,theta,qubits)
    for i,s in zip(inds,type):
        yield basis_backtrans(i,s,qubits)
def single_excitation_typed(inds,type,theta,qubits):
    """
    inds with order dl
    type sring "xy"
    theta singed value
    """
    for i,s in zip(inds,type):
        yield basis_trans(i,s,qubits)
    yield cnot_stair_2(inds,theta,qubits)
    for i,s in zip(inds,type):
        yield basis_backtrans(i,s,qubits)
def double_excitation_fix_ind(inds,theta,qubits):
    """inds is with order cdlk"""
    SIGN={"xxxy":1,
          "xxyx":1,
          "yyxy":-1,
          "yyyx":-1,
          "yxxx":-1,
          "xyxx":-1,
          "yxyy":1,
          "xyyy":1,}
    for key,sign in SIGN.items():
        yield double_excitation_typed(inds,key,sign*theta/8,qubits)
def single_excitation_fix_ind(inds,theta,qubits):
    """inds with order dl"""
    SIGN={"xy":1,
          "yx":-1}
    for key,sign in SIGN.items():
        yield single_excitation_typed(inds,key,sign*theta/2,qubits)    