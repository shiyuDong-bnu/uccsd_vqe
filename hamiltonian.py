from openfermion.chem import MolecularData
from openfermionpsi4 import run_psi4
from openfermion.transforms import  get_fermion_operator,jordan_wigner
# Set molecule parameters.
basis = 'sto-3g'
multiplicity = 1

# Set calculation parameters.
run_scf = 1
run_mp2 = 0
run_cisd = 0
run_ccsd = 0
run_fci = 0


# Generate molecule at different bond lengths.
bond_length=0.7
geometry = [('H', (0., 0., 0.)), ('H', (0., 0., bond_length))]
molecule = MolecularData(
    geometry, basis, multiplicity,
    description=str(round(bond_length, 2)))

# Run Psi4.
molecule = run_psi4(molecule,
                    run_scf=run_scf,
                    run_mp2=run_mp2,
                    run_cisd=run_cisd,
                    run_ccsd=run_ccsd,
                    run_fci=run_fci)
h_mol=molecule.get_molecular_hamiltonian()
print(h_mol)
h_mol_fermion=get_fermion_operator(h_mol)
jw_h_mol=jordan_wigner(h_mol_fermion)
# print(jw_h_mol)


## manually do molecule
import psi4
import numpy as np
from psi4.core import MintsHelper
psi4.set_output_file("test.out")
mol=psi4.geometry("""
    H 0. 0. 0.
    H 0. 0. 0.7
    symmetry c1
    """)
psi4.set_options({
    "basis":"sto-3g",
    "scf_type":"pk",
})
e_scf,wfn=psi4.energy("hf",return_wfn=True)

print("hf energy is ",e_scf)
n_electron=wfn.nalpha()
print(f"Active Molecule has {n_electron*2} total electron")
mints=MintsHelper(wfn)
h_core=wfn.H().to_array()
h_eri=mints.ao_eri().to_array().swapaxes(1,2).swapaxes(2,3)
coef=wfn.Ca()
h_core_mol=np.einsum("ij,iI,jJ->IJ",h_core,coef,coef)
h_eri_mol=np.einsum("ijkl,iI,jJ,kK,lL->IJKL",h_eri,coef,coef,coef,coef)
n_spatial_orbital=wfn.nmo()
n_spin=2*n_spatial_orbital
h_core_spin=np.zeros((n_spin,n_spin))
eri_spin=np.zeros((n_spin,n_spin,n_spin,n_spin))
for i in range(n_spin):
    for j in range(n_spin):
        if i%2==j%2: # same spin type
            h_core_spin[i,j]=h_core_mol[i//2,j//2]
for i in range(n_spin):
    for j in range(n_spin):
        for k in range(n_spin):
            for l in range(n_spin):
                if i%2==l%2 and k%2==j%2:
                    eri_spin[i,j,k,l]=h_eri_mol[i//2,j//2,k//2,l//2]
import openfermion.ops.representations as reps
constant=mol.nuclear_repulsion_energy()
one_body_coefficients=h_core_spin
two_body_coefficients=eri_spin
# Truncate.
one_body_coefficients[np.absolute(one_body_coefficients) < 1e-13] = 0.0
two_body_coefficients[np.absolute(two_body_coefficients) < 1e-13] = 0.0
mol_h_diy = reps.InteractionOperator(
            constant, one_body_coefficients, 1 / 2 * two_body_coefficients
        )
print(np.allclose(h_mol.two_body_tensor,mol_h_diy.two_body_tensor))
