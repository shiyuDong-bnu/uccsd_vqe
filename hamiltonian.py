from openfermion.chem import MolecularData
from openfermionpsi4 import run_psi4
from openfermion.transforms import  get_fermion_operator,jordan_wigner
# Set molecule parameters.
basis = 'sto-3g'
multiplicity = 1
bond_length_interval = 0.2
n_points = 10

# Set calculation parameters.
run_scf = 1
run_mp2 = 0
run_cisd = 0
run_ccsd = 0
run_fci = 0
delete_input = True
delete_output = True

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
print(molecule.hf_energy)
h_mol=molecule.get_molecular_hamiltonian()
h_mol_fermion=get_fermion_operator(h_mol)
jw_h_mol=jordan_wigner(h_mol_fermion)
print(jw_h_mol)
