from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit.primitives import Estimator
from qiskit_algorithms.optimizers import P_BFGS
from qiskit_algorithms import VQE
from qiskit_algorithms import NumPyMinimumEigensolver
from ct.rhf_energy import rhf_energy
from ct.ct import canonical_transform
import psi4
import numpy as np

from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.formats.qcschema_translator import qcschema_to_problem
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.algorithms.initial_points import MP2InitialPoint
MAPPER = JordanWignerMapper()
NUMPY_SOLVER = NumPyMinimumEigensolver()


# set geometry and basis
psi_mol = psi4.geometry(
    """
    H  0 0 0
    H 0 0 r
    symmetry c1
    """
)
psi4.core.clean_options()
psi4.core.clean()
psi4.core.clean_variables()
psi4.set_output_file("ct_ucc_test.out", True)
B_BASIS = "cc-pVDZ"
GAMMA = 0.6
bond_length = np.concatenate([np.linspace(0.4, 1.0, 7),
                              np.linspace(1.2, 2.0, 5)])
EHF = []
EHF_CCPVTZ = []
ECTHF = []
hf_energy = []
Eq_UCCSD = []
CT_F12_q_UCCSD = []
ECCSD = []


def do_ct(mol, r, b_basis, gamma):
    mol.r = r
    psi4.set_options({'basis': b_basis,
                      'df_basis_mp2': "cc-pVDZ-F12-Optri",
                      'scf_type': 'pk',
                      'maxiter': 40,
                      'screening': 'csam',
                      'e_convergence': 1e-10})
    e_hf, wfn = psi4.energy("scf", molecule=mol, return_wfn=True)
    basis = psi4.core.get_global_option('BASIS')
    df_basis = psi4.core.get_global_option('DF_BASIS_MP2')
    print("regular hf energy is ", e_hf)
    print("run ct scf")
    h_ct = canonical_transform(
        mol, wfn, basis, df_basis, gamma=gamma, frezee_core=False)
    rhf_ct = rhf_energy(psi_mol, wfn, h_ct)
    print("ct  hf energy is ", rhf_ct["escf"],
          " correlation energy is ", rhf_ct["escf"]-e_hf)
    return h_ct, rhf_ct


def convert_ct_to_quccsd(h_ct, hf_ct):
    h1e_ct = np.copy(h_ct['Hbar1'])
    h2e_ct = np.copy(h_ct['Hbar2'])
    h2e_ct = np.einsum("ijkl->ikjl", h2e_ct)  # to ao basis
    cp_ct = hf_ct['C']
    return h1e_ct, h2e_ct, cp_ct


def run_psi4_large_basis_scf(mol):
    print("Run psi4 cc-pvtz  rhf")
    psi4.core.clean_options()
    psi4.core.clean()
    psi4.core.clean_variables()
    psi4.set_options({'basis': "cc-pvtz",
                      'df_basis_mp2': "cc-pVDZ-F12-Optri",
                      'scf_type': 'pk',
                      'maxiter': 40,
                      'screening': 'csam',
                      'e_convergence': 1e-10})
    e_hf = psi4.energy("scf", molecule=mol)
    return e_hf


def run_qiskit_uccsd(problem, mapper):
    ansatz = UCCSD(
        problem.num_spatial_orbitals,
        problem.num_particles,
        mapper,
        initial_state=HartreeFock(
            problem.num_spatial_orbitals,
            problem.num_particles,
            mapper,
        )
    )
    vqe_solver = VQE(Estimator(), ansatz, P_BFGS())
    initial_point=MP2InitialPoint()
    initial_point.ansatz=ansatz
    initial_point.problem=problem
    vqe_solver.initial_point = initial_point.to_numpy_array()
    calc = GroundStateEigensolver(mapper, vqe_solver)
    result = calc.solve(problem)
    print("log uccsd reulst", result)
    return result


def construct_ct_problem(original_schema, ct_hamiltonian, ct_scf):
    ct_h1e, ct_h2e, mocoeff_ct = convert_ct_to_quccsd(ct_hamiltonian, ct_scf)
    schmea_dict = original_schema.to_dict().copy()
    schmea_dict["wavefunction"]['scf_eri'] = list(ct_h2e.flatten())
    schmea_dict["wavefunction"]['scf_fock_a'] = list(ct_h1e.flatten())
    schmea_dict["wavefunction"]["scf_orbitals_a"] = list(mocoeff_ct.flatten())
    scf_fock_mo_a = mocoeff_ct.T@ct_h1e@mocoeff_ct
    schmea_dict["wavefunction"]["scf_fock_mo_a"] = list(
        scf_fock_mo_a.flatten())
    scf_eri_mo_aa = np.einsum(
        "ijkl,iI,jJ,kK,lL->IJKL", ct_h2e, mocoeff_ct, mocoeff_ct, mocoeff_ct, mocoeff_ct)
    schmea_dict["wavefunction"]["scf_eri_mo_aa"] = list(
        scf_eri_mo_aa.flatten())
    ct_schema = original_schema.from_dict(schmea_dict)
    problem = qcschema_to_problem(ct_schema)
    return problem


for r_h2 in bond_length:
    print("at bond length {}".format(r_h2))
    psi4.core.clean_options()
    psi4.core.clean()
    psi4.core.clean_variables()
    # set up mol
    driver = PySCFDriver(
        atom="H 0 0 0; H 0 0 {}".format(r_h2),
        basis=B_BASIS,
        charge=0,
        spin=0,
        unit=DistanceUnit.ANGSTROM
    )
    # generate es_problem and es_schema
    # used as input for uccsd and ct_uccsd
    es_problem = driver.run()
    es_schema = driver.to_qcschema()
    
    print("log run pyscf rhf")
    print('log eRHF', es_problem.reference_energy)

    print("log run  q-usscd")
    res = run_qiskit_uccsd(problem=es_problem, mapper=MAPPER)
    print("log uccsd reulst", res)
    EHF.append(es_problem.reference_energy)
    Eq_UCCSD.append(res.total_energies)

    # do ct
    print("log run psi4 scf")
    Hct, ct_rhf = do_ct(psi_mol, r_h2, B_BASIS, GAMMA)
    ct_escf = ct_rhf['escf']
    ECTHF.append(ct_escf)

    print("log run ct-q-uccsd")
    ct_problem = construct_ct_problem(es_schema, Hct, ct_rhf)
    ct_res = res = run_qiskit_uccsd(problem=ct_problem, mapper=MAPPER)
    print("log ct uccsd")
    print(ct_res)
    CT_F12_q_UCCSD.append(ct_res.total_energies)
    e_hf_tz = run_psi4_large_basis_scf(psi_mol)
    EHF_CCPVTZ.append(e_hf_tz)
# save data
results = {}
results["EHF"] = np.array(EHF)
results["EHF_CCPVTZ"] = np.array(EHF_CCPVTZ)
results["ECTHF"] = np.array(ECTHF)
results["hf_energy"] = np.array(hf_energy)
results["Eq_UCCSD"] = np.array(Eq_UCCSD)
results["CT_F12_q_UCCSD"] = np.array(CT_F12_q_UCCSD)
np.savez("qiskit_dz.npz", **results)
