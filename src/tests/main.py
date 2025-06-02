from HC_Lattice import HCLattice
n_sites = [3, 3]
pbc = False
static_charges = None # {(0,0): 1, (1, 2): -1}
encoding = 'ed' # only ed works currently: gray not possible
HCL = HCLattice(n_sites=n_sites, pbc=pbc) # HyperCubeLattice


#HCL.draw_graph_func(gauss_law_fig=False,static_charges=static_charges)


from Hamiltonian_QED_sym import HamiltonianQED_sym
from Hamiltonian_QED_oprt import HamiltonianQED_oprt
config = {
    'latt': HCL,
    'n_sites': n_sites,
    'pbc': pbc,
    'puregauge': True,
    'static_charges_values': static_charges,
    'e_op_out_plus': False,
    'magnetic_basis': False,
    'encoding': encoding,
    'l': 1,
    'L': 1,
}
HS = HamiltonianQED_sym(config, display_hamiltonian=False) # Hamiltonian(as)Symbols
HO = HamiltonianQED_oprt(config, HS, sparse_pauli=True) # Hamiltonian(as)Operators


#HCL.draw_graph_func(gauss_law_fig=True, e_op_free=HO.e_op_free, static_charges=static_charges)


import primme
import scipy as sp

# Encoded Hamiltonian
EH = None
if encoding == 'ed':
    # here as Pauli matricies
    EH = HO.get_hamiltonian(g_var=1., m_var=0, omega=0, fact_b_op=0.5, fact_e_op=0.5, lambd=1e3)

    # Compress Sparse Row: faster computation
    EH = sp.sparse.csr_matrix(EH)
elif encoding == 'gray':
    # no encoding needed
    EH = HO



print(EH.toarray())


# Eigenwerte, Eigenbasis
EW, EB = primme.eigsh(EH, k=3, which='SA')

idx = EW.argsort()
EW = EW[idx]
EB = EB[:, idx]


for i, wert, in enumerate(EW):
    print('E_%d=%.2f' % (i, wert)) if wert<9e2 else print('', end='')
