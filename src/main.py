import numpy as np
import networkx as nx
import sympy as sp
from scipy.linalg import eigh
from scipy.sparse.linalg import eigs
from scipy import sparse
from scipy.sparse import csr_matrix
import primme
import matplotlib.pyplot as plt

def to_base_10(digits: np.ndarray, base: int) -> int:
    return (digits * base**np.arange(digits.shape[0])).sum()

def build_edges(config: dict) -> nx.MultiGraph: 
    '''
        Initiate all edges and remove periodic edges if they are unwanted (pbc=False).
    '''
    n_x = config['n_x'] if 'n_x' in config else config['n']
    n_y = config['n_y'] if 'n_y' in config else config['n']
    pbc = config['pbc']

    G = nx.MultiGraph()

    print('>>building edges..')  # init all edges
    for x in range(n_x):
        for y in range(n_y):
            for direction, (dx, dy) in zip(['x', 'y'], [(1, 0), (0, 1)]):
                idx = f'(%d, %d), %s'%(x, y, direction)
                G.add_edge((x, y), ((x+dx)%n_x, (y+dy)%n_y),
                           E=sp.Symbol(f'E_{{{idx}}}'), U=sp.Symbol(f'U_{{{idx}}}'),
                           direction=direction, position=(x, y))
            G.nodes[(x, y)]['static_charge'] = 0

    # if not pbc remove periodic edges of x and y rim
    if not pbc:
        print('>>removing periodic edges..')
        for y in range(n_y):
            G.remove_edge((0, y), (n_x-1, y))
        for x in range(n_x):
            G.remove_edge((x, 0), (x, n_y-1))

    return G

def add_static_charges(G: nx.MultiGraph, config: dict) -> nx.MultiGraph:
    '''
        Add static charges to the nodes.
    '''
    static_charges = config['static_charges']

    print('>>adding static charges..')  # add static charges to nodes
    for pos in static_charges.keys():
        print(pos)
        G.nodes[pos]['static_charge'] = static_charges[pos]

    return G

def build_linear_system(G: nx.MultiGraph) -> list:
    '''
        Build the linear system of equations from the gauss law.
    '''

    print('>>building linear system from gauss law..')  # use gauss law of nodes for linear system 
    linear_system = list()
    for pos in G.nodes:
        gauss_eq = -G.nodes[pos]['static_charge']
        for edge in G.edges(pos, data=True):
            if edge[2]['position'] == pos:
                gauss_eq += edge[2]['E']
            else:
                gauss_eq -= edge[2]['E']
        G.nodes[pos]['gauss_eq'] = gauss_eq
        linear_system.append(gauss_eq)

    return linear_system

def solve_linear_system(G: nx.MultiGraph, linear_system: list) -> tuple:
    '''
        Solve the linear system of equations and associate the solutions to the edges.
    '''

    print('>>solving linear system..')  # solve linear system
    e_op_list = [edge[2]['E'] for edge in list(G.edges(data=True))]
    solution = sp.linsolve(linear_system, e_op_list)
    free_e_op_list = solution.free_symbols
    print(f'\nFree electric field operators: %d\n'%(len(free_e_op_list)))

    print('>>associating solutions to edges..')  # associate solutions to edges
    solution_list = (next(iter(solution)))
    for i, edge in enumerate(G.edges(data=True)):
        edge[2]['gauss_sol'] = solution_list[i]

    return free_e_op_list, solution_list

def build_base(free_e_op_list: list, config: dict) -> np.ndarray:
    '''
        Build the base of the physical subspace.
    '''
    l = config['l']

    digit_base = 2*l+1
    
    print('>>building base of physical space..')  # build base of physical space
    phys_dim = digit_base ** len(free_e_op_list)
    print(f'\nDimension of physical Hamiltonian: %d x %d\n'%(phys_dim, phys_dim))

    numbers = np.arange(phys_dim)
    base = np.array(((numbers[:, None] // digit_base**np.arange(len(free_e_op_list)-1, -1, -1)) % digit_base) - l)
    return base

def build_plaquettes(G: nx.MultiGraph, config: dict) -> nx.MultiGraph:
    '''
        Build the plaquettes of the lattice.
    '''
    n_x = config['n_x'] if 'n_x' in config else config['n']
    n_y = config['n_y'] if 'n_y' in config else config['n']

    print('>>building plaquettes..')# build plaquettes
    free_edge_list = [edge for edge in G.edges(data=True) if edge[2]['gauss_sol'] == edge[2]['E']]
    for pos in G.nodes:
        G.nodes[pos]['plaquette'] = list()
    for edge in free_edge_list:
        (x, y) = edge[2]['position']
        if edge[2]['direction'] == 'x':
            G.nodes[(x, y)]['plaquette'].append({'index': free_edge_list.index(edge), 'shift': 1})
            G.nodes[(x, (y-1)%n_y)]['plaquette'].append({'index': free_edge_list.index(edge), 'shift': -1})
        if edge[2]['direction'] == 'y':
            G.nodes[(x, y)]['plaquette'].append({'index': free_edge_list.index(edge), 'shift': -1})
            G.nodes[((x-1)%n_x, y)]['plaquette'].append({'index': free_edge_list.index(edge), 'shift': 1})

    return G

def build_total_electric_operator(solution_list: list, free_e_op_list: list, base: np.ndarray) -> sparse.csr_matrix:
    '''
        Build the electric operator of the lattice.
    '''
    l = config['l']

    digit_base = 2*l+1
    phys_dim = base.shape[0]

    print('>>building electric hamiltonian..')  # build electric hamiltonian
    E = {'row': np.arange(phys_dim, dtype=np.int32),
           'column': np.arange(phys_dim, dtype=np.int32),
           'data':np.zeros(phys_dim)}

    E_OP = 0
    for equation in solution_list:
        E_OP += ((equation+l)%digit_base-l)**2

    for i in E['row']:
        E['data'][i] = E_OP.subs(zip(free_e_op_list, base[i]))/2

    E_sparse = sparse.csr_matrix((E['data'], (E['row'], E['column'])))
    return E_sparse
    

def build_total_plaquette_operator(G: nx.MultiGraph, base: np.ndarray) -> sparse.csr_matrix:
    '''
        Build the plaquette operator of the lattice.
    '''
    l = config['l']
    g = config['g']

    digit_base = 2*l+1
    phys_dim = base.shape[0]

    print('>>building magnetic hamiltonian..')  # build magnetic hamiltonian
    P = {'row': np.arange(phys_dim, dtype=np.int32),
         'column': np.arange(phys_dim, dtype=np.int32),
         'data':np.zeros(phys_dim)}

    for i in range(phys_dim):
        for pos in G.nodes:
            plaquette = G.nodes[pos]['plaquette']
            u = base[i]
            if not not plaquette:
                for op in plaquette:
                    u[op['index']] = (u[op['index']] + op['shift']) % digit_base
                j = to_base_10(u, digit_base)
                P['column'][i] = j
                P['data'][i] = -1/2
                #H_B[i, j] = H_B_data[i]

    P_sparse = sparse.csr_matrix((P['data'], (P['row'], P['column'])))
    return (P_sparse+P_sparse.T)/2

def diagonalize_hamiltonian(E_sparse: sparse.csr_matrix, B_sparse: sparse.csr_matrix, config: dict) -> tuple:
    '''
        Diagonalize the hamiltonian.
    '''
    g = config['g']

    k = 6
    phys_dim = E_sparse.shape[0]

    H_sparse = g**2*E_sparse+B_sparse/g**2

    if phys_dim < k:
        H_sparse = H_sparse.toarray()

    EW, EB = sparse.linalg.eigsh(H_sparse, k=k, which='LM')
    return EW, EB




config = {'l': 1,
          'g': 1,
          'n': 3,
          'k': 6, 
          'static_charges': {}, 
          'pbc': False}

G = build_edges(config)
G = add_static_charges(G, config)
linear_system = build_linear_system(G)
free_e_op_list, solution_list = solve_linear_system(G, linear_system)
base = build_base(free_e_op_list, config)
G = build_plaquettes(G, config)
E_sparse = build_total_electric_operator(solution_list, free_e_op_list, base)
B_sparse = build_total_plaquette_operator(G, base)
#diagonalize_hamiltonian(E_sparse, B_sparse, config)

betas = np.arange(10)*0.2+0.8
P_expectaiton_values = np.zeros(betas.shape[0])
for i, beta in enumerate(betas):
    config['g'] = (1/beta)**(1/2)
    EW, EB = diagonalize_hamiltonian(E_sparse, B_sparse, config)

    idx = EW.argsort()
    EW = EW[idx]
    EB = EB[:, idx]

    ground_state = EB[:, 0]

    P_expectaiton_values[i] = ground_state @ B_sparse @ ground_state.T

plt.plot(betas, P_expectaiton_values)
plt.show()
