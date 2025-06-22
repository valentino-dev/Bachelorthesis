from re import A
import numpy as np
import sympy
import copy
import scipy
import matplotlib.pyplot as plt
import time
import primme
import multiprocessing

plt.rcParams.update({"text.usetex": True, "font.family": "serif"})
plt.rc('font', size=16)

def timeit(func):
    def inner(*args):
        start = time.time()
        result = func(*args)
        stop = time.time()
        print(f'{func} took %.4f seconds.'%(stop-start))
        return result
    return inner

class Node:
    def __init__(self, idx, pos=np.array([]), static_charge = 0., edges=dict(), attributes=dict()):
        self.idx = idx
        self.attributes = attributes
        self.edges = edges
        self.pos = pos
        self.static_charge = 0


    def __str__(self):
        return f'{self.__repr__()}: {self.idx}, {self.pos}, {self.attributes}'

    def __getitem__(self, idx):
        return self.attributes[idx]

    def __setitem__(self, idx, item):
        self.attributes[idx] = item


class Edge:
    def __init__(self, u, v, attributes=dict()):
        self.u = u
        self.v = v
        self.attributes = attributes

        self.E=sympy.Symbol(f'E_{{{u} to {v}}}')
        self.U=sympy.Symbol(f'U_{{{u} to {v}}}')

    def __str__(self):
        return f'{self.u} to {self.v}'

    def __getitem__(self, idx):
        return self.attributes[idx]

    def __setitem__(self, idx, item):
        self.attributes[idx] = item


class Lattice:
    def __init__(self, config=dict()):
        '''
            Initiating Lattice.
        '''
        default_config = {
            'l': 1,
            'g': 1,
            'k': 3,
            'dims': (2, 2),
            'pbc': False,
            'static_charges': dict()
        }

        for attr in default_config.keys():
            self.__setattr__(attr, config[attr] if attr in config else default_config[attr])

        self.dims = np.array(self.dims)
        self.projection_vector = np.array([self.dims[:dim_idx].prod() for dim_idx in range(len(self.dims))])
        self.digit_base = 2*self.l+1

        self.build()
        self.add_static_charges()
        self.build_linear_system()
        self.solve_linear_system()
        self.build_total_electric_operator()
        self.build_plaquettes()
        self.build_total_plaquette_operator()

    def __str__(self) -> str:
        return f'{[str(node) for node in self]}'

    def __iter__(self):
        return iter(self.nodes.values())

    def __getitem__(self, idx):
        return self.nodes[idx]

    def __setitem__(self, idx, item):
        self.nodes[idx] = item


    def idx_to_pos(self, idx: int) -> np.ndarray:
        return np.array([(idx//(self.dims[:dim_idx].prod()))%self.dims[dim_idx] for dim_idx in range(len(self.dims))])

    def pos_to_idx(self, pos: np.ndarray) -> int:
        return int(pos @ self.projection_vector)

    def build(self):
        nodes = dict()
        for i in range(self.dims.prod()):
            nodes[i] = copy.deepcopy(Node(i, self.idx_to_pos(i)))
        edges = dict()

        for u_idx in nodes:
            u_node = nodes[u_idx]
            for dim_idx in range(len(self.dims)):
                v_pos = copy.deepcopy(u_node.pos)
                v_pos[dim_idx] = (v_pos[dim_idx] + 1) % self.dims[dim_idx]
                v_idx = self.pos_to_idx(v_pos)
                if v_idx > u_idx or self.pbc:
                    edge = Edge(u_idx, v_idx)
                    edge['direction'] = dim_idx
                    u_node.edges[(u_idx, v_idx)] = edge
                    nodes[v_idx].edges[(u_idx, v_idx)] = edge
                    edges[(u_idx, v_idx)] = edge
        
        self.nodes = nodes
        self.edges = edges

    def add_static_charges(self) -> None:
        for pos in self.static_charges.keys():
            self.nodes[self.pos_to_idx(pos)].static_charge = self.static_charges[pos]
        
    def build_linear_system(self) -> None:
        '''
            Build the linear system of equations from the gauss law.
        '''

        print('>>building linear system from gauss law..')  # use gauss law of nodes for linear system 
        linear_system = list()
        for node in self:
            gauss_eq = -node.static_charge
            #print(node.edges.keys())
            for edge in node.edges.values():
                if edge.v == node.idx:
                    gauss_eq += edge.E
                else:
                    gauss_eq -= edge.E
            #print(gauss_eq)
            node['gauss_eq'] = gauss_eq
            linear_system.append(gauss_eq)

        self.linear_system=linear_system

    def solve_linear_system(self) -> None:
        '''
            Solve the linear system of equations and associate the solutions to the edges.
        '''

        print('>>solving linear system..')  # solve linear system
        e_op_list = [edge.E for edge in self.edges.values()]

        solution = sympy.linsolve(self.linear_system, e_op_list)
        free_e_op_list = list(set(solution.free_symbols))

        code = [np.array([ord(char)*1e3**i for i, char in enumerate(str(e))]).sum() for e in free_e_op_list ]
        
        free_e_op_list = [free_e_op_list[i] for i in np.argsort(code)[::-1]]

        print(f'\nFree electric field operators: %d\n'%(len(free_e_op_list)))

        print('>>associating solutions to edges..')  # associate solutions to edges
        solution_list = (next(iter(solution)))
        for i, edge in enumerate(self.edges.values()):
            edge['gauss_sol'] = solution_list[i]
        
        #print(e_op_list)
        #print(solution_list)
        
        self.phys_dim = self.digit_base**len(free_e_op_list)
        print(f'Physical Dimension: %.0f'%(self.phys_dim))

        self.free_e_op_list = free_e_op_list
        self.solution_list = solution_list

    def build_plaquettes(self) -> None:
        '''
            Build the plaquettes of the lattice.
        '''

        print('>>building plaquettes..')  # build plaquettes
        free_edge_list = [edge for edge in self.edges.values() if edge['gauss_sol'] == edge.E]
        for node in self:
            node['plaquette'] = list()
        for edge in free_edge_list:
            idx = edge.u
            pos = copy.deepcopy(self.nodes[idx].pos)

            # If not deggad, then shift = 1. If deggad, then shift = -1.

            if edge['direction'] == 0:
                self.nodes[idx]['plaquette'].append({'index': free_edge_list.index(edge), 'shift': 1})
                pos[1] = (pos[1]-1) % self.dims[1]
                self.nodes[self.pos_to_idx(pos)]['plaquette'].append({'index': free_edge_list.index(edge), 'shift': -1})
            elif edge['direction'] == 1:
                self.nodes[idx]['plaquette'].append({'index': free_edge_list.index(edge), 'shift': -1})
                pos[0] = (pos[0]-1) % self.dims[0]
                self.nodes[self.pos_to_idx(pos)]['plaquette'].append({'index': free_edge_list.index(edge), 'shift': 1})

        if not self.pbc:
            print('>>removing periodic plaquettes..')
            for y in range(self.dims[1]):
                self.nodes[self.pos_to_idx(np.array([self.dims[1]-1, y]))]['plaquette'] = list()
            for x in range(self.dims[0]):
                self.nodes[self.pos_to_idx(np.array([x, self.dims[1]-1]))]['plaquette'] = list()

    def idx_to_state(self, idx):
        return np.array(((idx // self.digit_base**np.arange(len(self.free_e_op_list)-1, -1, -1)) % self.digit_base))

    def state_to_idx(self, digits: np.ndarray) -> int:
        return (digits * self.digit_base**np.arange(digits.shape[0]-1, -1, -1)).sum()

    def get_electric_element(self, idx):
        progress = (idx/self.phys_dim*100)
        print(f'         Progress: [{str((int(progress/5)*"|")+(int(20-progress/5)*" "))}]: %.0f%%' % (idx/self.phys_dim*100), end='\r')
        return self.E_OP.subs(zip(self.free_e_op_list, self.idx_to_state(idx) - self.l))

    @timeit
    def build_total_electric_operator(self) -> None:
        '''
            Build the electric operator of the lattice.
        '''

        print('>>building electric hamiltonian..')  # build electric hamiltonian
        E = {'row': np.arange(self.phys_dim, dtype=np.int32),
               'column': np.arange(self.phys_dim, dtype=np.int32),
               'data':np.zeros(self.phys_dim)}

        self.E_OP = 0
        for equation in self.solution_list:
            self.E_OP += equation**2

        pool_obj = multiprocessing.Pool()
        tasks = E['row']

        result = pool_obj.map(self.get_electric_element, tasks)
        
        '''

        result = list()
        for i, job_result in enumerate(pool_obj.imap_unordered(self.get_electric_element, tasks)):
            print('[%.0f%%]' % (i/len(tasks)*100), end='\r')
            result.append(job_result)
        '''


        E['data'] = np.array(result, dtype=np.float32)
        self.E = scipy.sparse.csr_matrix((E['data'], (E['row'], E['column']))) / 2

    def get_magnetic_vector(self, idx):
        progress = (idx/self.phys_dim*100)
        print(f'         Progress: [{str((int(progress/5)*"|")+(int(20-progress/5)*" "))}]: %.0f%%' % (idx/self.phys_dim*100), end='\r')
        i_state = self.idx_to_state(idx)
        idx_column = [0]
        idx_data = [0]
        for node in self:
            if not not node['plaquette']:
                f_state = copy.deepcopy(i_state)
                for u_op in node['plaquette']:
                    f_state[u_op['index']] += u_op['shift']
                
                # if all shifts dont exit truncation (if not one exits trunction), the plaquette operation makes a contribution
                if not (f_state - (f_state % self.digit_base)).any():
                    idx_column.append(self.state_to_idx(f_state))
                    idx_data.append(1.)

        return [[idx]*len(idx_column), idx_column, idx_data]

                    
    @timeit
    def build_total_plaquette_operator(self) -> None:
        '''
            Build the plaquette operator of the lattice.
        '''

        print('>>building magnetic hamiltonian..')  # build magnetic hamiltonian

        pool_obj = multiprocessing.Pool()
        tasks = np.arange(self.phys_dim) 


        result = pool_obj.map(self.get_magnetic_vector, tasks)

        '''

        result = list()
        for i, job_result in enumerate(pool_obj.imap(self.get_magnetic_vector, tasks)):
            print('[%.0f%%]' % (i/len(tasks)*100), end='\r')
            result.append(job_result)
        '''
        
        # sorting results
        row = list()
        column = list()
        data = list()
        for xs in result:
            for x in xs[0]:
                row.append(x)
            for x in xs[1]:
                column.append(x)
            for x in xs[2]:
                data.append(x)

        P_sparse = scipy.sparse.csr_matrix((data, (row, column)), shape=(self.phys_dim, self.phys_dim))
        self.B = (P_sparse+P_sparse.T)/2

    def diagonalize_hamiltonian(self) -> None:
        '''
            Diagonalize the hamiltonian.
        '''
        partial_diagonalisation = self.phys_dim > self.k and self.k > 0
        print(f'>>>diagonalizing Hamiltonian {f"to k=%d" % (self.k) if partial_diagonalisation else "entirely"}..')

        self.H_sparse = self.g**2*self.E-self.B/self.g**2

        
        #print((self.H_sparse == 21.0))
        #print('test', self.H_sparse[(self.H_sparse == 21.0)[:, 0]])

        #print(f'Hamiltonian is hermitian: {(self.H_sparse.toarray() == self.H_sparse.toarray().T).all()}')

        if partial_diagonalisation:
            #EW, EB = scipy.sparse.linalg.eigsh(self.H_sparse, k=self.k, which='SA')

            #X = np.random.default_rng(42).normal(size=(self.phys_dim, self.k)) 
            #EW, EB = scipy.sparse.linalg.lobpcg(self.H_sparse, X, largest=False, maxiter=500)

            EW, EB = primme.eigsh(self.H_sparse, k=self.k, which='SA')
            
        else:
            EW, EB = scipy.linalg.eigh(self.H_sparse.toarray())
            idx = EW.argsort()
            EW = EW[idx]
            EB = EB[:, idx]

        self.EW = EW
        self.EB = EB

    def printEW(self):
        idx = self.EW.argsort()
        EW = self.EW[idx]
        
        print('\nEW: ')
        for i in range(3):
            print(f'%.4f'%(EW[i]))

    def printE(self):
        print(f'\nTrace {np.diag(self.E.toarray()).sum()} with Diag of E: {np.diag(self.E.toarray())}')
        #print(f'{self.E.toarray()}')

    def printB(self):
        print(f'\nDiag of B: {np.diag(self.B.toarray())}')
        print(f'{self.B.toarray()}')
        print(f'{self.B.sum()}')

    def get_plaquette_expectation_values(self, beta_list):
        B_expectation_values = np.zeros(len(beta_list))

        num_plaquettes = (self.dims).prod() if self.pbc else (self.dims-1).prod()

        for i, beta in enumerate(beta_list):
            self.g = 1/beta**(1/2)
            self.diagonalize_hamiltonian()
            ground_state = self.EB[:, 0]


            B_expectation_values[i] = ground_state.T @ self.B @ ground_state / num_plaquettes
        return B_expectation_values

            

if __name__ == '__main__':
    EWs = np.zeros(10)
    #for i in range(10):
    #time.sleep(1)

    '''
    lattice = Lattice(config={'k': 3, 'dims': (3, 3), 'pbc': False, 'l':1})
    lattice.diagonalize_hamiltonian()
    lattice.printEW()
    #EWs[i] = lattice.EW[0]

    #print(EWs.std())
    #lattice.printE()
    #lattice.printB()
    '''

    plt.ylabel('$<P>$')
    plt.xlabel(r'$\beta=1/g^2$')
    plt.tight_layout()
    plt.grid()
    for k in np.arange(3, 4, dtype=np.int32):
        for l in np.arange(1)+4:
            lattice = Lattice(config={'k': k, 'dims': (2, 2), 'pbc': True, 'l':l})
            #betas = 10**(np.arange(10e1)*0.2-0.8e1)
            betas = 10**(np.linspace(-2, 2, 20))
            #betas = np.linspace(1e-3, 10, 50)
            p_exp = lattice.get_plaquette_expectation_values(betas)
            plt.plot(betas, p_exp, marker='v', label=f'k=%d, l=%d'%(k, l), linestyle='dashed')
    plt.xscale('log')
    plt.legend()
    plt.savefig('../latex/images/PlaquetteExp2x2PBC.pdf')
    '''

    '''
