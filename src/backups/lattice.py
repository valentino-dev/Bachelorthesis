import os
import hashlib
import numpy as np
import sympy
import copy
import scipy
import time
import multiprocessing
import pickle

def timeit(func):
    def inner(*args):
        start = time.time()
        result = func(*args)
        stop = time.time()
        print(f'{func} took %.4f seconds.'%(stop-start))
        return result
    return inner


class Site:
    def __init__(self, idx, pos=np.array([]), static_charge = 0., links=dict(), attributes=dict()):
        self.idx = idx
        self.attributes = attributes
        self.links = links
        self.pos = pos
        self.static_charge = 0


    def __str__(self):
        return f'{self.__repr__()}: {self.idx}, {self.pos}, {self.attributes}'

    def __getitem__(self, idx):
        return self.attributes[idx]

    def __setitem__(self, idx, item):
        self.attributes[idx] = item


class Link:
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

    default_config = {
        'l': 1,
        'g': 1,
        'k': 6,
        'dims': (2, 2),
        'pbc': False,
        'static_charges': dict(),
    }

    def __init__(self, config=dict()):
        '''
            Initiating Lattice.
        '''
        self.config = dict()
        for attr in self.default_config.keys():
            self.__setattr__(attr, config[attr] if attr in config else self.default_config[attr])
            self.config[attr] = config[attr] if attr in config else self.default_config[attr]

        self.dims = np.array(self.dims)
        self.projection_vector = np.array([self.dims[:dim_idx].prod() for dim_idx in range(len(self.dims))])
        self.digit_base = 2*self.l+1

        print(f'>>num of used threads: %d'%(multiprocessing.cpu_count()))


    def calculate_E_and_B_op(self):
        self.build()
        self.add_static_charges()
        self.build_linear_system()
        self.solve_linear_system()
        self.build_total_electric_operator()
        self.build_plaquettes()
        self.build_total_plaquette_operator()

    '''
    def __str__(self) -> str:
        dims_str = ''
        for i in range(len(dims)-1):
            dims_str += f'%d_'%(dims[i])
        dims_str += f'%d'%(dims[-1])

        static_charges_str = ''
        for i in range(len(static_charges)-1):
            charge_key = static_charges.keys()[i]
            charge = static_charges.values()[i]
            static_charges_str += f'({charge_key[0]}_{charge_key[1]})_{charge}'

        static_charges_str += f'({charge_key[0]}_{charge_key[1]})_{charge}'


        return f'l_%d.g_%.2f.k_%d.dims_({%s}).pbc_.static_charges_'
        #return f'{[str(site) for site in self]}'
    '''

    def set_param(self, param, value):
        self.config[param] = value
        self.__setattr__(param, value)

    def dict_to_tuple(self, dictionary):
        for key in dictionary:
            if type(dictionary[key]) == dict:
                dictionary[key] = self.dict_to_tuple(dictionary[key])
            
        return tuple(list(dictionary.items()))


    def __hash__(self):
        for key in self.default_config:
            self.config[key] = self.config[key] if key in self.config else self.default_config[key]

        #serialized = json.dumps(self.config, sort_keys=True, separators=(',', ':')).encode('utf-8')
        serialized = str(self.config).encode('utf-8')
        digest = hashlib.sha256(serialized).digest()
        return int.from_bytes(digest[:8], 'big')

    def __eq__(self, other):
        dprint(type(self), type(other))
        return isinstance(other, Lattice) and self.config == other.config
        #return isinstance(other, Lattice) and self.__hash__() == other.__hash__()

    def __iter__(self):
        return iter(self.sites.values())

    def __getitem__(self, idx):
        return self.sites[idx]

    def __setitem__(self, idx, item):
        self.sites[idx] = item


    def idx_to_pos(self, idx: int) -> np.ndarray:
        return np.array([(idx//(self.dims[:dim_idx].prod()))%self.dims[dim_idx] for dim_idx in range(len(self.dims))])

    def pos_to_idx(self, pos: np.ndarray) -> int:
        return int(pos @ self.projection_vector)

    def build(self):
        sites = dict()
        for i in range(self.dims.prod()):
            sites[i] = copy.deepcopy(Site(i, self.idx_to_pos(i)))
        links = dict()

        for u_idx in sites:
            u_site = sites[u_idx]
            for dim_idx in range(len(self.dims)):
                v_pos = copy.deepcopy(u_site.pos)
                v_pos[dim_idx] = (v_pos[dim_idx] + 1) % self.dims[dim_idx]
                v_idx = self.pos_to_idx(v_pos)
                if v_idx > u_idx or self.pbc:
                    link = copy.deepcopy(Link(u_idx, v_idx))
                    link['direction'] = dim_idx
                    link['idx'] = len(links)
                    u_site.links[(u_idx, v_idx)] = link
                    sites[v_idx].links[(u_idx, v_idx)] = link
                    links[(u_idx, v_idx)] = link

        
        self.sites = sites
        self.links = links

    def add_static_charges(self) -> None:
        for pos in self.static_charges.keys():
            self.sites[self.pos_to_idx(pos)].static_charge = self.static_charges[pos]
        
    def build_linear_system(self) -> None:
        '''
            Build the linear system of equations from the gauss law.
        '''

        print('>>building linear system from gauss law..')  # use gauss law of sites for linear system 
        linear_system = list()
        for site in self:
            gauss_eq = -site.static_charge
            for links in site.links.values():
                if links.v == site.idx:
                    gauss_eq -= links.E
                else:
                    gauss_eq += links.E
            site['gauss_eq'] = gauss_eq
            linear_system.append(gauss_eq)

        self.linear_system=linear_system

    def solve_linear_system(self) -> None:
        '''
            Solve the linear system of equations and associate the solutions to the links.
        '''

        print('>>solving linear system..')  # solve linear system
        e_ops = [link.E for link in self.links.values()]

        solution = sympy.linsolve(self.linear_system, e_ops)

        idx_order = np.arange(len(e_ops), dtype=np.int32)


        print('>>associating solutions to links..')  # associate solutions to links 
        self.dynamic_links = list()
        self.dynamic_e_ops = list()
        self.solution_list = [(next(iter(solution)))[i] for i in idx_order.tolist()]

        for i, link in enumerate([list(self.links.values())[i] for i in idx_order.tolist()]):
            link['gauss_sol'] = self.solution_list[i]
            if link['gauss_sol'] == link.E:
                self.dynamic_links.append(link)
                self.dynamic_e_ops.append(link.E)

        
        
        self.phys_dim = self.digit_base**len(self.dynamic_links)
        print(f'Physical Dimension: %.0f'%(self.phys_dim))
        print(f'\nDynamic links: %d\n'%(len(self.dynamic_links)))

    def build_plaquettes(self) -> None:
        '''
            Build the plaquettes of the lattice.
        '''

        print('>>building plaquettes..')  # build plaquettes
        for site in self:
            site['plaquette'] = list()
        for link in self.dynamic_links:
            idx = link.u
            pos = copy.deepcopy(self.sites[idx].pos)

            # If not deggad, then shift = 1. If deggad, then shift = -1.

            if link['direction'] == 0:
                self.sites[idx]['plaquette'].append({'index': self.dynamic_links.index(link), 'shift': 1})
                pos[1] = (pos[1]-1) % self.dims[1]
                self.sites[self.pos_to_idx(pos)]['plaquette'].append({'index': self.dynamic_links.index(link), 'shift': -1})
            elif link['direction'] == 1:
                self.sites[idx]['plaquette'].append({'index': self.dynamic_links.index(link), 'shift': -1})
                pos[0] = (pos[0]-1) % self.dims[0]
                self.sites[self.pos_to_idx(pos)]['plaquette'].append({'index': self.dynamic_links.index(link), 'shift': 1})

        
        if not self.pbc:
            print('>>removing periodic plaquettes..')
            for y in range(self.dims[1]):
                self.sites[self.pos_to_idx(np.array([self.dims[1]-1, y]))]['plaquette'] = list()
            for x in range(self.dims[0]):
                self.sites[self.pos_to_idx(np.array([x, self.dims[1]-1]))]['plaquette'] = list()

    def idx_to_state(self, idx):
        return np.array(((idx // self.digit_base**np.arange(len(self.dynamic_links)-1, -1, -1)) % self.digit_base))

    def state_to_idx(self, digits: np.ndarray) -> int:
        return (digits * self.digit_base**np.arange(digits.shape[0]-1, -1, -1)).sum()

    def get_electric_element(self, idx):
        progress = (idx/self.phys_dim*100)
        #print(f'         Progress: [{str((int(progress/5)*"|")+(int(20-progress/5)*" "))}]: %.0f%%' % (idx/self.phys_dim*100), end='\r')
        return self.E_OP.subs(zip(self.dynamic_e_ops, self.idx_to_state(idx) - self.l))

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
        #print(f'         Progress: [{str((int(progress/5)*"|")+(int(20-progress/5)*" "))}]: %.0f%%' % (idx/self.phys_dim*100), end='\r')
        i_state = self.idx_to_state(idx)
        idx_column = [0]
        idx_data = [0]
        for site in self:
            if not not site['plaquette']:
                f_state = copy.deepcopy(i_state)
                for u_op in site['plaquette']:
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

    @timeit
    def diagonalize_hamiltonian(self) -> None:
        '''
            Diagonalize the hamiltonian.
        '''
        partial_diagonalisation = self.phys_dim > self.k and self.k > 0
        print(f'>>>diagonalizing Hamiltonian {f"to k=%d" % (self.k) if partial_diagonalisation else "entirely"} with g=%.4f..'%(self.g))

        self.H_sparse = self.g**2*self.E-self.B/self.g**2


        if partial_diagonalisation:

            #X = np.random.default_rng(42).normal(size=(self.phys_dim, self.k)) 
            #EW, EB = scipy.sparse.linalg.lobpcg(self.H_sparse, X, largest=False, maxiter=500)
            EW, EB = scipy.sparse.linalg.eigsh(self.H_sparse, k=self.k, which='SA')
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
            self.set_param('g', 1/beta**(1/2))
            self.diagonalize_hamiltonian()
            ground_state = self.EB[:, 0]


            B_expectation_values[i] = ground_state.T @ self.B @ ground_state / num_plaquettes
        return B_expectation_values

def save(lattices):
    with open('../data/lattices.pickle', 'wb') as handle:
        pickle.dump(lattices, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
def load(path=None):
    if not path:
        path = '../data/lattices.pickle'

    if os.path.exists(path):
        lattcies = None
        with open(path, 'rb') as handle:
            lattices = pickle.load(handle)

        return lattices
    else:
        return dict()

def dprint(*args):
    print('debug:', *args)
