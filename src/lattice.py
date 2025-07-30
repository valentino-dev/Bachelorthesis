import os
import math
import hashlib
import numpy as np
import sympy
import copy
import scipy
import time
import multiprocessing
import pickle
import primme
from functools import partial



states_per_kernal = 16
def magnetic_kernal(sites, digit_base, num_dynamic_links, phys_dim, operation_idx):
    '''
        Kernal for multiprocessing. Placed outside for further performance improvement.
    '''

    def idx_to_state(digit_base, num_dynamic_links, idx):
        return np.array(((idx // digit_base**np.arange(num_dynamic_links-1, -1, -1)) % digit_base))

    def state_to_idx(digit_base, digits: np.ndarray) -> int:
        return (digits * digit_base**np.arange(digits.shape[0]-1, -1, -1)).sum()

    ideces = np.arange(operation_idx*states_per_kernal, (operation_idx+1)*states_per_kernal)
    ideces = ideces[ideces < phys_dim]

    idx_row = [0]
    idx_column = [0]
    idx_data = [0]
    for idx in ideces:
        if idx%20==0:
            progress = (idx/phys_dim*100)
            print(f'         Progress: [{str((int(progress/5)*"|")+(int(20-progress/5)*" "))}]: %.0f%%' % (progress), end='\r')
        i_state = idx_to_state(digit_base, num_dynamic_links, idx)
        for site in sites.values():
            if not not site['plaquette']:
                f_state = copy.deepcopy(i_state)
                for u_op in site['plaquette']:
                    f_state[u_op['index']] += u_op['shift']
                
                # if all shifts dont exit truncation (if not one exits trunction), the plaquette operation makes a contribution
                if not (f_state - (f_state % digit_base)).any():
                    idx_row.append(idx)
                    idx_column.append(state_to_idx(digit_base, f_state))
                    idx_data.append(1.)

    return [idx_row, idx_column, idx_data]

def timeit(func):
    '''
        Debugging funciton for timing of methods/functions.
    '''
    def inner(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        stop = time.time()
        print(f'{func} took %.4f seconds.'%(stop-start))
        return result
    return inner


class Site:
    '''
        Store all attributes for a site.
    '''
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
    '''
        Store all attributes for a link.
    '''
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
        'dims': (2, 2),
        'pbc': False,
        'static_charges': dict(),
    }

    def __init__(self, config=dict()):
        '''
            Initiating Lattice with attributes and do first calculations of constant values.
        '''
        self.config = dict()
        for attr in self.default_config.keys():
            self.__setattr__(attr, config[attr] if attr in config else self.default_config[attr])
            self.config[attr] = config[attr] if attr in config else self.default_config[attr]

        self.dims = np.array(self.dims)
        self.projection_vector = np.array([self.dims[:dim_idx].prod() for dim_idx in range(len(self.dims))])
        self.digit_base = 2*self.l+1
        self.eigenvalues = dict()
        self.eigenvectors = dict()

        print(f'>>num of used threads: %d'%(multiprocessing.cpu_count()))


    def calculate_E_and_B_op(self):
        '''
            Do all calculations to the point of the construction of the electric (E) and magnetic (B) operator.
        '''
        self.build()
        self.add_static_charges()
        self.build_linear_system()
        self.solve_linear_system()
        self.build_total_electric_operator()
        self.build_plaquettes()
        self.build_total_magnetic_operator()

    def set_param(self, param, value):
        '''
            Set the parameter to a given value in the config dictionary and the class attribute.
        '''
        self.config[param] = value
        self.__setattr__(param, value)

    def dict_to_tuple(self, dictionary):
        '''
            Convert a dictionary to a tuple.
        '''
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
        return isinstance(other, Lattice) and self.config == other.config
        #return isinstance(other, Lattice) and self.__hash__() == other.__hash__()

    def __iter__(self):
        return iter(self.sites.values())

    def __getitem__(self, idx):
        return self.sites[idx]

    def __setitem__(self, idx, item):
        self.sites[idx] = item


    def idx_to_pos(self, idx: int) -> np.ndarray:
        '''
            Map the indices i of the sites to the position of the site.
        '''
        return np.array([(idx//(self.dims[:dim_idx].prod()))%self.dims[dim_idx] for dim_idx in range(len(self.dims))])

    def pos_to_idx(self, pos: np.ndarray) -> int:
        '''
            Map a position of a site to the corresponding site index i.
        '''
        return int(pos @ self.projection_vector)

    def build(self):
        '''
            Build the lattice for storage of the operators and computation
        '''
        print('>>building lattice..')
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
        self.add_static_charges()

    def add_static_charges(self) -> None:
        '''
            Add the static charges to the sites.
        '''
        for pos in self.static_charges.keys():
            self.sites[self.pos_to_idx(pos)].static_charge = self.static_charges[pos]
        
    def build_linear_system(self) -> None:
        '''
            Build the linear system of equations from the gauss law.
        '''
        if not hasattr(self, 'sites'):
            self.build()

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

        if not hasattr(self, 'linear_system'):
            self.build_linear_system()

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

        if not self.dynamic_links:
            self.solve_linear_system()

        print('>>building plaquettes..')  # build plaquettes
        for site in self:
            site['plaquette'] = list()
        for link in self.dynamic_links:
            idx = link.u
            pos = copy.deepcopy(self.sites[idx].pos)

            # If not hermitian conjugate, shift = 1. If hermitian conjugate, shift = -1.
            if link['direction'] == 0:
                self.sites[idx]['plaquette'].append({'index': self.dynamic_links.index(link), 'shift': 1})
                pos[1] = (pos[1]-1) % self.dims[1]
                self.sites[self.pos_to_idx(pos)]['plaquette'].append({'index': self.dynamic_links.index(link), 'shift': -1})
            elif link['direction'] == 1:
                self.sites[idx]['plaquette'].append({'index': self.dynamic_links.index(link), 'shift': -1})
                pos[0] = (pos[0]-1) % self.dims[0]
                self.sites[self.pos_to_idx(pos)]['plaquette'].append({'index': self.dynamic_links.index(link), 'shift': 1})

        
        # Remove periodic Plaquettes if the lattice has no PBC.
        if not self.pbc:
            print('>>removing periodic plaquettes..')
            for y in range(self.dims[1]):
                self.sites[self.pos_to_idx(np.array([self.dims[1]-1, y]))]['plaquette'] = list()
            for x in range(self.dims[0]):
                self.sites[self.pos_to_idx(np.array([x, self.dims[1]-1]))]['plaquette'] = list()

    def idx_to_state(self, idx):
        '''
            Map the state index i to a state with link states in a tensor product. The returned 
        '''
        return np.array(((idx // self.digit_base**np.arange(len(self.dynamic_links)-1, -1, -1)) % self.digit_base))

    def state_to_idx(self, digits: np.ndarray) -> int:
        '''
            Map state in a 
        '''
        return (digits * self.digit_base**np.arange(digits.shape[0]-1, -1, -1)).sum()

    def electric_kernal(self, idx):
        progress = (idx/self.phys_dim*100)
        if idx%20==0:
            print(f'         Progress: [{str((int(progress/5)*"|")+(int(20-progress/5)*" "))}]: %.0f%%' % (idx/self.phys_dim*100), end='\r')
        return self.E_OP.subs(zip(self.dynamic_e_ops, self.idx_to_state(idx) - self.l))

    @timeit
    def build_total_electric_operator(self) -> None:
        '''
            Build the electric operator of the lattice.
        '''

        if not hasattr(self, 'dynamic_links'):
            self.solve_linear_system()

        print('>>building electric hamiltonian..')  # build electric hamiltonian

        E = {'row': np.arange(self.phys_dim, dtype=np.int32),
               'column': np.arange(self.phys_dim, dtype=np.int32),
               'data':np.zeros(self.phys_dim)}

        self.E_OP = 0
        for equation in self.solution_list:
            self.E_OP += equation**2

        print('>>Mapping results..')
        tasks = E['row']
        with multiprocessing.Pool() as pool:
            result = pool.map(self.electric_kernal, tasks)
        
        '''

        result = list()
        for i, job_result in enumerate(pool_obj.imap_unordered(self.get_electric_element, tasks)):
            print('[%.0f%%]' % (i/len(tasks)*100), end='\r')
            result.append(job_result)
        '''

        print('>>Createing sparse matrix..')
        E['data'] = np.array(result, dtype=np.float32)
        self.E = scipy.sparse.csr_matrix((E['data'], (E['row'], E['column']))) / 2

                    
    @timeit
    def build_total_magnetic_operator(self) -> None:
        '''
            Build the plaquette operator of the lattice.
        '''
        if  not hasattr(next(self.__iter__()), 'plquette'):
            self.build_plaquettes()

        print('>>building magnetic hamiltonian..')  # build magnetic hamiltonian

        print('>>mapping results..')
        tasks = np.arange(math.ceil(self.phys_dim/states_per_kernal))
        with multiprocessing.Pool() as pool:
            result = pool.starmap(magnetic_kernal, [(self.sites, self.digit_base, len(self.dynamic_links), self.phys_dim, task) for task in tasks])


        '''

        result = list()
        for i, job_result in enumerate(pool_obj.imap(self.get_magnetic_vector, tasks)):
            print('[%.0f%%]' % (i/len(tasks)*100), end='\r')
            result.append(job_result)
        '''
        
        print('>>sorting results..')# sorting results
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

        print('>>createing sparse matrix..')
        P_sparse = scipy.sparse.csr_matrix((data, (row, column)), shape=(self.phys_dim, self.phys_dim))
        self.B = (P_sparse+P_sparse.T)/2

    @timeit
    def diagonalize_hamiltonian(self, g, k=1) -> None:
        '''
            Diagonalize the hamiltonian.
        '''
        if g in self.eigenvalues:
            return

        if not hasattr(self, 'E'):
            self.build_total_electric_operator()

        if not hasattr(self, 'B'):
            self.build_total_magnetic_operator()


        partial_diagonalisation = self.phys_dim > k and k > 0
        print(f'>>>diagonalizing Hamiltonian {f"to k=%d" % (k) if partial_diagonalisation else "entirely"} with g=%.4f..'%(g))
        
        self.H_sparse = g**2*self.E-self.B/g**2
    
        ncv = min(max(2*k+1, 20), self.phys_dim)
        if partial_diagonalisation:
            #eigval, eigvec  = scipy.sparse.linalg.eigsh(self.H_sparse, k=k, sigma=0.0, which='LM', ncv=ncv)
            eigval, eigvec  = scipy.sparse.linalg.eigsh(self.H_sparse, k=k, which='SA', ncv=ncv)
            #eigval, eigvec  = primme.eigsh(self.H_sparse, k=k, which='SA', ncv=ncv)
        else:
            eigval, eigvec = scipy.linalg.eigh(self.H_sparse.toarray())
            idx = eigval.argsort()
            eigval = eigval[idx]
            eigvec = eigvec[:, idx]

        self.eigenvalues[g] = eigval
        self.eigenvectors[g] = eigvec


    def printEV(self):
        for g in self.eigenvalues:
            print(f'\nEW for g=%.2f: ' % (g))
            for i in range(3):
                print(f'%.4f'%(self.eigenvalues[g][i]))

    def get_plaquette_expectation_values(self, beta_list):
        print('>>calculating plqauette expection values..')
        g_list = 1/beta_list**(1/2)
        expectation_values = np.zeros(len(beta_list))

        num_plaquettes = (self.dims).prod() if self.pbc else (self.dims-1).prod()
        for i, g in enumerate(g_list):
            if g not in self.eigenvectors:
                self.diagonalize_hamiltonian(g, k=1)

            ground_state = self.eigenvectors[g][:, 0]

            expectation_values[i] = ground_state.T @ self.B @ ground_state / num_plaquettes
        return expectation_values

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
        print(f'WARNING: Path {path} not existant. Returning empty dict!')
        return dict()

def dprint(*args):
    print('debug:', *args)
