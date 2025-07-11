from lattice import *


def potential():
    lattices = load()
    static_charges = [{},
                      {(0, 0): 1, (0, 1): -1},
                      {(0, 0): 1, (1, 1): -1},
                      {(0, 0): 1, (0, 2): -1},
                      {(0, 0): 1, (1, 2): -1},
                      {(0, 0): 1, (2, 2): -1},
                      {(0, 0): 1, (0, 3): -1},
                      {(0, 0): 1, (1, 3): -1},
                      {(0, 0): 1, (2, 3): -1},
                      {(0, 0): 1, (3, 3): -1}]
    #betas = (np.arange(10)*0.2+0.8)
    #betas = np.array([1])
    betas = np.linspace(1.4, 80, 5)

    config = {'k': 3, 'dims': (4, 4), 'pbc': False, 'l':1}
    config = {'k': 3, 'dims': (4, 4), 'pbc': False, 'l':1}
    lattice = Lattice(config=config)
    calc_con = False
    for j in range(betas.shape[0]):
        lattice.set_param('g', 1/betas[j]**(1/2))
        if lattice.__hash__() not in lattices:
            calc_con = True
            break

    if calc_con:
        lattice.calculate_E_and_B_op()
        for j in range(betas.shape[0]):
            lattice.set_param('g', 1/betas[j]**(1/2))
            if lattice.__hash__() not in lattices:
                lattice.diagonalize_hamiltonian()
                lattices[lattice.__hash__()] = copy.deepcopy(lattice)


    for i, static_charge in enumerate(static_charges):
        config['static_charges'] = static_charge
        lattice = Lattice(config=config)
        calc_con = False
        for j, b in enumerate(betas):
            lattice.set_param('g', 1/b**(1/2))
            if lattice.__hash__() not in lattices:
                calc_con = True
                break

        if calc_con:
            lattice.calculate_E_and_B_op()
            for j, b in enumerate(betas):
                lattice.set_param('g', 1/b**(1/2))
                if lattice.__hash__() not in lattices:
                    lattice.diagonalize_hamiltonian()
                    lattices[lattice.__hash__()] = copy.deepcopy(lattice)

    save(lattices)



def one_lattice():
    lattices = load()
    lattice = Lattice(config={'k': 6, 'dims': (3, 3), 'pbc': True, 'l':2, 'g':1})
    #if lattice.__hash__() not in lattices:
    lattice.calculate_E_and_B_op()
    lattice.diagonalize_hamiltonian()
    lattice.printEW()
    lattices[lattice.__hash__()] = copy.deepcopy(lattice)
    save(lattices)

def plaquette_exp():
    pass
    '''

    lattices = load()
    configs = [
        #{'dims': (2, 2), 'pbc': False, 'log':True, 'max_l':8},
        #{'dims': (2, 2), 'pbc': True, 'log':True, 'max_l':4},
        #{'dims': (3, 3), 'pbc': False, 'log':True, 'max_l':5},
        #{'dims': (3, 3), 'pbc': True, 'log':True, 'max_l':1},
        #{'dims': (2, 2), 'pbc': False, 'log':False, 'max_l':8},
        #{'dims': (2, 2), 'pbc': True, 'log':False, 'max_l':4},
        #{'dims': (3, 3), 'pbc': False, 'log':False, 'max_l':5},
        {'dims': (3, 3), 'pbc': True, 'log':False, 'max_l':1},
    ]
    for i in range(len(configs)):
        plt.clf()
        if configs[i]['log']:
            betas = 10**(np.linspace(-2, 2, 20))
        else: 
            betas = np.arange(10)*0.2+0.8
        for l in np.arange(configs[i]['max_l'])+1:
            print(f'\n------------- Truncating to l=%d -------------'%(l))
            configs[i]['l'] = l
            lattice = Lattice(config=configs[i])
            p_exp = lattice.get_plaquette_expectation_values(betas)
            plt.plot(betas, p_exp, marker='v', label=f'l=%d'%(l), linestyle='dashed')
            print(betas, p_exp)

        plt.ylabel('$\Braket{P}$')
        plt.xlabel(r'$\beta=1/g^2$')
        plt.tight_layout()
        plt.grid()
        if configs[i]['log']:
            plt.xscale('log')
        plt.legend()
        string = f'{configs[i]["dims"][0]}x{configs[i]["dims"][1]}'
        string += 'PBC' if configs[i]['pbc'] else ''
        string += ('Log' if configs[i]['log'] else '')
        plt.savefig(f'../latex/images/PlaquetteExp{string}.pdf')


    '''

if __name__ == '__main__':
    one_lattice()
    '''
    for j in range(e_pots.shape[0]):
        plt.scatter(distances, e_pots[j], marker='x', label=f'$\\beta=%.2f$'%(betas[j]))
    plt.grid()
    plt.legend()
    plt.xlabel(r'r')
    plt.ylabel(r'$V$')
    plt.tight_layout()
    plt.savefig(f'../latex/images/quark-antiquark-potential2.pdf')
    plt.clf()
    '''
