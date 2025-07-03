from lattice import *

if __name__ == '__main__':
    '''
    EWs = np.zeros(10)
    #for i in range(10):
    #time.sleep(1)
    lattices = dict()
    lattice = Lattice(config={'k': 6, 'dims': (2, 2), 'pbc': True, 'l':1, 'g':1})
    lattice.diagonalize_hamiltonian()
    lattices[lattice.config: lattice]
    save(lattices)

    '''

    '''
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

    distances = np.zeros((len(static_charges)))
    for i, charge in enumerate(static_charges):
        if not charge:
            distances[i] = 0
        else:
            position = tuple(charge.items())[1][0]
            distances[i] = ((np.array(position))**2).sum()**(1/2)

    e_pots = np.zeros((len(betas), len(static_charges)))
    for j, beta in enumerate(betas):
        for i, charges in enumerate(static_charges):
            config['g'] = 1/beta**(1/2)
            config['static_charges'] = charges
            e_pots[j, i] = (lattices[Lattice(config=config).__hash__()].EW[0])

    rrF = distances[1:]**2*(e_pots-np.roll(e_pots, 1, axis=1))[:, 1:]/(distances-np.roll(distances, 1))[1:]

    #for i, b in enumerate(betas):
        #plt.scatter(distances[1:], rrF[i, :], label=f'$\\beta=%.2f$'%(b))
    
    print(distances)
    for i in np.arange(len(distances))[[1, 4]]:
        plt.scatter(betas, rrF[:, i], label=f'r=%.2f'%(distances[i]))


    plt.grid()
    plt.legend()
    plt.xlabel('$\\beta$')
    plt.ylabel(r'$r^2F$')
    plt.tight_layout()
    plt.savefig(f'../latex/images/step_scaling.pdf')


    '''
    '''
