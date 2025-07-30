from lattice import *
import matplotlib.pyplot as plt
import matplotlib as mpl


f_size = 20
mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": [],                   # blank entries should cause plots
    "font.sans-serif": [],              # to inherit fonts from the document
    "font.monospace": [],
    "axes.labelsize": f_size,               # LaTeX default is 10pt font.
    "font.size": f_size,
    "legend.fontsize": f_size,               # Make the legend/label fonts
    "xtick.labelsize": f_size,               # a little smaller
    "ytick.labelsize": f_size,
    "text.latex.preamble": r"""
        \usepackage{amsmath}
        \usepackage{siunitx}
        \usepackage{physics}
        \usepackage{amsfonts}
        \usepackage{amsmath}
        \usepackage{amssymb}
        \usepackage{braket}
    """
})

def potential(g):
    plt.clf()
    plt.figure(figsize=(8, 8))
    lattices = load()
    static_charges = [
                      {(0, 0): 1, (0, 1): -1},
                      {(0, 0): 1, (1, 1): -1},
                      {(0, 0): 1, (0, 2): -1},
                      {(0, 0): 1, (1, 2): -1},
                      {(0, 0): 1, (2, 2): -1},
                      #{(0, 0): 1, (0, 3): -1},
                      #{(0, 0): 1, (1, 3): -1},
                      #{(0, 0): 1, (2, 3): -1},
                      #{(0, 0): 1, (3, 3): -1},
    ]

    offset_static_charges = [
        {(1, 0): 1, (1, 1): -1}
    ]

    l_list = [1, 2, 3]

    config = {'dims': (3, 3), 'pbc': False}

    def check_lat(config, g):
        #print(config)
        lattice = Lattice(config=config)
        if lattice.__hash__() not in lattices:
            lattice.diagonalize_hamiltonian(g,k=1)
            lattices[lattice.__hash__()] = lattice
        else:
            lattice = lattices[lattice.__hash__()]
            if g not in lattice.eigenvalues:
                lattice.diagonalize_hamiltonian(g,k=1)

    # pot for different truncations
    for i, l in enumerate(l_list):
        config['l'] = l
        config.pop('static_charges', None)
        dprint(config)
        check_lat(config, g)
        
        
        for charge in static_charges:
            config['static_charges'] = charge
            check_lat(config, g)

    # pot for offset charge pair
    config = {'dims': (3, 3), 'pbc': False}
    config['l'] = 1
    check_lat(config, g)

    for charge in offset_static_charges:
        config['static_charges'] = charge
        check_lat(config, g)

    # pot for pbc
    pbc_static_charges = [
        {(1, 1): 1, (1, 2): -1},
        {(1, 1): 1, (2, 2): -1},
    ]
    config = {'dims': (3, 3), 'pbc': True}
    check_lat(config, g)

    for charge in pbc_static_charges:
        config['static_charges'] = charge
        check_lat(config, g)

    # pot for 4x4 lat
    x4_static_charges_offset = [
        {(1, 1): 1, (1, 2): -1},
        {(1, 1): 1, (2, 2): -1},
    ]
    x4_static_charges = [
        {(0, 0): 1, (0, 1): -1},
        {(0, 0): 1, (1, 1): -1},
        {(0, 0): 1, (2, 0): -1},
        {(0, 0): 1, (2, 1): -1},
        {(0, 0): 1, (2, 2): -1},
        {(0, 0): 1, (3, 0): -1},
        {(0, 0): 1, (3, 1): -1},
        {(0, 0): 1, (3, 2): -1},
    ]
    config = {'dims': (4, 4),'pbc':False, 'l':1}
    check_lat(config, g)

    for charge in x4_static_charges:
        config['static_charges'] = charge
        check_lat(config, g)
        save(lattices)

    for charge in x4_static_charges_offset:
        config['static_charges'] = charge
        check_lat(config, g)


    save(lattices)

    style = [
        {'marker':'v', 'linestyle':''},
        {'marker':'v', 'linestyle':''},
        {'marker':'.', 'linestyle':'dashed'},
             ]
    marker_size = 10

    config = {'dims': (3, 3), 'pbc': False}
    for i, l in enumerate(l_list):
        config['l'] = l
        config.pop('static_charges', None)
        print(config)
        lattice0 = Lattice(config=config)
        e0_nocharge = lattices[lattice0.__hash__()].eigenvalues[g][0]
        
        e0_wcharge = np.zeros(len(static_charges))
        distance = np.zeros(len(static_charges))
        for j, charge in enumerate(static_charges):
            config['static_charges'] = charge
            lattice = Lattice(config)
            e0_wcharge[j] = lattices[lattice.__hash__()].eigenvalues[g][0]
            distance[j] = (np.array(list(charge.items())[1][0])**2).sum()**(1/2)
        
        #print(e0_nocharge, e0_wcharge)
        plt.plot(distance, e0_wcharge - e0_nocharge, label=f'$l=%d$'%(l), **(style[i]), markersize=marker_size)
    
    config = {'dims': (3, 3), 'pbc': False}
    config['l'] = 1
    lattice0 = Lattice(config=config)
    e0_nocharge = lattices[lattice0.__hash__()].eigenvalues[g][0]

    e0_wcharge = np.zeros(len(offset_static_charges))
    distance = np.zeros(len(offset_static_charges))
    for i, charge in enumerate(offset_static_charges):
        config['static_charges'] = charge
        lattice = Lattice(config=config)
        lattice = lattices[lattice.__hash__()]
        e0_wcharge[i] = lattices[lattice.__hash__()].eigenvalues[g][0]
        charge_list = list(charge.items())
        distance[i] = ((np.array(charge_list[1][0])-np.array(charge_list[0][0]))**2).sum()**(1/2)
    plt.plot(distance, e0_wcharge - e0_nocharge, label=f'$l=1$, offset', marker='x', linestyle='', c='C0', markersize=15)
   
    config = {'dims': (3, 3), 'pbc': True}
    lattice0 = Lattice(config=config)
    e0_nocharge = lattices[lattice0.__hash__()].eigenvalues[g][0]

    e0_wcharge = np.zeros(len(pbc_static_charges))
    distance = np.zeros(len(pbc_static_charges))
    for i, charge in enumerate(pbc_static_charges):
        config['static_charges'] = charge
        lattice = Lattice(config=config)
        lattice = lattices[lattice.__hash__()]
        e0_wcharge[i] = lattices[lattice.__hash__()].eigenvalues[g][0]
        charge_list = list(charge.items())
        distance[i] = ((np.array(charge_list[1][0])-np.array(charge_list[0][0]))**2).sum()**(1/2)
    plt.plot(distance, e0_wcharge - e0_nocharge, label=f'$l=1$, pbc', marker='o', linestyle='', c='C0', markersize=marker_size)

    config = {'dims': (4, 4),'pbc':False, 'l':1}
    lattice0 = Lattice(config=config)
    e0_nocharge = lattices[lattice0.__hash__()].eigenvalues[g][0]

    e0_wcharge = np.zeros(len(x4_static_charges))
    distance = np.zeros(len(x4_static_charges))
    for i, charge in enumerate(x4_static_charges):
        config['static_charges'] = charge
        lattice = Lattice(config=config)
        lattice = lattices[lattice.__hash__()]
        e0_wcharge[i] = lattices[lattice.__hash__()].eigenvalues[g][0]
        charge_list = list(charge.items())
        distance[i] = ((np.array(charge_list[1][0])-np.array(charge_list[0][0]))**2).sum()**(1/2)
    plt.plot(distance, e0_wcharge - e0_nocharge, label=f'$l=1$, 4x4', marker='.', linestyle='dashed', c='C0', markersize=marker_size)

    e0_wcharge = np.zeros(len(x4_static_charges_offset))
    distance = np.zeros(len(x4_static_charges_offset))
    for i, charge in enumerate(x4_static_charges_offset):
        config['static_charges'] = charge
        lattice = Lattice(config=config)
        lattice = lattices[lattice.__hash__()]
        e0_wcharge[i] = lattices[lattice.__hash__()].eigenvalues[g][0]
        charge_list = list(charge.items())
        distance[i] = ((np.array(charge_list[1][0])-np.array(charge_list[0][0]))**2).sum()**(1/2)
    plt.plot(distance, e0_wcharge - e0_nocharge, label=f'$l=1$, 4x4, offset', marker='.', linestyle='dashed', c='red', markersize=marker_size)





    plt.xlabel('$r/a$')
    plt.ylabel('$aV(r)$')
    plt.grid()
    #plt.tight_layout()
    plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=2, frameon=True, 
               edgecolor='black', fancybox=False)
    gsize = ''

    print(g)

    if g == 1.:
        gsize = 'normal'
        #plt.ylim((-1, 2))
    elif g > 1.:
        gsize = 'large'
        #plt.ylim((-2.5, 2.5))
    elif g < 1:
        gsize = 'small'
    print(gsize)
    plt.savefig(f'../latex/images/quark_antiquark_potential_g_{g}.pdf', bbox_inches='tight')



def one_lattice():
    #lattices = load()
    lattice = Lattice(config={'dims': (3, 3), 'pbc': True, 'l':1})
    lattice.diagonalize_hamiltonian(2)
    lattice.diagonalize_hamiltonian(1)
    #lattice.printEV()
    #lattices[lattice.__hash__()] = copy.deepcopy(lattice)
    #save(lattices)

def plaquette_exp():
    #lattices = load('../data/hpc/lattices.pickle')

    plt.figure(figsize=(8, 6))

    '''

    configs = [
        {'dims': (2, 2), 'pbc': True, 'l':1},
        {'dims': (2, 2), 'pbc': True, 'l':3},
        {'dims': (2, 2), 'pbc': True, 'l':7},
    ]
    '''
    configs = [
        {'dims': (3, 3), 'pbc': True, 'l':1},
        {'dims': (3, 3), 'pbc': True, 'l':2},
        {'dims': (3, 3), 'pbc': True, 'l':3},
    ]
    log = False
    '''
    lattices = load()

    l_list = np.arange(3)+1
    if log:
        betas = 10**(np.linspace(-2, 2, 20))
    else:
        betas = np.arange(10)*0.2+0.8

    exp_list = np.zeros((len(configs), len(betas)))
    for i, config in enumerate(configs):
        lattice = Lattice(config)
        if lattice.__hash__() in lattices:
            print('lattice existing')
            lattice = lattices[lattice.__hash__()]
            
        exp_list[i] = lattice.get_plaquette_expectation_values(betas)
        lattices[lattice.__hash__()] = copy.deepcopy(lattice)
        save(lattices)
    print(betas, exp_list)
    save(lattices)

    for i, exp in enumerate(exp_list):
        plt.plot(betas, exp, marker='v', label=f'$l=%d$'%(configs[i]['l']), linestyle='dashed', markersize=7)
    
    #exp = 

    #plt.plot(betas, exp, marker='v', label=f'$l=3$', linestyle='dashed')
    '''

    
    plt.ylabel('$\Braket{P}$')
    plt.xlabel(r'$\beta=1/g^2$')
    if log:
        plt.xscale('log')
    #plt.tight_layout()
    x = np.array([0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6])

    y = np.array([
        [0.2784131,  0.3793732,  0.45965064, 0.50891304, 0.53400858, 0.54694496, 0.55426179, 0.55878894, 0.56178858, 0.56387754],
        [0.30204604, 0.44248772, 0.56788021, 0.6442727,  0.68666457, 0.71301778, 0.73088793, 0.74359415, 0.75290069, 0.75987167],
        [0.30225266, 0.44394352, 0.57305357, 0.65511315, 0.70443844, 0.73841885, 0.76394733, 0.78385305, 0.79963463, 0.81226427]
    ])
    for i, ydata in enumerate(y):
        plt.plot(x, ydata, marker='v', label=f'$l={i+1}$', linestyle='dashed')
    plt.grid()
    plt.legend(bbox_to_anchor=(0.012, 0.98, 0.25, 0.0), loc="upper left",
                mode="expand", borderaxespad=0, ncol=1, frameon=True, 
               edgecolor='black', fancybox=False)
    string = f'{configs[0]["dims"][0]}x{configs[0]["dims"][1]}'
    string += 'PBC' if configs[0]['pbc'] else ''
    plt.savefig(f'../latex/images/PlaquetteExp{string}.pdf', bbox_inches='tight')


if __name__ == '__main__':
    plaquette_exp()
    #one_lattice()

    #potential(10)
    potential(5)
    potential(1)
    potential(0.5)
    potential(0.1)
    #potential(1e-7)
    #potential(1e-10)
    #potential(1e-20)
    #potential(1e-30)
