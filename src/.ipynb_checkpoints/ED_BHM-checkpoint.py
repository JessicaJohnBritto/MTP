import numpy as np
from numpy import linalg
from scipy import linalg as splinalg
import matplotlib.pyplot as plt
from scipy import sparse as sp
import scipy.sparse.linalg
from functools import reduce
import itertools
from scipy import linalg
from scipy.linalg import expm, logm
from scipy.special import comb
from itertools import combinations_with_replacement, product
from collections import Counter
import copy
from scipy.linalg import ishermitian

params = {
    'N': 0, # no. of sites
    't':0, # Hopping Amplitude
    'T': 0, # Total Time Evolution
    'tau': 0 # Time step
}

params['dim'], params['truncationParam_n'] = 0, 0
params['occupationTuple_lst'] = []
params['Map_ParticleLoc_OccupNo']={}
params['map_occupToD_Dim'] = {}

def dimension(params, **kwargs):
    '''
    For k identical bosonic particles on N lattice site, 
    Changes the dim and initializes the vac state
    based on dim.
    Makes change to the params - 'dim', 'vac' within the function.
    Make sure to define the parameters list as params.
    Return: Null
    '''
    N, k, dim = params['N'], params['k'], int(params['dim']) # N = number of sites, k = no. of identical bosonic particles
    params['dim'] = int(comb(N+k-1, k)) # This is for k identical bosonic particles
    params['vac'] = sp.csc_matrix(np.zeros(N))
    pass


def normalizeWF(psi,**kwargs):
    '''
    Return a Normalized Wavefunction
    '''
    shape, dtype = psi.shape, psi.dtype
    NWF = psi
    if np.array_equal(psi, np.zeros(shape, dtype = dtype)) == True:
        NWF = psi
    elif np.vdot(psi, psi) == 0:
        NWF = psi
    else:
        NWF = psi/(np.sqrt(np.vdot(psi, psi)))
    return NWF

def creationOpMatrix(params, **kwargs):
    '''
    Returns Bosonic Creation Operator
    '''
    A = sp.diags(np.sqrt(np.arange(1, params['truncationParam_n']+1, 1)), -1)
    return A

def annihilationOpMatrix(params, **kwargs):
    '''
    Returns Bosonic Annihilation Operator
    '''
    A = sp.diags(np.sqrt(np.arange(1, params['truncationParam_n']+1, 1)), 1)
    return A

def numOpMatrix(params, site_no, **kwargs):
    '''
    Returns Bosonic Number Operator
    '''
    createOp = creationOpMatrix(params)
    nOp = createOp@createOp.transpose()
    I = sp.identity(params['truncationParam_n']+1)
    lst = [I for _ in range(params['N'])]
    if 'tilt' in kwargs:
        lst[site_no] = nOp*site_no
        matrx = sp.csc_matrix(reduce(sp.kron, lst))
        return matrx
    else: 
        lst[site_no] = nOp
        matrix1 = sp.csc_matrix(reduce(sp.kron, lst))
        lst[site_no] = nOp@nOp
        matrix2 = sp.csc_matrix(reduce(sp.kron, lst))
        return matrix1, matrix2

def HoppingOpMatrix(params, site_no, **kwargs):
    '''
    Returns Bosonic Hopping Operator
    '''
    n, N = params['truncationParam_n'], params['N']
    matrixx = sp.csc_matrix(((n+1)**N, (n+1)**N))
    if site_no != params['N']-1:
        creationOp, annihOp = creationOpMatrix(params), annihilationOpMatrix(params)
        I = sp.identity(params['truncationParam_n']+1)
        lst = [I for _ in range(params['N'])]
        lst[site_no], lst[site_no+1] = creationOp, annihOp
        matrixx = sp.csc_matrix(reduce(sp.kron, lst))
    return matrixx

def Prod_OccupBasis(params, **kwargs):
    '''
    Generates all combinations using product from itertools.
    Returns: valid_combinations under the k-constraint (particle
    number conservation) and all combinations.
    '''
    n, N, k = params['truncationParam_n']+1, params['N'], params['k']
    all_combinations = dict(enumerate(itertools.product(range(n), repeat=N)))
    valid_combinations = dict(filter(lambda x: sum(x[1]) == k, all_combinations.items()))
    return valid_combinations, all_combinations
params['truncationParam_n'], params['N'], params['k'] = 2, 2, 2

def projectionMatrix(params, **kwargs):
    '''
    Creates a projection matrix whose elements are non-zero
    for the indices of the occup_states obeying k-constraint.
    '''
    valid_combinations, all_combinations = Prod_OccupBasis(params)
    rows, cols = len(valid_combinations), len(all_combinations)
    PM = sp.csc_matrix((rows, cols))
    for i, key in enumerate(list(reversed(valid_combinations.keys()))):
        PM[i, key] = 1.0
    return PM

def HamiltonianMatrix(params, **kwargs):
    '''
    Returns BHM Hamiltonian Matrix and ground state
    '''
    n, N, k = params['truncationParam_n'], params['N'], params['k']
    H = sp.csc_matrix(((n+1)**N, (n+1)**N))
    PM = projectionMatrix(params)
    for i in range(N):
        HopOp, nOp_mu = HoppingOpMatrix(params, i), numOpMatrix(params, i, tilt = True)
        NumOp, NumOp_2 = numOpMatrix(params, i)
        H += -params['t']*(HopOp+HopOp.transpose()) + 0.5*params['U']*(NumOp_2 - NumOp) - params['mu']*nOp_mu
    H = PM@H@PM.transpose()
    eigenval, eigenvec = sp.linalg.eigsh(H, k=1, which='SA')
    return H, eigenvec

def density_reduced_dmatrix(params, eigenvec, **kwargs):
    '''
    Based on Multidimensional Numpy Arrays, therefore, works upto 4 sites, 4 particles
    Generates density matrix as well.
    Returns Reduced_density matrix of the left subsystem of the lattice.
    '''
    S, entropy = 0,[]
    valid_combinations, total_comb = Prod_OccupBasis(params)
    nn = params['truncationParam_n']+1
    particle_loc_list, states_lists, probamp_lists = [list(value) for value in valid_combinations.values()], [], []
    params['states_vec'] = np.array(particle_loc_list)
    center_index = params['N']//2
    ## Right-side has larger no. of Lattice sites. For eg. 5 lattice sites = 3R + 2L
    nL = len(list(itertools.product(list(range(nn)), repeat=(center_index))))
    nR = len(list(itertools.product(list(range(nn)), repeat=params['N']-(center_index))))

    Rho = sp.kron(eigenvec, eigenvec.transpose())
    reverse_comb = dict(reversed(list(valid_combinations.items())))
    total_psi = sp.csc_matrix(((eigenvec.transpose()).tolist()[0], (list(reverse_comb.keys()), [0] * len(list(reverse_comb.keys())))), shape=(len(total_comb), 1))
    dm_total_psi = sp.kron(total_psi, total_psi.transpose())
    dm_total_psi_reshape = np.reshape(dm_total_psi.toarray(), (nL, nR, nL, nR))
    # rho_reduced = np.zeros((nL, nL), dtype=complex)
    # for nR1 in range(nR):
    #     rho_reduced += dm_total_psi_reshape[:, nR1, :, nR1]
    rho_reduced = np.tensordot(dm_total_psi_reshape, np.eye(nR), axes=([1, 3], [0, 1]))
    # print(ishermitian(rho_reduced))
    # print(rho_reduced.trace())
    
    return rho_reduced

def plot_Bipartite_EntanglementEntropyVsJ(params, **kwargs):
    '''
    Returns plot of Bipartite Entanglement Entropy vs J for a constant U.
    '''
    entropy = []
    for t in params['hop_list']:
        S = 0
        params['t'] = t
        _, eigenvec = HamiltonianMatrix(params)
        ## For Sparse Matrices
        # trace_right_sites = density_reduced_dmatrix(params,eigenvec)
        # eigval_small, _ = sp.linalg.eigsh(trace_right_sites, k=trace_right_sites.shape[0]-1, which='SA')
        # eigL, _ = sp.linalg.eigsh(trace_right_sites, k=1, which='LA')
        # for e in np.concatenate((eigval_small, eigL)):
        #     if e>0:
        #         S+= (np.power(np.abs(e),2))*np.log(np.power(np.abs(e),2))
        
        ## For numpy arrays
        trace_right_sites = density_reduced_dmatrix(params,eigenvec)
        eig, _ = np.linalg.eig(trace_right_sites)
        for e in eig:
            if e>0:
                S+= e*np.log(e)
        
        entropy+=[-S]
    # print(entropy)
    plt.plot(params['hop_list'], entropy)
    plt.xlabel('J')
    plt.ylabel('Entanglement Entropy')
    plt.title('Entanglement Entropy between two parts of the Lattice')
    plt.show()
        ## The lines of code below are used to visualize the Reduced Density Matrix for each t.
        # fig, axs = plt.subplots(1, figsize=(14, 6))
        # im = axs.imshow(trace_right_sites, cmap="Blues")
        # # axs.set_xlabel('Site i')
        # # axs.set_ylabel('Site j')
        # axs.set_title(f'Bipartite Entanglement at t = {t:.1f}, U={params["U"]}')
        # fig.colorbar(im, ax=axs, label = 'Correlation between two halves of the lattice')
    pass


def plot_first_ExcitedGapVsJ(params, **kwargs):
    '''
    Returns plot of first excited gap as a function of J for constant U and mu.
    '''
    gap = []
    for t in params['hop_list']:
        params['t'] = t
        H, _ = HamiltonianMatrix(params)
        eigenval, eigenvec = sp.linalg.eigsh(H, k=2, which='SA')
        gap+=[eigenval[1]-eigenval[0]]
    # print(gap)
    plt.plot(params['hop_list'], gap)
    plt.xlabel('J')
    plt.ylabel('First Excitation Gap')
    U = params['U']
    plt.title(f'First excited gap as a function of J for U={U}')
    plt.grid(True)
    plt.show
    pass

def sparse_plotwithColorBar(params, mat, **kwargs):
    '''
    Params: mat - is the sparse matrix.
    Returns scatter plot with a colorbar for visualizing
    sparse matrices. 
    '''
    fig,ax = plt.subplots(figsize=(8, 5), dpi= 80, facecolor='w', edgecolor='k')
    plot_list = []
    nonzero_indices = mat.nonzero()
    plot_list = np.array([[col, row, mat[row, col]] for row, col in zip(*nonzero_indices)])
    plot_list = np.array(plot_list)
    plt.scatter(plot_list[:,0],plot_list[:,1],c=plot_list[:,2], s=50)
    cb = plt.colorbar()
    plt.xlim(-1,mat.shape[1])
    plt.ylim(-1,mat.shape[0])
    plt.gca().invert_yaxis()

def plot_std_numOpVSJ_overU(params, **kwargs):
    '''
    Returns plot of standard deviation of numberOp vs J/U
    '''
    PM, std_val, std = projectionMatrix(params), 0, []
    for t in params['hop_list']:
        params['t'] = t
        _, eigenvec = HamiltonianMatrix(params)
        for i in range(params['N']):
            NumOp, NumOp_2 = numOpMatrix(params, i)
            nOp, nOp_2 = PM@NumOp@PM.transpose(), PM@NumOp_2@PM.transpose()
            expval_nOp, expval_nOp_2 = np.vdot(eigenvec, nOp@eigenvec), np.vdot(eigenvec, nOp_2@eigenvec)
            std_val += expval_nOp_2 - expval_nOp*expval_nOp
        std+=[np.mean(std_val)]
        std_val = 0
    plt.plot(params['hop_list'], std)
    plt.ylabel('Fluctuations in Number Operator')
    plt.xlabel('J')
    mu, U = params['mu'], params['U']
    # plt.title(f'For U={U} and μ={mu}\nFluctuations in Number Operator vs J')
    plt.title(f'Fluctuations in Number Operator vs J for U={U}')
    plt.show()


def plot_ExpValHoppVsJ_overU(params, **kwargs):
    '''
    Returns plots of Expectation Value of Hopping Ops vs J/U,
    Correlation between particle at the center of lattice and
    the remaining sites.
    '''
    I, annihOp = sp.identity(params['truncationParam_n']+1), annihilationOpMatrix(params)
    PM, createOp, lst = projectionMatrix(params), creationOpMatrix(params), [I for _ in range(params['N'])]
    for t in params['hop_list']:
        params['t'] = t
        H = np.zeros((params['N'], params['N']))
        _, eigenvec = HamiltonianMatrix(params)
        for i in range(params['N']):
            for j in range(params['N']):
                if i == j:
                    lst[i] = createOp@annihOp
                else:
                    lst[i] = createOp
                    lst[j] = annihOp
                HopOp = PM@reduce(sp.kron, lst)@PM.transpose()
                H[i, j] =  np.vdot(eigenvec, HopOp@eigenvec)
                lst[i], lst[j] = I, I
        ## These are used for large lattice sites for which  
        ## sparse matrices are needed.
        ## Use it to visualize sparse matrix without a colorbar.
        # plt.spy(H)
        ## Use it to visualize sparse matrix using a scatterplot with a colorbar
        # sparse_plotwithColorBar(params, mat = H)
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))

        im = axs[0].imshow(H, cmap="Blues")
        axs[0].set_xlabel('Site i')
        axs[0].set_ylabel('Site j')
        axs[0].set_title(f'$\\langle b_i^\\dagger b_j \\rangle$ Matrix at t = {t:.1f}, U={params["U"]}')
        fig.colorbar(im, ax=axs[0])

        # Second subplot for the correlation plot
        middle_index = (params['N'] - 1) // 2
        correlation_rate = H[middle_index, :]
        axs[1].plot(list(range(params['N'])), correlation_rate)
        axs[1].set_xlabel('Site i')
        axs[1].set_ylabel('Correlation')
        axs[1].set_title(f'Correlation between particle at site i={middle_index} and remaining sites')
        axs[1].grid(True)
        plt.show()
    pass

