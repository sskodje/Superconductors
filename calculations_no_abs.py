# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 18:20:23 2021

@author: Ting42
"""
#%%
import numpy as np
from scipy.optimize import fsolve
from scipy.optimize import dual_annealing
from numba import jit

@jit(nopython = True, nogil = True)
def calculate_delta_pluss_minus(delta, diag_coupling, off_diag_coupling, epsilon_pluss, epsilon_minus):
    """
    Calculates the error resulting from wrong choice of order parameters based on the current guess

    Parameters
    ----------
    delta : List
        List with the two order parameters for the different bands
    diag_coupling : Float
        Coupling between diagonal elements (Cooper pair coupling strength).
    off_diag_coupling : Float
        Coupling between off diagonal elements (Josephson coupling strength)
    epsilon : Array of floats
        
    Returns
    -------
    results : dict
        Dictionary of results from scipy annealing

    """ 
    delta_pluss = (delta[0])
    delta_minus = (delta[1])
    
    E_pluss = np.sqrt(epsilon_pluss**2 + delta_pluss**2)
    E_minus = np.sqrt(epsilon_minus**2 + delta_minus**2)
    
    pluss_factor = diag_coupling / 2 * np.sum(delta_pluss / E_pluss) + off_diag_coupling / 2 * np.sum(delta_minus / E_minus)
    
    minus_factor = off_diag_coupling / 2 * np.sum(delta_pluss / E_pluss) + diag_coupling / 2 * np.sum(delta_minus / E_minus)
    
    error = np.abs(delta_pluss - pluss_factor) + np.abs(delta_minus - minus_factor)
    
    return error

def anneal_delta_pluss_minus(diag_coupling, off_diag_coupling, epsilon_pluss, epsilon_minus):
    """
    Calculates the two order parameters using simulated annealing

    Parameters
    ----------
    diag_coupling : Float
        Coupling between diagonal elements (Cooper pair coupling strength).
    off_diag_coupling : Float
        Coupling between off diagonal elements (Josephson coupling strength)
    epsilon : Array of floats
        
    Returns
    -------
    results : dict
        Dictionary of results from scipy annealing

    """    
    min_lim = 1e-5
    parameter_space_min = [1e-5] * 2
    parameter_space_min = [-1e-3] * 2
    parameter_space_max = [1e-3] * 2
    parameter_limits = list(zip(parameter_space_min, parameter_space_max))
    
    results = dual_annealing(calculate_delta_pluss_minus, bounds = parameter_limits, args=(diag_coupling, off_diag_coupling, epsilon_pluss, epsilon_minus), maxiter= 2700, initial_temp=12000)
    # while True:    
    #     if results["x"][0] or results["x"][1] == min_lim:
    #         results = dual_annealing(calculate_delta_pluss_minus, bounds = parameter_limits, args=(diag_coupling, off_diag_coupling, epsilon_pluss, epsilon_minus), maxiter= 2500)
    #     else:
    #         break  
    return results

@jit(nopython=True)
def calculate_epsilon(kx,ky, t, band, spinorbit):
    """
    Calculates the kinetic energy of particles with pseudospinn + (-) on a square lattice

    Parameters
    ----------
    kx : Array of floats
        Selected momentum values on a grid.
    ky : Array of floats
        Selected momentum values on a grid.
    t : Float
        Kinetic coupling strength.
    band : String
        Indicates band pluss (minus)
    spinorbit : Float
        Strength of spin-orbit interaction
    Returns
    -------
    Array of floats
        Kinetic energy for the given momentum values and band.

    """
    mu = -t

    epsilon_tilde = - 2 * t * (np.cos(kx) + np.cos(ky)) - mu 
    B_k = np.sqrt((np.sin(kx))**2 + (np.sin(ky))**2)

    if band == "pluss":
        return epsilon_tilde + spinorbit * B_k
    if band == "minus":
        return epsilon_tilde - spinorbit * B_k

@jit(nopython = True, nogil = True)
def select_k_values_soc(kx,ky, N, t, soc, band):
    """
    Calculates the momentum values innvolved in the Cooper-pair interactions in a given band

    Parameters
    ----------
    kx : Array of floats
        Momentum values on a grid.
    ky : Array of floats
        Momentum values on a grid.
    t : Float
        Kinetic coupling strength.
    soc : Float
        Strength of spin-orbit interaction.
    band : String
        Indicates band pluss (minus)
    Returns
    -------
    Array of lists
        Selected momentum values in two arrays.
    """
    coord_list = []
    counter = 0
    kx_list = []
    ky_list=[]
    for i in range(N):
        for j in range(N):
            x = kx[i,j]
            y = ky[i,j]
            limit = 8.3 / 9479 / 2  ##Value chosen for lead
            energy = calculate_epsilon(x,y, t, band, soc)
            if abs(energy) <= limit:    
                coord_list.append([i,j])
                kx_list.append(kx[i,j])                                
                ky_list.append(ky[i,j])
                counter += 1
    print("Number of points used in ", band, counter)
    return np.array(kx_list), np.array(ky_list)

@jit(nopython = True, nogil = True)
def solver_function_no_field_lambda(coupling: float,gamma, epsilon, delta: float) -> float:
    """
    Calculates the error resulting from a guess of the coupling strength
    (in zero external field) given the gap value

    Parameters
    ----------
    coupling : Float
        initial guess for coupling strength.
    gamma : Array of floats
        Symmetry factor.
    epsilon : Array of floats
        Kinetic energy.
    delta : Float
        Gap value
    Returns
    -------
    error : Float
        Error resulting from a guess of the coupling strength.
    """
    E = np.real(np.sqrt((epsilon)**2 + gamma * np.conj(gamma) * (delta)**2))
    prefactor = np.real(gamma * np.conj(gamma) / (2 * E))
    guess = np.sum(prefactor)
    return (coupling*guess - 1)

def calculate_initial_lambda(delta_i, delta_f, epsilon_solve, gamma):
    """
    Calculates the initial and final coupling strengths given initial and quenched 
    values of the gap

    Parameters
    ----------
    gamma : Array of floats
        Symmetry factor.
    epsilon_solve : Array of floats
        Kinetic energy.
    delta_i : Float
        Initial gap value
    delta_f : Float
        Final gap value
        
    Returns
    -------
    lambda_i, lambda_f : Touple of Floats
        initial and final coupling strengths.
    """
    lambda_init = 0.00001   ## initial value of the coupling for root finding  
    lambda_i = fsolve(solver_function_no_field_lambda,(lambda_init), (gamma,epsilon_solve,delta_i),  full_output=0, maxfev=500)[0]
    lambda_i = lambda_i
    lambda_f = fsolve(solver_function_no_field_lambda,(lambda_init), (gamma,epsilon_solve,delta_f),  full_output=0, maxfev=500)[0]
    lambda_f = lambda_f
    return lambda_i, lambda_f
    
def solver_function_no_field(delta: float,gamma, epsilon, coupling: float) -> float:
    """
    Calculates the error resulting from a guess of the gap value
    (in zero external field) given the coupling strength

    Parameters
    ----------
    delta : Float
        Initial guess for the gap value.
    gamma : Array of floats
        Symmetry factor.
    epsilon : Array of floats
        Kinetic energy.
    coupling : Float
        Coupling strength.
        
    Returns
    -------
    error : Float
        Error resulting from a guess of the gap.
    """
    E = np.real(np.sqrt((epsilon)**2 + gamma * np.conj(gamma) * (delta)**2))
    prefactor = np.real(gamma * np.conj(gamma) / (2 * E))
    guess = np.sum(prefactor)
    return coupling*guess - 1

def calculate_initial_delta(lambda_i, lambda_f, epsilon_solve, gamma):
    """
    Solves self-consistently for the order parameter with the initial coupling
    strength and the final coupling strength

    Parameters
    ----------
    lambda_i : Float
        Initial coupling strength.
    lambda_f : Float
        Final coupling strength.
    epsilon_solve : Array of floats
        Free particle kinetic energy.
    gamma : Array of floats
        Symmetry factor.
        
    Returns
    -------
    delta_i : Float
        Initial order parameter.
    delta_f : Float
        Final order parameter.
    """
    delta_initial = 0.001   ## initial value of delta for root finding    
    delta_i = fsolve(solver_function_no_field,(delta_initial), (gamma,epsilon_solve,lambda_i),  full_output=0)
    delta_i = (delta_i[0])
    delta_f = fsolve(solver_function_no_field,(delta_initial), (gamma,epsilon_solve,lambda_f),  full_output=0)
    delta_f = (delta_f[0])
    return delta_i, delta_f

def calculate_symmetry(kx,ky,symmetry):
    """
    Calculates a symmetry factor based on
    crystal symmetry

    Parameters
    ----------
    kx : Array of floats
        Momentum values on a grid.
    ky : Array of floats
        Momentum values on a grid.
    symmetry : String
        s or d-wave.

    Returns
    -------
    gamma : Array of floats/float
        Array with the symmetry of the grid.
    """
    if symmetry == "s":
        gamma = np.full_like(kx**2, 1)
    if symmetry == "d":
        #gamma = (kx**2-ky**2)/(kx**2+ky**2)
        gamma = np.abs(np.cos(kx) - np.cos(ky))
    return gamma

def calculate_G0(delta_i, epsilon, gamma):
    """
    Initial value of the Greens function G(0) 
    in the low temperature limit

    Parameters
    ----------
    delta_i : Float
        Order parameter.
    epsilon : Array of floats
        Array with kinetic energy.
    gamma : Array of floats
        Array with symmetry factor.
        
    Returns
    -------
    G_0 : Array of floats
        Initial value G(0).
    """
    glist = - 0.5 * epsilon/np.sqrt((epsilon)**2+gamma * np.conj(gamma) * (delta_i)**2)
    G_0 = glist + 0.5
    G_0 = G_0.astype(np.complex128) 
    return G_0
  
def calculate_F0(delta_i, epsilon, gamma):
    """
    Calculates the initial value of the anomalous Greens function
    F(0) in the low temperature limit

    Parameters
    ----------
    delta_i : Float
        Order parameter.
    epsilon : Array of floats
        Array with free particle kinetic energy.
    gamma : Array of floats
        Array with symmetry factor.

    Returns
    -------
    F_0 : Float
        Array of F(0) values for given initial order parameter.

    """
    F_0 = delta_i * gamma / (2 * np.sqrt((epsilon)**2 + gamma * np.conj(gamma) * (delta_i)**2))
    F_0 = F_0.astype(np.complex128) 
    return F_0

def free_energy(epsilon_pluss, epsilon_minus, delta_pluss, delta_minus, beta):
    E_p = np.sqrt(epsilon_pluss**2+delta_pluss**2) 
    E_m = np.sqrt(epsilon_minus**2+delta_minus**2) 
    E_dynamic_p =np.log(1+np.exp(-beta*E_p))
    E_dynamic_m = np.log(1+np.exp(-beta*E_m))

    energy = np.sum(E_p)+np.sum(E_m)+ np.sum(E_dynamic_m)/beta+np.sum(E_dynamic_p)/beta
    return energy

def calculate_initial_values(delta_i, delta_f, gamma, kx, ky, N,t, spinorbit, SoC_quench, od_factor):
    """
    Calculates all the initial values needed for solving the differential equations

    Parameters
    ----------
    delta_i : Float
        Initial order parameter in zero field.
    delta_f : Float
        Final order parameter in zero field.
    gamma : Array of floats
        Array with symmetry factor.
    kx : Array of floats
        Momentum values in the first BZ discreetized on a grid.
    ky : Array of floats
        Momentum values in the first BZ discreetized on a grid.
    N : Int
        Number of discreetization points in the x (y) direction.
    t : Float
        Hopping parameter strength.
    spinorbit : float
        Initial strength of the spin-orbit interaction.
    SoC_quench : float
        Final strength of the spin-orbit interaction.
    Returns
    -------
    delta_i : List of Floats
        List of values of the order parameters for the different bands.
    lambda_i : Float
        Initial coupling strength.
    lambda_f : Float
        Final coupling strength.
    initial_values : Array
        Array with initial values for the different Greens functions
    arguments : Array
        Array with initial and final values for coupling, kinetic energy and
        symmetry factors.
    """
    
    kx_init, ky_init = select_k_values_soc(kx, ky, N, t, 0, "pluss")
    
    startband ="pluss"
    startspin = 0
    epsilon_solve_init = calculate_epsilon(kx_init,ky_init, t, startband, startspin)

    
    
    kx_pluss, ky_pluss = select_k_values_soc(kx, ky, N, t, spinorbit, "pluss")
    kx_minus, ky_minus = select_k_values_soc(kx, ky, N, t, spinorbit, "minus")
    
    
    gamma_pluss = (np.sin(ky_pluss)+1j*np.sin(kx_pluss)) /np.sqrt((np.sin(ky_pluss)**2 + np.sin(kx_pluss)**2))
    gamma_minus = -(np.sin(ky_minus)+1j*np.sin(kx_minus)) /np.sqrt((np.sin(ky_minus)**2 + np.sin(kx_minus)**2))
    gamma_init =  (np.sin(ky_init)+1j*np.sin(kx_init)) /np.sqrt((np.sin(ky_init)**2+ np.sin(kx_init)**2))


    #### we have to fix ze shapes, and initial vs final gammas
    #gamma_pluss = 1
    #gamma_minus = 1
    gamma_init = 1
    
    lambda_i, lambda_f = calculate_initial_lambda(delta_i, delta_f, epsilon_solve_init, gamma_init)
    
    lambda_i_od = lambda_i*od_factor
    lambda_f_od = lambda_f*od_factor
    
    print("lambda", lambda_i, lambda_f)
    
    epsilon_solve_pluss_init = calculate_epsilon(kx_pluss,ky_pluss,t, "pluss", spinorbit)
    epsilon_solve_minus_init = calculate_epsilon(kx_minus,ky_minus,t, "minus", spinorbit)
        
    epsilon_solve_pluss_quench = calculate_epsilon(kx_pluss,ky_pluss,t, "pluss", SoC_quench)
    epsilon_solve_minus_quench = calculate_epsilon(kx_minus,ky_minus,t, "minus", SoC_quench)
    
    energy_list =[]
    for i in range(10):

        a = anneal_delta_pluss_minus(lambda_i, lambda_i_od, epsilon_solve_pluss_init, epsilon_solve_minus_init)
        print(a["x"], "delta_i")
        if i == 0:
            delta_i_pluss = a["x"][0]
            delta_i_minus = a["x"][1]
        temp_delta_i_pluss = a["x"][0]
        temp_delta_i_minus = a["x"][1]
        energy = free_energy(epsilon_solve_pluss_init, epsilon_solve_minus_init, temp_delta_i_pluss, temp_delta_i_minus, 1)
        energy_list.append(energy)
        print(energy)
        if i!=0:
            if energy_list[i]<energy_list[i-1]:
                delta_i_pluss = a["x"][0]
                delta_i_minus = a["x"][1]

    energy_list =[]
    for i in range(10):

        b = anneal_delta_pluss_minus(lambda_i, lambda_i_od, epsilon_solve_pluss_quench, epsilon_solve_minus_quench)
        print(b["x"], "delta_f")
        if i == 0:
            delta_f_pluss = b["x"][0]
            delta_f_minus = b["x"][1]
        temp_delta_f_pluss = b["x"][0]
        temp_delta_f_minus = b["x"][1]
        energy = free_energy(epsilon_solve_pluss_quench, epsilon_solve_minus_quench, temp_delta_f_pluss, temp_delta_f_minus, 1)
        energy_list.append(energy)
        print(energy)
        if i!=0:
            if energy_list[i]<energy_list[i-1]:
                delta_f_pluss = b["x"][0]
                delta_f_minus = b["x"][1]
    
    print("delta error",calculate_delta_pluss_minus(a["x"], lambda_i, lambda_i_od, epsilon_solve_pluss_init, epsilon_solve_minus_init))
    
    test_delta_pluss = lambda_i* delta_i_pluss* np.sum(1/(2*np.sqrt(epsilon_solve_pluss_init**2+delta_i_pluss**2))) + lambda_i_od* delta_i_minus* np.sum(1/(2*np.sqrt(epsilon_solve_minus_init**2+delta_i_minus**2)))
    test_delta_minus = lambda_i_od* delta_i_pluss* np.sum(1/(2*np.sqrt(epsilon_solve_pluss_init**2+delta_i_pluss**2))) + lambda_i* delta_i_minus* np.sum(1/(2*np.sqrt(epsilon_solve_minus_init**2+delta_i_minus**2)))
    print(delta_i_pluss, test_delta_pluss)
    print(delta_i_minus, test_delta_minus)
    print(delta_i_pluss+ delta_i_minus-test_delta_pluss-test_delta_minus)
    
    delta_i = a["x"]
    delta_f = b["x"]
    free_energy(epsilon_solve_pluss_quench, epsilon_solve_minus_quench, delta_f[0], delta_f[1], 1)
    
    F_0_pluss = calculate_F0(delta_i_pluss, epsilon_solve_pluss_init, gamma_pluss) 
    G_0_pluss = calculate_G0(delta_i_pluss, epsilon_solve_pluss_init, gamma_pluss)
    F_0_minus = calculate_F0(delta_i_minus, epsilon_solve_minus_init, gamma_minus)
    G_0_minus = calculate_G0(delta_i_minus, epsilon_solve_minus_init, gamma_minus)

    temp = lambda_i * np.sum(np.conj(gamma_pluss)*F_0_pluss) + lambda_i_od *np.sum(np.conj(gamma_minus)*F_0_minus)
    
    print("delta_i_pluss", temp)

    if len(gamma_pluss)< len(gamma_minus):
        for i in range(len(gamma_minus)-len(gamma_pluss)):
            gamma_pluss = np.append(gamma_pluss,0)

    
    ## Pads array lengths for integrator
    if len(epsilon_solve_pluss_init) > len(epsilon_solve_minus_init):
        print("saft")
        for i in range(len(epsilon_solve_pluss_init)-len(epsilon_solve_minus_init)):
             epsilon_solve_minus_init = np.append(epsilon_solve_minus_init,0)
             epsilon_solve_minus_quench = np.append(epsilon_solve_minus_quench,0)
             F_0_minus = np.append(F_0_minus,0)
             G_0_minus = np.append(G_0_minus,1/2)
             
    if len(epsilon_solve_minus_init) > len(epsilon_solve_pluss_init):
        for i in range(len(epsilon_solve_minus_init)-len(epsilon_solve_pluss_init)):
             epsilon_solve_pluss_init = np.append(epsilon_solve_pluss_init,0)
             epsilon_solve_pluss_quench = np.append(epsilon_solve_pluss_quench,0)
             F_0_pluss = np.append(F_0_pluss,0)
             G_0_pluss = np.append(G_0_pluss,1/2)    
         
    initial_values = np.array([F_0_pluss, F_0_minus, G_0_pluss, G_0_minus])

    temp = lambda_i * np.sum(np.conj(gamma_pluss)*F_0_pluss) + lambda_i_od *np.sum(np.conj(gamma_minus)*F_0_minus)
    
    print("delta_i_pluss", temp)

    arguments = (lambda_i,lambda_f, epsilon_solve_pluss_init, epsilon_solve_minus_init, epsilon_solve_pluss_quench, epsilon_solve_minus_quench, gamma_pluss, gamma_minus, od_factor)  ## *0.7test dual band SC, change lambda ratio lambda_f, lambda_i *0.6 etc
    
    return delta_i, delta_f, lambda_i, lambda_f, initial_values, arguments

@jit(nopython=True, nogil = True)
def calculate_delta(F_pluss, F_minus, diag_coupling, off_diag_coupling, gamma_pluss, gamma_minus, band):
    """
    Calculates the order parameter based on the current value of the anomalous
    Greens function F

    Parameters
    ----------
    F : Array
        Array of values for F in the pluss (minus) band.
    diag_coupling : Float.
        Coupling strength of cooper pairs.
    off_diag_coupling : Float.
        Coupling strength of interband pair-hopping.
    gamma : Array
        Symmetry factor for the pluss (minus) band.
    band : String
        Indicates pluss (minus) band.
    Returns
    -------
    delta : Complex float
        Gap.

    """
    if band == "pluss":
        delta = diag_coupling* np.sum((np.conj(gamma_pluss))*F_pluss)+off_diag_coupling*np.sum((np.conj(gamma_minus))*F_minus)
    if band == "minus":
        delta = off_diag_coupling* np.sum((np.conj(gamma_pluss))*F_pluss)+diag_coupling*np.sum((np.conj(gamma_minus))*F_minus)
    return delta

@jit(nopython=True, nogil = True)
def calculate_dFdt_pluss(F_pluss, F_minus ,G, coupling, epsilon_solve, gamma_pluss, gamma_minus, off_diag_factor):
    """
    Calculates the derivative of the anomalous Greens function F

    Parameters
    ----------
    F : Array of floats
        Current value of F.
    G : Array of floats
        Current value of G.
    coupling : Float
        Coupling strength.
    epsilon_solve : Array of floats
        Free particle kinetic energy.
    gamma : Array of floats
        Array with symmetry factor.
    off_diag_factor : Float
        Ratio of coupling strength for pair-hopping to intra band pairing.

    Returns
    -------
    dF : Array of floats
        The derivative of F.

    """
    diag_coupling = coupling
    off_diag_coupling = coupling*off_diag_factor 
    delta_temp = calculate_delta(F_pluss, F_minus ,diag_coupling, off_diag_coupling, gamma_pluss, gamma_minus, "pluss")
    delta_temp = gamma_pluss*delta_temp   ## test if this solves
    prefactor = -1j
    first_factor = 2*(epsilon_solve)*F_pluss
    second_factor = delta_temp*(2*G-1)
    dF = prefactor*(first_factor+second_factor)
    return dF

@jit(nopython=True, nogil = True)
def calculate_dFdt_minus(F_pluss, F_minus , G, coupling, epsilon_solve, gamma_pluss, gamma_minus, off_diag_factor):
    """
    Calculates the derivative of the anomalous Greens function F

    Parameters
    ----------
    F : Array of floats
        Current value of F.
    G : Array of floats
        Current value of G.
    coupling : Float
        Coupling strength.
    epsilon_solve : Array of floats
        Free particle kinetic energy.
    gamma : Array of floats
        Array with symmetry factor.
    off_diag_factor : Float
        Ratio of coupling strength for pair-hopping to intra band pairing.

    Returns
    -------
    dF : Array of floats
        The derivative of F.

    """  
    diag_coupling = coupling
    off_diag_coupling = coupling*off_diag_factor
    delta_temp = gamma_minus*calculate_delta(F_pluss, F_minus ,diag_coupling, off_diag_coupling, gamma_pluss, gamma_minus, "minus")
    prefactor = -1j
    first_factor = 2*(epsilon_solve)*F_minus
    second_factor = delta_temp*(2*G-1)
    dF = prefactor*(first_factor+second_factor)
    return dF

@jit(nopython=True, nogil = True)
def calculate_dGdt_pluss(F_pluss, F_minus, coupling, gamma_pluss, gamma_minus, off_diag_factor):
    """
    Calculates the derivative of the Greens function G

    Parameters
    ----------
    F_pluss, F_minus : Array of floats
        Current value of anomalous Greens functions.
    coupling : Float
        Coupling strength of Cooper pairs.
    gamma_minus, gamma_pluss : Array of floats
        Array describing the symmetry of the crystal lattice.
    off_diag_factor : Float
        Coupling factor describing the chance of Cooper pairs hopping to
        a different band.

    Returns
    -------
    Array of floats
        Derivative of the normal Greens function G.
    """
    diag_coupling = coupling
    off_diag_coupling = coupling*off_diag_factor
    delta = gamma_pluss*calculate_delta(F_pluss, F_minus ,diag_coupling, off_diag_coupling, gamma_pluss, gamma_minus, "pluss")
    prefactor = -1j   
    first_factor = np.conjugate(delta)*F_pluss- delta*np.conjugate(F_pluss)
    dg = prefactor * (first_factor)
    return dg

@jit(nopython=True, nogil = True)
def calculate_dGdt_minus(F_pluss, F_minus, coupling, gamma_pluss, gamma_minus, off_diag_factor):
    """
    Calculates the derivative of the Greens function G

    Parameters
    ----------
    F_pluss, F_minus : Array of floats
        Current value of anomalous Greens functions.
    coupling : Float
        Coupling strength of Cooper pairs.
    gamma_minus, gamma_pluss : Array of floats
        Array describing the symmetry of the crystal lattice.
    off_diag_factor : Float
        Coupling factor describing the chance of Cooper pairs hopping to
        a different band.

    Returns
    -------
    Array of floats
        Derivative of the normal Greens function G.
    """
    diag_coupling = coupling
    off_diag_coupling = coupling*off_diag_factor

    delta = gamma_minus*calculate_delta(F_pluss, F_minus ,diag_coupling, off_diag_coupling, gamma_pluss, gamma_minus, "minus")
    prefactor = -1j   
    first_factor = np.conjugate(delta)*F_minus- delta*np.conjugate(F_minus)
    dg = prefactor * (first_factor)
    return dg

def calculate_function_array(y,t,lambda_i,lambda_f, epsilon_solve_pluss_init, epsilon_solve_minus_init, epsilon_solve_pluss_quench, epsilon_solve_minus_quench, gamma_pluss, gamma_minus, od_factor):
    if t==0:
        coupling = lambda_i
        epsilon_solve_pluss = epsilon_solve_pluss_init 
        epsilon_solve_minus = epsilon_solve_minus_init
    else:
        epsilon_solve_pluss = epsilon_solve_pluss_quench
        epsilon_solve_minus = epsilon_solve_minus_quench
        coupling = lambda_f
    




    #off_diag_factor = 9/10
    off_diag_factor = od_factor

    F_pluss = y[0]
    F_minus = y[1]
    G_pluss = y[2]
    G_minus = y[3]
    
    dF_pluss =  calculate_dFdt_pluss(F_pluss, F_minus, G_pluss, coupling, epsilon_solve_pluss, gamma_pluss, gamma_minus, off_diag_factor)
    dF_minus =  calculate_dFdt_minus(F_pluss, F_minus, G_minus, coupling, epsilon_solve_minus, gamma_pluss, gamma_minus, off_diag_factor)
    dG_pluss =  calculate_dGdt_pluss(F_pluss, F_minus, coupling, gamma_pluss, gamma_minus, off_diag_factor)
    dG_minus =  calculate_dGdt_minus(F_pluss, F_minus, coupling, gamma_pluss, gamma_minus, off_diag_factor)
    return np.array([dF_pluss, dF_minus, dG_pluss, dG_minus])