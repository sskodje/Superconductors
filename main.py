# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 18:17:16 2021

@author: Ting42
"""

#%%
import numpy as np
import matplotlib.pyplot as plt
from odeintw import odeintw
from timeit import default_timer as timer
#import calculations as calc
import calculations_no_abs as calc
from joblib import Parallel, delayed
from Band import Band

"""
N gives the size of the grid
limit is the limit of the momentum values in the first BZ
kx,ky is a grid of momenta
"""

    
def main(delta_i, delta_f, gamma, t, save_plot, spinorbit_initial, spinorbit_quench, save_data):


    od_factor = -1/10
    od_factor = 0.5/10
    #od_factor = 0.001/10
    #od_factor = 6/10
    #od_factor = 4/10
    N = 16000
    N = 12000
    #N= 18000 

    limit_first_bz = np.pi
    kxlist = np.linspace(-limit_first_bz,limit_first_bz,N)
    kylist = kxlist    
    kx,ky = np.meshgrid(kxlist, kylist)
    """
    preparing initial values and for the integrator and printing some relevant information
    """ 
    delta_init, delta_final, lambda_i, lambda_f, initial_values, arguments = calc.calculate_initial_values(delta_i, delta_f, gamma, kx, ky, N,t, spinorbit_initial, spinorbit_quench, od_factor) 
    
    """
    Adjustable integration time and number of steps, integration must start at zero to include quench
    """
    #od_factor = 8/10
    
    lambda_f_od = lambda_f *od_factor


    delta_f = (-delta_final[0]+delta_final[1])/2
    print(delta_f, "final delta etter alt")
    delta_i = (delta_init[0]+ delta_init[1])/2
    #delta_f = delta_i*2
    
    end =150/delta_f
    end = 30/delta_f
    end =5/delta_f
    steps = 801*4
    
    t= np.linspace(0,end,steps)
    
    """
    Performs integration with complex matrices using the odeint wrapper, this will take a few minutes
    """
    start = timer()
    print("start")
    z, infodict = odeintw(calc.calculate_function_array, initial_values, t, args=arguments,full_output=True)
    stop = timer()
    print("Time: ", stop-start)
    
    """
    Calculates delta based on the results for F at each time step from the integrator
    """
    delta = np.zeros(steps)
    delta = delta.astype(np.complex128)
    delta_trip = np.zeros(steps)
    delta_trip = delta_trip.astype(np.complex128)
    delta_pluss = np.zeros(steps)
    delta_minus = np.zeros(steps)
    delta_pluss = delta_pluss.astype(np.complex128)
    delta_minus = delta_pluss.astype(np.complex128)
    
    
    
    """Factors describing lattice asymmetry, these may or may not be needed depending on what 
    basis one chooses to work in"""
    kx_minus, ky_minus = calc.select_k_values_soc(kx, ky, N, 1, spinorbit_initial, Band.MINUS)
    kx_pluss, ky_pluss = calc.select_k_values_soc(kx, ky, N, 1, spinorbit_initial, Band.PLUS)
   
    gamma_minus = -( (np.sin(ky_minus)+1j*np.sin(kx_minus)) /np.sqrt((np.sin(ky_minus)**2+ np.sin(kx_minus)**2)) )
    gamma_pluss =( (np.sin(ky_pluss)+1j*np.sin(kx_pluss)) /np.sqrt((np.sin(ky_pluss)**2+ np.sin(kx_pluss)**2)) )
    
    if len(gamma_pluss)< len(gamma_minus):
        for i in range(len(gamma_minus)-len(gamma_pluss)):
            gamma_pluss = np.append(gamma_pluss,0)
    """"""


    #gamma_pluss = 1
    #gamma_minus = 1
    
    for i in range(steps):  
        delta_minus[i]= lambda_f*np.sum(np.conj(gamma_minus)*(z[i][1]))+ lambda_f_od *np.sum(np.conj(gamma_pluss)*z[i][0])  ## mult by gamma_minus to get the "true" delta in pseudspin
        delta_pluss[i]= lambda_f*np.sum(np.conj(gamma_pluss)*(z[i][0]))+ lambda_f_od *np.sum(np.conj(gamma_minus)*z[i][1])
        delta[i]= (delta_minus[i]+ delta_pluss[i])/2
        delta_trip[i]=(-delta_minus[i]+ delta_pluss[i])/2
    absolute_quench_size = round((spinorbit_quench-spinorbit_initial), 9)
    print(absolute_quench_size)
    if save_data == True:
        store_data = np.array([[delta_init, delta_final,  absolute_quench_size], delta_minus, delta_pluss, t])
        #store_data = np.array([[delta_i, delta_f,  absolute_quench_size], delta_minus, delta_pluss, t])
        #data_array = np.column_stack((np.real(delta),np.imag(delta), t))
        # np.savetxt("real.csv", np.real(delta), delimiter=",")
        # np.savetxt("imag.csv", np.imag(delta), delimiter=",")
        # np.savetxt("time.csv", t, delimiter= ",")
        #np.savetxt("data_soc_pluss.csv", data_array, delimiter =",")
        name = str(round(delta_f,9))
        np.save(name, store_data)

    
    # plt.plot(t*delta_f, np.real(delta)/delta_f, color = "blue", label = "$Re (\Delta_+ + \Delta_-)$/2")  
    # plt.plot(t*delta_f, np.imag(delta)/delta_f, color = "purple", label = "$Im (\Delta_+ + \Delta_-|$)/2") 

    plt.plot(t*delta_f, np.abs(delta)/delta_f, color = "green", label = "$|\Delta_+ + \Delta_-|$/2") 
    plt.plot(t*delta_f, np.abs(delta_trip)/delta_f, color = "red", label = "$|\Delta_+ - \Delta_-|$/2") 
    # plt.plot(t*delta_f, np.real(delta_trip)/delta_f, color = "blue", label = "$Re (\Delta_+ - \Delta_-)$/2")  
    # plt.plot(t*delta_f, np.imag(delta_trip)/delta_f, color = "purple", label = "$Im (\Delta_+ - \Delta_-|$)/2") 
    # plt.plot(t*delta_f, np.abs(delta_trip)/delta_f, color = "green", label = "$|\Delta_+ - \Delta_-|$/2") 
    
    # plt.plot(t*delta_f, np.real(delta_minus)/delta_f, color = "red", label = "$Re \Delta_-$") 
    # plt.plot(t*delta_f, np.imag(delta_minus)/delta_f, color = "green", label = "$Im \Delta_-$")  

    plt.plot(t*delta_f, np.abs(delta_minus)/delta_f, color = "black", label = "$|\Delta_-|$")  
    
    plt.plot(t*delta_f, np.abs(delta_pluss)/delta_f, color = "purple", label = "$|\Delta_+|$") 
    
    #plt.xlim(0,30)
    #plt.ylim(0,1.3)
    #plt.ylim(0,2)
    #plt.xlabel("t*$\Delta_f$")
    #plt.ylabel("$\Delta(t)/\Delta_f$")
    plt.legend( prop={'size': 8}, loc = 1)
    if save_figure == True:
        #namefig = "initialsoc"+str(SoC_initial)+"quench"+str(absolute_quench_size)+"od"+str(od_factor)+".pdf"
        namefig = "initialsoc"+str(SoC_initial)+"delta_f"+str(delta_f)+"od"+str(od_factor)+".pdf"
        plt.savefig(namefig)
    plt.show()
    
    length = len(delta_minus)
    avg_length = int(length/4)
    maximum = np.max(np.abs(delta_minus[-avg_length:]))
    minimum = np.min(np.abs(delta_minus[-avg_length:]))
    avg = (maximum+minimum)/2
    print(str(avg)+ "delta_st_minus" )

    length = len(delta_pluss)
    avg_length = int(length/4)
    maximum = np.max(np.abs(delta_pluss[-avg_length:]))
    minimum = np.min(np.abs(delta_pluss[-avg_length:]))
    avg = (maximum+minimum)/2
    print(str(avg)+ "delta_st_pluss" )


    
""""
Starting values for initial and final gap sizes,
kinetic coupling strength t, symmetry set to s-wave, initial and final spin-
orbit coupling strength
"""    

t= 1

delta_i = 1.35/9470
delta_i = delta_i/2
delta_f = delta_i*1#.0001
#delta_f = delta_i*10
#delta_f = delta_i
#delta_f = delta_i*4
save_figure = False
save_data = False
symmetry_factor = 1

SoC_initial = 0.2
SoC_quench = 0.2000001
SoC_quench = 0.2000401
SoC_quench = 0.2002
#SoC_quench = SoC_initial
#SoC_quench = 0.200081
#SoC_initial = 1.8
#SoC_quench = 1.8#000001

main(delta_i, delta_f, symmetry_factor, t, save_figure, SoC_initial, SoC_quench, save_data)

#soc_list = np.geomspace(0.20000001,0.2001, 100)
#soc_list = np.geomspace(0.20020001,0.2003, 100)
#soc_list = np.geomspace(0.20030001,0.2004, 100)

#def process(i):
#    main(delta_i, delta_f, symmetry_factor, t, save_figure, SoC_initial, i, save_data)

#results = Parallel(n_jobs=2)(delayed(process)(soc_list[i]) for i in range(100))
#main(delta_i, delta_f, symmetry_factor, t, save_figure, SoC_initial, SoC_quench, save_data)


def process(i):
    main(delta_i, i, symmetry_factor, t, save_figure, SoC_initial, SoC_quench, save_data)

delta_f_list = np.linspace(delta_f, delta_f/10, 200)

#results = Parallel(n_jobs=2)(delayed(process)(delta_f_list[i+191]) for i in range(100))