# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 18:17:16 2021

@author: Ting42
"""

from matplotlib import rc
import numpy as np
import matplotlib.pyplot as plt
from odeintw import odeintw
from timeit import default_timer as timer
import calculations as calc

"""
N gives the size of the grid
limit is the limit of the momentum values in the first BZ
kx,ky is a grid of momenta
symmetry gives the symmetry of the system, where "s" gives an s-wave symmetry
and "d" gives a d-wave symmetry
"""
N=10000 
N = 12000
N = 14000
# N=18000
limit = np.pi
kxlist = np.linspace(-limit,limit,N)
kylist = kxlist    
kx,ky = np.meshgrid(kxlist, kylist)
hopping_parameter = 1
    
def main(delta_i, delta_f, gamma, t, save_plot, spinorbit_initial, spinorbit_quench, save_data, kx, ky):
    """
    preparing initial values and for the integrator and printing some relevant information
    """
    delta_i, lambda_i, lambda_f, initial_values, arguments = calc.calculate_initial_values(delta_i, delta_f, gamma, kx, ky, N,t, spinorbit_initial, spinorbit_quench) 
    
    """
    Adjustable integration time and number of steps, integration must start at zero to include quench
    """
    od_factor = 1/10
    lambda_f_od = lambda_f *od_factor

    print(delta_i)    
    delta_f = delta_i
     
    end =30/delta_f
    steps = 801*5
    
    t= np.linspace(0,end,steps)
    """
    Performs integration with complex matrices using the odeint wrapper, this will take a few minutes
    """
    start = timer()
    
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
    
    
    # kx_minus, ky_minus = calc.select_k_values_soc(kx, ky, N, 1, spinorbit_initial, "minus")
    # kx_pluss, ky_pluss = calc.select_k_values_soc(kx, ky, N, 1, spinorbit_initial, "pluss")
   
    # gamma_minus = ( (np.sin(ky_minus)-1j*np.sin(kx_minus)) /np.sqrt((np.sin(ky_minus)**2+ np.sin(kx_minus)**2)) )
    # gamma_pluss =( (np.sin(ky_pluss)-1j*np.sin(kx_pluss)) /np.sqrt((np.sin(ky_pluss)**2+ np.sin(kx_pluss)**2)) )
    
    # if len(gamma_pluss)< len(gamma_minus):
    #     for i in range(len(gamma_minus)-len(gamma_pluss)):
    #         gamma_pluss = np.append(gamma_pluss,0)
    
    gamma_pluss = 1
    gamma_minus = 1
    """test"""
    
    for i in range(steps):
        delta_minus[i]= lambda_f*np.sum(np.conj(gamma_minus)*(z[i][0]))+ lambda_f_od *np.sum(np.conj(gamma_pluss)*z[i][1])
        delta_pluss[i]= lambda_f*np.sum(np.conj(gamma_pluss)*(z[i][1]))+ lambda_f_od *np.sum(np.conj(gamma_minus)*z[i][0])
        delta[i]= (delta_minus[i]+ delta_pluss[i])/2
        delta_trip[i]=(-delta_minus[i]+ delta_pluss[i])/2
    absolute_quench_size = spinorbit_quench-spinorbit_initial
    print(absolute_quench_size)
    if save_data == True:
        #store_data = np.array([[delta_i, delta_f,  absolute_quench_size], delta, t])
        #store_data = np.array([[delta_i, delta_f,  absolute_quench_size], delta_minus, delta_pluss, t])
        #data_array = np.column_stack((np.real(delta),np.imag(delta), t))
        np.savetxt("real.csv", np.real(delta), delimiter=",")
        np.savetxt("imag.csv", np.imag(delta), delimiter=",")
        np.savetxt("time.csv", t, delimiter= ",")
        #np.savetxt("data_soc_pluss.csv", data_array, delimiter =",")
        #name = str(round(absolute_quench_size,5))+ symmetry
        #np.save(name, store_data)
    delta_s = (np.abs(delta_minus) + np.abs(delta_pluss))/2
    delta_t = (np.abs(delta_minus) - np.abs(delta_pluss))/2

    
    plt.plot(t*delta_f, np.real(delta)/delta_f, color = "blue", label = "$Re (\Delta_+ + \Delta_-)$/2")  
    plt.plot(t*delta_f, np.imag(delta)/delta_f, color = "purple", label = "$Im (\Delta_+ + \Delta_-|$)/2") 
    plt.plot(t*delta_f, np.abs(delta)/delta_f, color = "green", label = "$|\Delta_+ + \Delta_-|$/2") 
    
    
    # plt.plot(t*delta_f, np.real(delta_trip)/delta_f, color = "blue", label = "$Re (\Delta_+ - \Delta_-)$/2")  
    # plt.plot(t*delta_f, np.imag(delta_trip)/delta_f, color = "purple", label = "$Im (\Delta_+ - \Delta_-|$)/2") 
    # plt.plot(t*delta_f, np.abs(delta_trip)/delta_f, color = "green", label = "$|\Delta_+ - \Delta_-|$/2") 
    
    # plt.plot(t*delta_f, np.abs(delta)/(delta_f*3), color = "green", label = "$|\Delta_+ + \Delta_-|$/2")    ### 3*f, why does this work?
    # plt.plot(t*delta_f, np.abs(delta)/(delta_f), color = "green", label = "$|\Delta_+ + \Delta_-|$/2")  
    
    # plt.plot(t*delta_f, np.real(delta_minus)/delta_f, color = "red", label = "$Re \Delta_-$") 
    # plt.plot(t*delta_f, np.imag(delta_minus)/delta_f, color = "green", label = "$Im \Delta_-$")  
    # plt.plot(t*delta_f, np.abs(delta_minus)/delta_f, color = "black", label = "$|\Delta_-|$")  
    # plt.plot(t*delta_f, np.abs(delta_pluss)/delta_f, color = "green", label = "$|\Delta_+|$")  
    
    #plt.plot(t*delta_f, np.abs(delta_s)/delta_f, color = "red", label = "$(|\Delta_+| \pm  |\Delta_-|)$/2")
    #plt.plot(t*delta_f, np.abs(delta_s)/delta_f, color = "red", label = "$(|\Delta_+| +  |\Delta_-|)$/2")
    #plt.plot(t*delta_f, np.abs(delta_t)/delta_f, color = "red")
    
    #plt.xlim(0,30)
    plt.ylim(-0.5,1.5)
    plt.xlabel("t*$\Delta_i$")
    plt.ylabel("$\Delta(t)/\Delta_i$")
    plt.legend( prop={'size': 8}, loc = 1)
    if save_figure == True:
        namefig = "initialsoc"+str(SoC_initial)+"quench"+str(absolute_quench_size)+"od"+str(od_factor)+".pdf"
        plt.savefig(namefig)
    plt.show()
    
    
    # plt.plot(t*delta_f, np.abs(fft((delta))))
    # plt.xlim(0,1.3)
    # plt.show()
    
    """for different order parameters"""
    
    
""""
Starting values for initial and  coupling strength lambda_i, lambda_f, and
kinetic coupling strength t, symmetry can be "s" or "d" for the different
symmetry groups, lattice toggles the free or lattice kinetic energy
"""    

symmetry ="s"
t= 1

delta_i = 1.35/9470
delta_f = delta_i*1.001
# delta_f = delta_i*3

save_figure = True
save_data = False
symmetry_factor = 1

SoC_initial = 0.1
SoC_quench = 0.10005

# SoC_initial = 0
# SoC_quench = 0.00002

main(delta_i, delta_f, symmetry_factor, t, save_figure, SoC_initial, SoC_quench, save_data, kx, ky)