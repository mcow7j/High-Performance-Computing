# Matthew Cowley
# 01059624
"""MATH 96012 Project 3
Contains four functions:
    simulate2: Simulate bacterial dynamics over m trials. Return: all positions at final time
        and alpha at nt+1 times averaged across the m trials.
    performance: To be completed -- analyze and assess performance of python, fortran, and fortran+openmp simulation codes
    correlation: To be completed -- compute and analyze correlation function, C(tau)
    visualize: To be completed -- generate animation illustrating "non-trivial" particle dynamics
"""
import numpy as np
import matplotlib.pyplot as plt
from m1 import bmotion as bm # assumes that p3_dev_mfin.f90 has been compiled with: f2py3 --f90flags='-fopenmp' -c p3.f90 -m a1 -lgomp
import scipy.spatial.distance as scd
import time
# May also use scipy and time modules as needed


def simulate2(M=10,N=64,L=8,s0=0.2,r0=1,A=0,Nt=100):
    """Simulate bacterial colony dynamics
    Input:
    M: Number of simulations
    N: number of particles
    L: length of side of square domain
    s0: speed of particles
    r0: particles within distance r0 of particle i influence direction of motion
    of particle i
    A: amplitude of noise
    Nt: number of time steps

    Output:
    X,Y: position of all N particles at Nt+1 times
    alpha: alignment parameter at Nt+1 times averaged across M simulation

    Do not modify input or return statement without instructor's permission.

    Add brief description of approach of differences from simulate1 here:
    This code carries out M simulations at a time with partial vectorization
    across the M samples.
    """
    # Set initial condition
    phi_init = np.random.rand(M,N)*(2*np.pi)
    r_init = np.sqrt(np.random.rand(M,N))
    Xinit,Yinit = r_init*np.cos(phi_init),r_init*np.sin(phi_init)
    Xinit+=L/2
    Yinit+=L/2
    # ---------------------

    # Initialize variables
    P = np.zeros((M,N,2)) # positions
    P[:,:,0],P[:,:,1] = Xinit,Yinit
    alpha = np.zeros((M,Nt+1)) # alignment parameter
    S = np.zeros((M,N),dtype=complex) # hases
    T = np.random.rand(M,N)*(2*np.pi) #direction of motion
    n = np.zeros((M,N)) #number of neighbors
    E = np.zeros((M,N,Nt+1),dtype=complex)
    d = np.zeros((M,N,N))
    dtemp = np.zeros((M,N*(N-1)//2))
    AexpR = np.random.rand(M,N,Nt)*(2*np.pi)
    AexpR = A*np.exp(1j*AexpR)

    r0sq = r0**2
    E[:,:,0] = np.exp(1j*T)

    #Time marching-----------
    for i in range(Nt):
        for j in range(M):
            dtemp[j,:] = scd.pdist(P[j,:,:],metric='sqeuclidean')

        dtemp2 = dtemp<=r0sq
        for j in range(M):
            d[j,:,:] = scd.squareform(dtemp2[j,:])
        n = d.sum(axis=2) + 1
        S = E[:,:,i] + n*AexpR[:,:,i]

        for j in range(M):
            S[j,:] += d[j,:,:].dot(E[j,:,i])

        T = np.angle(S)

        #Update X,Y
        P[:,:,0] = P[:,:,0] + s0*np.cos(T)
        P[:,:,1] = P[:,:,1] + s0*np.sin(T)

        #Enforce periodic boundary conditions
        P = P%L

        E[:,:,i+1] = np.exp(1j*T)
    #----------------------

    #Compute order parameter
    alpha = (1/(N*M))*np.sum(np.abs(E.sum(axis=1)),axis=0)

    return P[:,:,0],P[:,:,1],alpha


def performance(input_p=(None),display=False):
    """Assess performance of simulate2, simulate2_f90, and simulate2_omp
    Modify the contents of the tuple, input, as needed
    When display is True, figures equivalent to those
    you are submitting should be displayed
    """

    #set up values of of n,m and n&m we wish to test the differents codes over
    # used different values of n,m,nmarray for ease of testing as other code takes a very long time to run,

    #but to produce figure use code commented below

    narray=[10,20,30,40,50,60,70,80,90,100,200,300,400,500,750]
    marray=[10,20,30,40,50,60,70,80,90,100,200,300,400,500,750]
    nmarray=[10,50,100,150,200,250,300,400]







    M1=64
    N1=64
    L1=16
    r01=1
    Nt1=100
    bm.bm_l=L1
    bm.bm_s0=0.2
    bm.bm_r0=r01
    bm.bm_a=0.64
    bm.numthreads=int(2)
    #find how many ns and ms we are varing over
    m_len=len(marray)
    n_len=len(narray)
    nm_len=len(nmarray)
    #set up data for time entries m varying
    time_ser_fortan_m=np.zeros(m_len)
    time_para_fortran_m=np.zeros(m_len)
    time_python_m=np.zeros(m_len)
    time_para_thread4_fortran_m=np.zeros(m_len)
    #set up data for time entries n varying
    time_ser_fortan_n=np.zeros(n_len)
    time_para_fortran_n=np.zeros(n_len)
    time_python_n=np.zeros(n_len)
    time_para_thread4_fortran_n=np.zeros(n_len)
    #set up data for time entries n and m varying
    time_ser_fortan_nm=np.zeros(nm_len)
    time_para_fortran_nm=np.zeros(nm_len)
    time_python_nm=np.zeros(nm_len)
    time_para_thread4_fortran_nm=np.zeros(nm_len)
    #generate data for m varying
    for i in range(m_len):
        bm.numthreads=int(2)
        #generates time data for fortran simulate2 serial code
        start_ser_fortran = time.time()
        x,y,alpha_ave = bm.simulate2_f90(marray[i],N1,Nt1)
        end_ser_fortran = time.time()
        time_ser_fortan_m[i]=end_ser_fortran - start_ser_fortran
        #genrates time data for fortran OMP parallel code threads 2
        start_para_fortran = time.time()
        x1,y1,alpha_ave1 = bm.simulate2_omp(marray[i],N1,Nt1)
        end_para_fortran = time.time()
        time_para_fortran_m[i]=end_para_fortran - start_para_fortran
        #generates time data for python simulate 2
        start_time_python = time.time()
        x2,y2,alpha_ave2 = simulate2(marray[i],N1,L1,0.2,r01,0.64,Nt1)
        end_time_python = time.time()
        time_python_m[i]=end_time_python - start_time_python
        #genrates time data for fortran OMP parallel code threads 4
        bm.numthreads=int(4)
        start_para_fortran = time.time()
        x1,y1,alpha_ave1 = bm.simulate2_omp(marray[i],N1,Nt1)
        end_para_fortran = time.time()
        time_para_thread4_fortran_m[i]=end_para_fortran - start_para_fortran
    for i in range(n_len):
        bm.numthreads=int(2)
        #generates time data for fortran simulate2 serial code
        start= time.time()
        x,y,alpha_ave_n = bm.simulate2_f90(M1,narray[i],Nt1)
        end = time.time()
        time_ser_fortan_n[i]=end - start
        #genrates time data for fortran omp parallel code threads 2
        start = time.time()
        x1,y1,alpha_ave1 = bm.simulate2_omp(M1,narray[i],Nt1)
        end = time.time()
        time_para_fortran_n[i]=end - start
        #generates time data for python simulate 2
        start = time.time()
        x2,y2,alpha_ave2 = simulate2(M1,narray[i],L1,0.2,r01,0.64,Nt1)
        end = time.time()
        time_python_n[i]=end - start
        #genrates time data for fortran omp parallel code threads 4
        bm.numthreads=int(4)
        start = time.time()
        x1,y1,alpha_ave1 = bm.simulate2_omp(M1,narray[i],Nt1)
        end = time.time()
        time_para_thread4_fortran_n[i]=end - start
    for i in range(nm_len):
        #generates time data for fortran simulate2 serial code
        start= time.time()
        x,y,alpha_ave = bm.simulate2_f90(nmarray[i],nmarray[i],Nt1)
        end = time.time()
        time_ser_fortan_nm[i]=end - start
        #genrates time data for fortran omp parallel code
        bm.numthreads=int(2)
        start = time.time()
        x1,y1,alpha_ave1 = bm.simulate2_omp(nmarray[i],nmarray[i],Nt1)
        end = time.time()
        time_para_fortran_nm[i]=end - start
        #generates time data for python simulate 2
        start = time.time()
        x2,y2,alpha_ave2 = simulate2(nmarray[i],nmarray[i],L1,0.2,r01,0.64,Nt1)
        end = time.time()
        time_python_nm[i]=end - start
        #genrates time data for fortran omp parallel code
        bm.numthreads=int(4)
        start = time.time()
        x1,y1,alpha_ave1 = bm.simulate2_omp(nmarray[i],nmarray[i],Nt1)
        end = time.time()
        time_para_thread4_fortran_nm[i]=end - start
    if display==True:
        #plots m varying graph
        plt.figure()
        plt.plot(marray,time_ser_fortan_m, label='serial fortan code')
        plt.plot(marray,time_para_fortran_m, label='OMP parallelised threads=2')
        plt.plot(marray,time_python_m, label='python code')
        plt.plot(marray,time_para_thread4_fortran_m, label='OMP parallelised threads=4')
        plt.legend()
        plt.xlabel("Value of M")
        plt.ylabel("run time (seconds)")
        plt.title("Matthew Cowley, How varying M effects \n performance of different models for N=%i,Nt=100" %N1)
        #plots small m varying graph
        plt.figure()
        plt.plot(marray[0:10],time_ser_fortan_m[0:10], label='serial fortan code')
        plt.plot(marray[0:10],time_para_fortran_m[0:10], label='OMP parallelised threads=2')
        plt.plot(marray[0:10],time_python_m[0:10], label='python code')
        plt.plot(marray[0:10],time_para_thread4_fortran_m[0:10], label='OMP parallelised threads=4')
        plt.legend()
        plt.xlabel("Value of M")
        plt.ylabel("run time (seconds)")
        plt.title("Matthew Cowley, How varying for small M effects \n performance of different models for N=%i,Nt=100" %N1)
        #plots n varying graph

        plt.figure()
        plt.plot(narray,time_ser_fortan_n, label='serial fortan code')
        plt.plot(narray,time_para_fortran_n, label='OMP parallelised threads=2')
        plt.plot(narray,time_python_n, label='python code')
        plt.plot(narray,time_para_thread4_fortran_n, label='OMP parallelised threads=4')
        plt.legend()
        plt.xlabel("Value of N")
        plt.ylabel("run time (seconds)")
        plt.title("Matthew Cowley, How varying N effects \n performance of different models for M=%i,Nt=100" %M1)

        #plots n and m varying
        plt.figure()
        plt.plot(nmarray,time_ser_fortan_nm, label='serial fortan code')
        plt.plot(nmarray,time_para_fortran_nm, label='OMP parallelised threads=2')
        plt.plot(nmarray,time_python_nm, label='python code')
        plt.plot(nmarray,time_para_thread4_fortran_nm, label='OMP parallelised threads=4')
        plt.legend()
        plt.xlabel("Value of N and M")
        plt.ylabel("run time (seconds)")
        plt.title("Matthew Cowley, How varying N and M effects \n performance of different models for Nt=100")



    return None #Modify as needed

def correlation(input_c=(None),display=False):
    """Compute and analyze temporal correlation function, C(tau)
    Modify the contents of the tuple, input, as needed
    When display is True, figures equivalent to those
    you are submitting should be displayed
    """
    #function calulate c(tau) for m simulations for tau in [0,tau_max] for set values of a and b
    #uses omp parralelised altered form of simulate2 to calulate alpha
    def c_tau(a,b,tau_max,m):
        #set l,s0,r0,a-noise,numteads
        bm.bm_l=16
        bm.bm_s0=0.1
        bm.bm_r0=1
        bm.bm_a=0.625
        bm.numthreads=int(2)
        x,y,alpha = bm.simulate2_omp_tau(m,400,b+tau_max)

        #set up tau array
        tau=np.zeros(tau_max)
        #set up array to calculate <alpha(t+T)alpha(t)>
        farray=np.zeros((m,tau_max))
        #set up array to calculate <alpha(t)>**2
        darray=np.zeros((m,tau_max))
        #set up final ctau array
        ctau=np.zeros((m,tau_max))
        #loop over T
        for T in range(tau_max):
            #loop which sums t0 effectively
          for i in range(b-a):
              darray[:,T]=darray[:,T]+np.multiply(alpha[:,a+i+T+1],alpha[:,a+i+1])
              farray[:,T]=farray[:,T]+alpha[:,a+i+1]
        #calculate tau value for graph purposes
          tau[T]=T

        farray=farray*(1/(b-a))
        darray=darray*(1/(b-a))
        ctau=np.subtract(darray,np.multiply(farray,farray))
        return ctau,tau

    if display==True:
        m=8
        ctau,tau=c_tau(1200,1400,80,m)
        plt.figure()
        for m1 in range(m):
            plt.plot(tau,ctau[m1,:])
        plt.xlabel(r"$\tau$")
        plt.ylabel(r"C($\tau$)")
        plt.title("Matthew Cowley, How C(tau) varies for a=1200,b=1400,8 simulations")

        m=40
        plt.figure()
        ctau,tau=c_tau(1200,1400,80,m)
        plt.plot(tau,np.sum(ctau,axis=0)/m,label='a=1200,b=1400')
        ctau,tau=c_tau(1200,1600,80,m)
        plt.plot(tau,np.sum(ctau,axis=0)/m,label='a=1200,b=1600')
        ctau,tau=c_tau(1200,1800,80,m)
        plt.plot(tau,np.sum(ctau,axis=0)/m,label='a=1200,b=1800')
        ctau,tau=c_tau(1200,2000,80,m)
        plt.plot(tau,np.sum(ctau,axis=0)/m,label='a=1200,b=2000')
        ctau,tau=c_tau(1200,2500,80,m)
        plt.plot(tau,np.sum(ctau,axis=0)/m,label='a=1200,b=2500')
        plt.legend(loc='upper right')
        plt.xlabel(r"$\tau$")
        plt.ylabel(r"C($\tau$) average")
        plt.title("Matthew Cowley, How increasing b-a effects C(tau) average \n (averaged over 40 simulations)")

    return None #Modify as needed



def visualize():
    """Generate an animation illustrating particle dynamics
    """
    #reuse project 1 simulate1 function to produce X and Y at all time steps
    def simulate1(N=64,L=8,s0=0.2,r0=1,A=0,Nt=100,randomT=False):
        """Part1: Simulate bacterial colony dynamics
        Input:
        N: number of particles
        L: length of side of square domain
        s0: speed of particles
        r0: particles within distance r0 of particle i influence direction of motion
        of particle i
        A: amplitude of noise
        Nt: number of time steps

        Output:
        X,Y: position of all N particles at Nt+1 times
        alpha: alignment parameter at Nt+1 times

        Do not modify input or return statement without instructor's permission.

        Add brief description of approach to problem here:
        """
        #Set initial condition
        phi_init = np.random.rand(N)*(2*np.pi)
        r_init = np.sqrt(np.random.rand(N))
        Xinit,Yinit = r_init*np.cos(phi_init),r_init*np.sin(phi_init)
        Xinit+=L/2
        Yinit+=L/2
        #---------------------

        #Initialize variables
        P = np.zeros((Nt+1,N,2)) #positions
        P[0,:,0],P[0,:,1] = Xinit,Yinit
        alpha = np.zeros(Nt+1) #alignment parameter
        S = np.zeros(N,dtype=complex) #phases
        T = np.random.rand(N)*(2*np.pi) #direction of motion
        n = np.zeros(N) #number of neighbors
        E = np.zeros((N,Nt+1),dtype=complex)
        r0sq = r0**2

        AexpR = np.random.rand(N,Nt)*(2*np.pi)
        AexpR = A*np.exp(1j*AexpR)

        E[:,0] = np.exp(1j*T)

        #Time marching-----------
        for i in range(Nt):

            d = scd.pdist(P[i,:,:],metric='sqeuclidean')
            d = d<=r0sq
            d = scd.squareform(d)
            n = d.sum(axis=0)+1
            S = E[:,i].copy()
            #Compute n, sum phases---
            S += d.dot(E[:,i]) + n*AexpR[:,i]

            #This loop avoids multiplications with zero that occur
            #in the line above
            # for j in range(N):
            #     S[j] = E[d[j,:],i].sum()
            #S += n*AexpR[:,i]
            T = np.angle(S)

            #Update X,Y
            P[i+1,:,0] = P[i,:,0] + s0*np.cos(T)
            P[i+1,:,1] = P[i,:,1] + s0*np.sin(T)

            #Enforce periodic boundary conditions
            P[i+1,:,:] = P[i+1,:,:]%L


            E[:,i+1] = np.exp(1j*T)

        #----------------------

        #Compute order parameter
        Ninv = 1/N
        alpha = Ninv*np.abs(E.sum(axis=0))

        X,Y = P[:,:,0],P[:,:,1]
        return X,Y,alpha

        #gnerate data using simulate1
    X,Y,alpha=simulate1(128,16,0.1,1,0.625,1000)


    #set up animation
    fig, ax = plt.subplots()
    line, = ax.plot(X[0,:],Y[0,:],'.')
    ax.set_xlim(0,16)
    ax.set_ylim(0,16)
    ax.set_title("Matthew Cowley, Simulation of particles \n for N=128,s0=0.1,r0=1,A=0.625")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")



    def updatefig(i):
        """Updates figure each time function is called
        and returns new figure 'axes'
        """
        print("time=",i)
        line.set_ydata(Y[i,:])
        line.set_xdata(X[i,:])
        return line,

    ani = animation.FuncAnimation(fig, updatefig, frames=1000,interval=100,repeat=False)
    #save animation
    FFwriter = animation.FFMpegWriter(fps=25)
    ani.save('p3movie.mp4', writer=FFwriter)

    return None #Modify as needed


if __name__ == '__main__':
    #Modify the code here so that it calls performance analyze and
    # generates the figures that you are submitting with your code

    input_p = None
    output_p = performance(input_p) #modify as needed

    input_c = None
    output_c = correlation(input_c)
