"""Final project, part 1"""
import numpy as np
import matplotlib.pyplot as plt
from m1 import flow as fl
import time

#assumes files have been compiled with belo command lind to get omp form and just seriel form
#f2py3 --f90flags='-fopenmp' -c p41.f90 -m m1 -lgomp
#f2py3 -c p41.f90 -m q1

def jacobi(n,kmax=10000,tol=1.0e-8,s0=0.1,display=False):
    """ Solve liquid flow model equations with
        jacobi iteration.
        Input:
            n: number of grid points in r and theta
            kmax: max number of iterations
            tol: convergence test parameter
            s0: amplitude of cylinder deformation
            display: if True, plots showing the velocity field and boundary deformation
            are generated
        Output:
            w,deltaw: Final velocity field and |max change in w| each iteration
    """

    #-------------------------------------------
    #Set Numerical parameters and generate grid
    Del_t = 0.5*np.pi/(n+1)
    Del_r = 1.0/(n+1)
    Del_r2 = Del_r**2
    Del_t2 = Del_t**2
    r = np.linspace(0,1,n+2)
    t = np.linspace(0,np.pi/2,n+2) #theta
    tg,rg = np.meshgrid(t,r) # r-theta grid

    #Factors used in update equation (after dividing by gamma)
    rg2 = rg*rg
    fac = 0.5/(rg2*Del_t2 + Del_r2)
    facp = rg2*Del_t2*fac*(1+0.5*Del_r/rg) #alpha_p/gamma
    facm = rg2*Del_t2*fac*(1-0.5*Del_r/rg) #alpha_m/gamma
    fac2 = Del_r2*fac #beta/gamma
    RHS = fac*(rg2*Del_r2*Del_t2) #1/gamma

    #set initial condition/boundary deformation
    w0 = (1-rg**2)/4 #Exact solution when s0=0
    s_bc = s0*np.exp(-10.*((t-np.pi/2)**2))/Del_r
    fac_bc = s_bc/(1+s_bc)

    deltaw = []
    w = w0.copy()
    wnew = w0.copy()

    #Jacobi iteration
    for k in range(kmax):
        #Compute wnew
        wnew[1:-1,1:-1] = RHS[1:-1,1:-1] + w[2:,1:-1]*facp[1:-1,1:-1] + w[:-2,1:-1]*facm[1:-1,1:-1] + (w[1:-1,:-2] + w[1:-1,2:])*fac2[1:-1,1:-1] #Jacobi update

        #Apply boundary conditions
        wnew[:,0] = wnew[:,1] #theta=0
        wnew[:,-1] = wnew[:,-2] #theta=pi/2
        wnew[0,:] = wnew[1,:] #r=0
        wnew[-1,:] = wnew[-2,:]*fac_bc #r=1s

        #Compute delta_p
        deltaw += [np.max(np.abs(w-wnew))]
        w = wnew.copy()
        if k%1000==0: print("k,dwmax:",k,deltaw[k])
        #check for convergence
        if deltaw[k]<tol:
            print("Converged,k=%d,dw_max=%28.16f " %(k,deltaw[k]))
            break

    deltaw = deltaw[:k+1]

    if display:
        #plot final velocity field, difference from initial guess, and cylinder
        #surface
        plt.figure()
        plt.contour(t,r,w,50)
        plt.xlabel(r'$\theta$')
        plt.ylabel('r')
        plt.title('Final velocity field')

        plt.figure()
        plt.contour(t,r,np.abs(w-w0),50)
        plt.xlabel(r'$\theta$')
        plt.ylabel('r')
        plt.title(r'$|w - w_0|$')

        plt.figure()
        plt.polar(t,np.ones_like(t),'k--')
        plt.polar(t,np.ones_like(t)+s_bc*Del_r,'r-')
        plt.title('Deformed cylinder surface')

    return w,deltaw



def performance():
    """Analyze performance of codes
    Add input/output variables as needed.
    """

    return None



if __name__=='__main__':
    #Add code below to call performance
    #and generate figures you are submitting in
    #your repo.
    marray=[10,20,30,40,50,60,70,80,90,100,150,200,300,400,500]
    m_len=len(marray)
    #set up data for time entries m varying
    time_ser=np.zeros(m_len)
    time_par2=np.zeros(m_len)
    time_par4=np.zeros(m_len)
    from q1 import flow as fl

    for i in range(m_len):
            start_ser_fortran = time.time()
            w1 = fl.sgisolve(marray[i])
            end_ser_fortran = time.time()
            time_ser[i]=end_ser_fortran - start_ser_fortran
    from m1 import flow as fl
    fl.numthreads=2
    for i in range(m_len):
            start_ser_fortran = time.time()
            w1 = fl.sgisolve(marray[i])
            end_ser_fortran = time.time()
            time_par2[i]=end_ser_fortran - start_ser_fortran

    fl.numthreads=4
    for i in range(m_len):
            start_ser_fortran = time.time()
            w1 = fl.sgisolve(marray[i])
            end_ser_fortran = time.time()
            time_par4[i]=end_ser_fortran - start_ser_fortran


            plt.figure()
            plt.plot(marray,time_ser/time_par2,label='2 threads')
            plt.plot(marray,time_ser/time_par4,label='4 threads')
            plt.plot(marray,time_ser*0+1)
            plt.legend()
            plt.legend()
            plt.xlabel("M value")
            plt.ylabel("run time (seconds)")
            plt.title("Matthew Cowley, Speed up for MPI with varying M Relative to Serial")


    marray=[10,20,30,40,50,60,70,80,90,100,110,120]
    m_len=len(marray)
    #set up data for time entries m varying
    time_ser=np.zeros(m_len)
    time_par1=np.zeros(m_len)
    time_python=np.zeros(m_len)
    time_par2=np.zeros(m_len)
    time_jack=np.zeros(m_len)

    fl.numthreads=1
    for i in range(m_len):
            start_ser_fortran = time.time()
            w1 = fl.sgisolve(marray[i])
            end_ser_fortran = time.time()
            time_par1[i]=end_ser_fortran - start_ser_fortran

    fl.numthreads=2
    for i in range(m_len):
            start_ser_fortran = time.time()
            w1 = fl.sgisolve(marray[i])
            end_ser_fortran = time.time()
            time_par2[i]=end_ser_fortran - start_ser_fortran

    for i in range(m_len):
            start_ser_fortran = time.time()
            w1 = fl.jacobi(marray[i])
            end_ser_fortran = time.time()
            time_jack[i]=end_ser_fortran - start_ser_fortran


    from q1 import flow as fl
    for i in range(m_len):
            start_ser_fortran = time.time()
            w1 = fl.sgisolve(marray[i])
            end_ser_fortran = time.time()
            time_ser[i]=end_ser_fortran - start_ser_fortran

    marray=[10,20,30,40,50,60,70,80,90,100,150,200,300,400,500]
    m_len=len(marray)
    #set up data for time entries m varying
    time_ser=np.zeros(m_len)
    time_par2=np.zeros(m_len)
    time_par4=np.zeros(m_len)
    from q1 import flow as fl
    for i in range(m_len):
            start_ser_fortran = time.time()
            w1 = fl.sgisolve(marray[i])
            end_ser_fortran = time.time()
            time_ser[i]=end_ser_fortran - start_ser_fortran
    from m1 import flow as fl
    fl.numthreads=2
    for i in range(m_len):
            start_ser_fortran = time.time()
            w1 = fl.sgisolve(marray[i])
            end_ser_fortran = time.time()
            time_par2[i]=end_ser_fortran - start_ser_fortran

    fl.numthreads=4
    for i in range(m_len):
            start_ser_fortran = time.time()
            w1 = fl.sgisolve(marray[i])
            end_ser_fortran = time.time()
            time_par4[i]=end_ser_fortran - start_ser_fortran


    plt.figure()
    plt.plot(marray,time_ser/time_par2,label='2 threads')
    plt.plot(marray,time_ser/time_par4,label='4 threads')
    plt.plot(marray,time_ser*0+1)
    plt.legend()
    plt.legend()
    plt.xlabel("Speed up")
    plt.ylabel("run time (seconds)")
    plt.title("Matthew Cowley, Speed up for MPI with varying M Relative to Seriel code")





#last figure

    from m1 import flow as fl
fl.numthreads=2
w1 = fl.sgisolve(300)
g=fl.fl_deltaw
w2=fl.jacobi(300)
f=fl.fl_deltaw

plt.figure()
plt.plot(np.arange(1,101,1),f[1:101],label='m=300, Jacobi')
plt.plot(np.arange(1,101,1),g[1:101],label='m=300, SGI 2 cores')
plt.legend()
plt.xlabel("iterations")
plt.ylabel("max change in velocity ")
plt.title("Matthew Cowley, How Jacobi and SGI converge")
