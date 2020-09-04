"""MATH96012 2019 Project 1
Matthew Cowley 01059624
"""
import numpy as np
import matplotlib.pyplot as plt
#--------------------------------

def simulate1(N=64,L=8,s0=0.2,r0=1,A=0.2,Nt=100):
    """Part1: Simulate bacterial colony dynamics
    Input:
    N: number of particles
    L: length of side of square domain
    s0: speed of particles
    r0: particles within distance r0 of particle i influence direction of motion
    of particle i
    A: amplitude of noise
    dt: time step
    Nt: number of time steps

    Output:
    X,Y: position of all N particles at Nt+1 times
    alpha: alignment parameter at Nt+1 times

    Do not modify input or return statement without instructor's permission.

       The matrices X and Y show how the particles move over time given initial conditions, with the rows representing a particle and columns a step in time.
    For every step in time you calculate new position of particles dependent on the previous position(i.e the previous column).
    the way the particles move dependents on other particles and a noise. These effects are calculated essentially by a dummy variable "theta".

    The code uses built in numpy functions wherever possible combined with turning everything into matrix problem (instead of multiple for loops) to improve effici
    """
    #Set initial condition
    phi_init = np.random.rand(N)*(2*np.pi)
    r_init = np.sqrt(np.random.rand(N))
    Xinit,Yinit = r_init*np.cos(phi_init),r_init*np.sin(phi_init)
    Xinit+=L/2
    Yinit+=L/2

    theta_init = np.random.rand(N)*(2*np.pi) #initial directions of motion

    #---------------------
    #create Matrix columns number represents time and row which particl
    X=np.zeros((N,Nt+1))
    Y=np.zeros((N,Nt+1))

    #creates alpha
    alpha_row=np.zeros((1,Nt+1))

    #X-mat and Y-mat helped to calculate  theta at every time step.
    X_mat=np.zeros((N,N))
    Y_mat=np.zeros((N,N))

    #create theta matix, rows are for a corresponding partice, columns a step forward in time
    theta=np.zeros((N,Nt+1))

    #input intial conditions into X-matrix
    X[:,0]=Xinit.copy()
    Y[:,0]=Yinit.copy()
    theta[:,0]=theta_init.copy()


    #a for loop is need to iterate over time
    for i in range(Nt):
        #X-mat and Y-mat helped to calculate  theta at every time step.
        X_mat=np.zeros((N,N))
        Y_mat=np.zeros((N,N))

        #columns are just the x_ith position at time j going from 1:n
        X_mat[:,:]=X[:,i]
        Y_mat[:,:]=Y[:,i]

        #dist_ij finds distance^2 between x_i and x_j at time j
        dist=np.square(X_mat-np.transpose(X_mat))+np.square(Y_mat-np.transpose(Y_mat))

        #creates matrix of 1s and zeros, 1 if x_i and x_j are within r0^2 of each other
        dist_matrix=(np.less_equal(dist,r0**2)*1)

        #sums how many particles are within distance r_o of each other
        sum_matrix=np.sum(dist_matrix,axis=1,keepdims = True)


        #makes randon uniform variable and genrates noise matrix making sure it has correct dimensions
        noise=np.reshape(A*sum_matrix*np.exp(1j*np.random.uniform(0,2*np.pi,(N,1))),(N,))

        #calculates theta using matrix multpilcation to improve effieceny
        theta[:,i+1]=np.angle(np.matmul(dist_matrix,np.exp(1j*theta[:,i]))+noise)

        #updates x and y valuse using modular arithiitc to ensuring boundary conditions are met.
        X[:,i+1]=np.remainder(X[:,i]+s0*np.cos(theta[:,i+1]),L)
        Y[:,i+1]=np.remainder(Y[:,i]+s0*np.sin(theta[:,i+1]),L)

    #calculates alpha after loop from theta matrix, using numpy so it is effiencent
    alpha_row=(1/N)*np.absolute(np.sum(np.exp(theta*1j),axis=0,keepdims = True))

    return X,Y,alpha_row

def simulate2(N=64,L=8,s0=0.2,r0=1,A=0.2,Nt=100):
    """Same as simulate1, apart from returns alpha as column vector as easier to plot wih graphs, doesn't return X or Y
    """
    #Set initial condition
    phi_init = np.random.rand(N)*(2*np.pi)
    r_init = np.sqrt(np.random.rand(N))
    Xinit,Yinit = r_init*np.cos(phi_init),r_init*np.sin(phi_init)
    Xinit+=L/2
    Yinit+=L/2

    theta_init = np.random.rand(N)*(2*np.pi) #initial directions of motion

    #---------------------
    #create Matrix columns number represents time and row which particl
    X=np.zeros((N,Nt+1))
    Y=np.zeros((N,Nt+1))

    #creates alpha
    alpha_row=np.zeros((1,Nt+1))

    #X-mat and Y-mat helped to calculate  theta at every time step.
    X_mat=np.zeros((N,N))
    Y_mat=np.zeros((N,N))

    #create theta matix, rows are for a corresponding partice, columns a step forward in time
    theta=np.zeros((N,Nt+1))

    #input intial conditions into X-matrix
    X[:,0]=Xinit.copy()
    Y[:,0]=Yinit.copy()
    theta[:,0]=theta_init.copy()


    #a for loop is need to iterate over time
    for i in range(Nt):
        #X-mat and Y-mat helped to calculate  theta at every time step.
        X_mat=np.zeros((N,N))
        Y_mat=np.zeros((N,N))

        #columns are just the x_ith position at time j going from 1:n
        X_mat[:,:]=X[:,i]
        Y_mat[:,:]=Y[:,i]

        #dist_ij finds distance^2 between x_i and x_j at time j
        dist=np.square(X_mat-np.transpose(X_mat))+np.square(Y_mat-np.transpose(Y_mat))

        #creates matrix of 1s and zeros, 1 if x_i and x_j are within r0^2 of each other
        dist_matrix=(np.less_equal(dist,r0**2)*1)

        #sums how many particles are within distance r_o of each other
        sum_matrix=np.sum(dist_matrix,axis=1,keepdims = True)


        #makes randon uniform variable and genrates noise matrix making sure it has correct dimensions
        noise=np.reshape(A*sum_matrix*np.exp(1j*np.random.uniform(0,2*np.pi,(N,1))),(N,))

        #calculates theta using matrix multpilcation to improve effieceny
        theta[:,i+1]=np.angle(np.matmul(dist_matrix,np.exp(1j*theta[:,i]))+noise)

        #updates x and y valuse using modular arithiitc to ensuring boundary conditions are met.
        X[:,i+1]=np.remainder(X[:,i]+s0*np.cos(theta[:,i+1]),L)
        Y[:,i+1]=np.remainder(Y[:,i]+s0*np.sin(theta[:,i+1]),L)

    #calculates alpha after loop from theta matrix, using numpy so it is effiencent
    alpha_row=(1/N)*np.absolute(np.sum(np.exp(theta*1j),axis=0,keepdims = True))
    #for ease of graph plotting transpose alpha
    alpha_col=alpha_row.transpose()

    return alpha_col

def Analyse(A=A,N=16,Nt=1000,Samples=5,nskip=100):
    """Part 2:
    Input:
    A-matrix giving you different values for A
    N-number of particlies for each simulations
    Nt- number of time steps
    Samples- number of samples for each value of A
    nskip-skips this many time steps, when calulating varinace

    Output:
    w-same size as A, with corresponding average varience of alpha of n samples, igonring nskip interations.
    i.e should wait till there is relatively stable behavior
    max_A- scalar value of A for which alpha varies the most

    uses simulate2 for data with default values of L=4,s0=0.2,r0=1.
    """
    #sets up w same dimensions as A
    w=A*0
    #generates sam number of samples for each value of A by iteratng over sams
    for f in range(Samples):
        #dummy variable to keep track of indexing
        k=0
        #sets up Alpha varable
        Big_Alpha=np.zeros((Nt+1,len(A)))

        #    loops over A
        for x in A:
            #generates dat
            A1= simulate2(N=N,L=4,s0=0.2,r0=1,A=x,Nt=Nt)
            #places alpha in to big alpha matrix
            Big_Alpha[:,k]=A1.reshape((Nt+1,))
            #maintain indexing
            k=k+1
            #calulates variance of alpha after skipping x iterations keeping rolling sum of variance for each value of A
            w=np.var(Big_Alpha[nskip:Nt,:],axis=0)+w
            #
    w*(1/Samples)
    #finds A star value for which alpha varies most
    A_index=np.where(w == np.amax(w))
    #converts into scalar
    max_A=np.asscalar(A[A_index])


if __name__ == '__main__':
    #The code here should call analyze and
    #generate the figures that you are submitting with your
    #discussion.


    #function defined to construct first figure
    def figures1(A1,N1,Nt1):
        """Part 2:
        Input:
        A1-matrix giving you different values for A
        N1-number of particlies for each simulations
        Nt1: number of time steps

        Output:
        X,Y: position of all N particles at Nt+1 times
        alpha: alignment parameter at Nt+1 times

        uses simulate2 for data with default values of L=4,s0=0.2,r0=1.
        """
        #dummy variable to keep track of indexing in for loop
        k=0
        #premakes matrix which shows how alpha changes with time
        Big_Alpha=np.zeros((Nt1+1,len(A1)))
        #creates a dummy time vector for graph use
        t=np.arange(Nt1+1)
        for x in A1:
            #generates data from simulate2
            A1= simulate2(N=N1,L=4,s0=0.2,r0=1,A=x,Nt=Nt1)
            #also stores aplha data from a simulation for a given value of A, also reshapes data to keep plt happy
            Big_Alpha[:,k]=A1.reshape((Nt1+1,))
            #creates graph data, with legen
            plt.plot(t, Big_Alpha[:,k],label='A=%1.1f' %x)
            k=k+1
            plt.xlabel('Time Steps')
            plt.ylabel('Degree of Alignment')
            plt.legend(loc='center right')
            plt.title('How varying A effects Alignment')
            plt.ylim(0,1.1)
            plt.show()
        return


        #generates the first figure
        A=np.arange(0.2,0.8,0.2)
        figures1(A1=A,N1=16,Nt1=300)
        #gerates 2nd figure
        A=np.arange(0.2,0.8,0.1)
        figures1(A1=A,N1=32,Nt1=300)

        A=np.arange(0.2,0.8,0.01)
        w,A_max1=Analyse(A=A,N=16,Nt=500,Samples=120,nskip=100)
        #generates the 3rd figure,N=16
        plt.plot(A,w)
        plt.xlabel('Value of A')
        plt.ylabel('Average variance of alpha')
        plt.title('How A effects the variation of Alignment N=16, S=100')
        plt.text(0.6, 10, r'A*=%1.2f' %A_max1)
        plt.show()

        A=np.arange(0.2,0.8,0.01)
        w,A_max=Analyse(A=A,N=32,Nt=500,Samples=120,nskip=100)
        #generates the 4th figure, n=32
        plt.plot(A,w)
        plt.xlabel('Value of A')
        plt.ylabel('Average variance of alpha')
        plt.title('How A effects the variation of Alignment N=32, S=100')
        plt.text(0.6, 10, r'A*=%1.2f' %A_max)
        plt.show()
