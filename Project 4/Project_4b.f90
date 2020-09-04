!Final project part 2
!This file contains 1 module, 1 main program and 4 subroutines:
! params: module contain problem parameters and useful constants
! crickets: main program which reads in parameters from data.in
! 	calls simulate, and writes the computed results to output files
! simulate: subroutine for simulating coupled oscillator model
! 	using explicit-Euler time-marching and distributed-memory
! 	parallelization
! RHS: subroutine called by simulate, generates right-hand side
!		of oscillator model equations
! MPE_DECOMP1D: subroutine called by simulate and used to assign
!		oscillators to processes
! random_normal: subroutine called by main program and used to generate
!		natural frequencies, w
!
! to compile and run use following code
! mpif90 -o p42.exe p42.f90
! mpiexec -n 4 p42.exe

!-------------------------------------------------------------
module params
	implicit none
	real(kind=8), parameter :: pi = acos(-1.d0)
	complex(kind=8), parameter :: ii=cmplx(0.0,1.0) !ii = sqrt(-1)
    integer :: ntotal !total number of oscillators,
	real(kind=8) :: c,mu,sigma,coeff !coupling coefficient, mean, std for computing omega
	integer :: nlocal_min
	integer :: nlocal,nfsize
	integer, allocatable, dimension(:) :: ai_copy
	save
end module params
!-------------------------------

program crickets
    use mpi
    use params
    implicit none
    integer :: i1,j1
    integer :: nt !number of time steps
    real(kind=8) :: dt!time step
    integer :: myid, numprocs, ierr, istart, iend
    real(kind=8), allocatable, dimension(:) :: f0,w,f ! initial phases, frequencies, final phases
		real(kind=8), allocatable, dimension(:) :: r !synchronization parameter

 ! Initialize MPI
    call MPI_INIT(ierr)
    call MPI_COMM_SIZE(MPI_COMM_WORLD, numprocs, ierr)

!gather input
    open(unit=10,file='data.in')
        read(10,*) ntotal !total number of oscillators
        read(10,*) nt !number of time steps
        read(10,*) dt !size of time step
        read(10,*) c ! coupling parameter
        read(10,*) sigma !standard deviation for omega calculation
    close(10)

    allocate(f0(ntotal),f(ntotal),w(ntotal),r(nt))


!generate initial phases
    call random_number(f0)
    f0 = f0*2.d0*pi


!generate frequencies
    mu = 1.d0
    call random_normal(ntotal,w)
    w = sigma*w+mu

!compute min(nlocal)
		nlocal_min = ntotal
		do i1 = 0,numprocs-1
			call mpe_decomp1d(ntotal,numprocs,i1,istart,iend)
			nlocal_min = min(iend-istart+1,nlocal_min)
		end do


!compute solution
    call simulate(MPI_COMM_WORLD,numprocs,ntotal,0.d0,f0,w,dt,nt,f,r)


!output solution (after collecting solution onto process 0 in simulate)
     call MPI_COMM_RANK(MPI_COMM_WORLD, myid, ierr)
     if (myid==0) then
        open(unit=11,file='theta.dat')
        do i1=1,ntotal
            write(11,*) f(i1)
        end do
        close(11)

        open(unit=12,file='R.dat')
        do i1=1,nt
	    		write(12,*) r(i1)
				end do
				close(12)
    	end if
    !can be loaded in python, e.g. theta=np.loadtxt('theta.dat')

    call MPI_FINALIZE(ierr)
end program crickets



subroutine simulate(comm,numprocs,n,t0,y0,w,dt,nt,y,r)
    !explicit Euler method, parallelized with mpi
    !input:
    !comm: MPI communicator
    !numprocs: total number of processes
    !n: number of oscillators
    !t0: initial time
    !y0: initial phases of oscillators
    !w: array of frequencies, omega_i
    !dt: time step
    !nt: number of time steps
    !output: y, final solution
    !r: synchronization parameter at each time step
    use mpi
    use params
    implicit none
    integer, intent (in) :: n,nt
    real(kind=8), dimension(n), intent(in) :: y0,w
    real(kind=8), intent(in) :: t0,dt
    real(kind=8), dimension(n), intent(out) :: y
		real(kind=8), dimension(nt), intent(out) :: r
    real(kind=8) :: t
    integer :: i1,k,l3,istart,iend
    integer :: comm,myid,ierr,numprocs
		integer, allocatable, dimension(:) :: seed,ai,y_array_pos
		real(kind=8), allocatable, dimension(:) ::  temp
		integer :: nseed,time
		!add other variables as needed
		real(kind=8), allocatable, dimension(:) ::  f_send_above,f_send_below,fi,f_center
		integer :: amax_from_above,amax_from_below,amax_needed_by_below,amax_needed_by_above
		integer :: request,below,above
		integer, dimension(MPI_STATUS_SIZE) :: status
		complex(kind=8) :: rsum,rtemp

    call MPI_COMM_RANK(comm, myid, ierr)
    print *, 'start simulate, myid=',myid

    !set initial conditions
    y = y0
    t = t0
    !generate decomposition and allocate sub-domain variables
    call mpe_decomp1d(size(y),numprocs,myid,istart,iend)
    print *, 'istart,iend,threadID=',istart,iend,myid
		!number of ocillators in processor
		nlocal=iend-istart

		!Set coupling ranges, ai
		allocate(ai(iend-istart+1),temp(iend-istart+1),ai_copy(iend-istart+1))
		call random_seed(size=nseed)
		call system_clock(time)
		allocate(seed(nseed))
		seed = myid+time !remove the "+time" to generate same ai each run
		call random_seed(put=seed)
		call random_number(temp)
		ai = 1 + FLOOR((nlocal_min-1)*temp)
		ai_copy=ai

		!calculate coefficient making sure its double precision
		coeff=DBLE(c)/DBLE(ntotal)

		!set ids for transfering between processors
		if (myid==0) then
			below = numprocs-1
			above = 1
		elseif (myid<numprocs-1) then
			below = myid-1
			above = myid+1
		else
			below = myid-1
			above = 0
		end if

		amax_from_below=0
		amax_from_above=0
		!works out how many other theta j's this proc will need from other procs
		do i1=1,iend-istart+1
			amax_from_below=max(ai(i1)-i1+1,amax_from_below)
			amax_from_above=max(ai(iend-istart+2-i1)-i1+1,amax_from_above)
		end do


		!transfers this info to the relevent proccessors
		call MPI_ISEND(amax_from_below,1,MPI_INTEGER,below,0,MPI_COMM_WORLD,request,ierr)
		call MPI_ISEND(amax_from_above,1,MPI_INTEGER,above,0,MPI_COMM_WORLD,request,ierr)
		!recieve data
		call MPI_RECV(amax_needed_by_above,1,MPI_INTEGER,above,0,MPI_COMM_WORLD,status,ierr)
		call MPI_RECV(amax_needed_by_below,1,MPI_INTEGER,below,0,MPI_COMM_WORLD,status,ierr)
		!waits for all processors to have this info
		call MPI_BARRIER(MPI_COMM_WORLD,ierr)

		! allocating dimensions of f matrices for all time
		nfsize=nlocal+1+amax_from_below+amax_from_above
		allocate(f_center(nlocal+1))
		allocate(fi(nfsize))
		allocate(f_send_above(amax_needed_by_above))
		allocate(f_send_below(amax_needed_by_below))

		!start initial conditions extract data from y0 and placing them in y
		fi(amax_from_below+1:amax_from_below+nlocal+1)=y0(istart:iend)
		if (myid==0) then
			fi(1:amax_from_below)=y0(ntotal-amax_from_below+1:ntotal)
			fi(nfsize-amax_from_above+1:nfsize)=y0(iend+1:iend+amax_from_above)
		else if (myid==numprocs-1)then
			fi(1:amax_from_below)=y0(istart-amax_from_below:istart-1)
			fi(nfsize-amax_from_above+1:nfsize)=y0(1:amax_from_above)
		else
			fi(1:amax_from_below)=y0(istart-amax_from_below:istart-1)
			fi(nfsize-amax_from_above+1:nfsize)=y0(iend+1:iend+amax_from_above)
		end if
		!work out RHS first time step
		call RHS(amax_from_below,t,w(istart:iend),fi,f_center)
		!work out next value of theta inputting it straight into f
		fi(amax_from_below+1:amax_from_below+1+nlocal)= &
		fi(amax_from_below+1:amax_from_below+1+nlocal) + dt*f_center

		!now transfers this updated theta to other time steps
	!	call MPI_ISEND(fi(amax_from_below+1:amax_from_below+amax_needed_by_below),amax_needed_by_below,&
	!	MPI_DOUBLE_PRECISION,below,1,MPI_COMM_WORLD,request,ierr)

	!	call MPI_ISEND(fi(nlocal+2-amax_needed_by_above+amax_from_below:&
	!	nlocal+1+amax_from_below),amax_needed_by_above,MPI_DOUBLE_PRECISION,above,1,&
!		MPI_COMM_WORLD,request,ierr)

		call MPI_ISEND(f_center(1:amax_needed_by_below),amax_needed_by_below,&
		MPI_DOUBLE_PRECISION,below,1,MPI_COMM_WORLD,request,ierr)

		call MPI_ISEND(f_center(nlocal+1-amax_needed_by_below:nlocal),amax_needed_by_above,&
		MPI_DOUBLE_PRECISION,above,1,MPI_COMM_WORLD,request,ierr)

		!waits for all processors to have this info
		call MPI_BARRIER(MPI_COMM_WORLD,ierr)

    !time marching
    do k = 1,nt-1
				!compute t for no reason
				t=t+dt
				!recieve other needed thetas
				call MPI_RECV(fi(1:amax_from_below),amax_from_below,MPI_DOUBLE_PRECISION,&
				below,k,MPI_COMM_WORLD,status,ierr)
				call MPI_RECV(fi(nfsize-amax_from_above+1:nfsize),amax_from_above,&
				MPI_DOUBLE_PRECISION,above,k,MPI_COMM_WORLD,status,ierr)

				!work out rsum for every core
				rsum=sum(exp(f_center*ii))

				if (myid==0) then
					!recive rsum from every other core
					do l3=1,numprocs-1
						!recieve rsum from all other sores
						call MPI_RECV(rtemp,1,MPI_COMPLEX,l3,l3,MPI_COMM_WORLD,status,ierr)
						!create a total rsum
						rsum=rsum+rtemp
					end do
					! place rsum in r and work out modulus of sum
					r(k)=abs(rsum)/dble(ntotal)
				else
					!send rsum local to myid==0
					call MPI_ISEND(rsum,1,MPI_COMPLEX,0,myid,MPI_COMM_WORLD,request,ierr)
				end if

				!work out next time step
				call RHS(amax_from_below,t,w(istart:iend),fi,f_center)
				!update next time step
				fi(amax_from_below+1:amax_from_below+1+nlocal)= &
				fi(amax_from_below+1:amax_from_below+1+nlocal) + dt*f_center

				if (k<(nt-1)) then
					call MPI_ISEND(f_center(1:amax_needed_by_below),&
					amax_needed_by_below,MPI_DOUBLE_PRECISION,below,k+1,MPI_COMM_WORLD,request,ierr)
					call MPI_ISEND(f_center(nlocal+1-amax_needed_by_below:nlocal)&
					,amax_needed_by_above,MPI_DOUBLE_PRECISION,above,k+1,MPI_COMM_WORLD,request,ierr)
				end if
    end do


		if (myid==0) then
			!recive rsum from every other core
			do l3=1,numprocs-1
				!recieve rsum from all other sores
				call MPI_RECV(rtemp,1,MPI_COMPLEX,l3,l3,MPI_COMM_WORLD,status,ierr)
				!create a total rsum
				rsum=rsum+rtemp
			end do
			! place rsum in r at nt and work out modulus of sum
			r(nt)=abs(rsum)/dble(ntotal)
		else
			!send rsum local to myid==0
			call MPI_ISEND(rsum,1,MPI_COMPLEX,0,myid,MPI_COMM_WORLD,request,ierr)
		end if


		!call MPI_GATHERV(Nlocal,1,MPI_INT,Nper_proc,1,MPI_INT 0,MPI_COMM_WORLD,ierr)

!disps = [0, Nper_proc(1), Nper_proc(1)+Nper_proc(2), ...]

    print *, 'before collection',myid, maxval(abs(f_center))

		allocate(y_array_pos(numprocs))

		!SENDING SIZE OF INCOMING YLOCAL ARRAY TO THEAD ZERO SO IT NOWS WHAT TO EXPECT
		if (myid==0) then
			do l3=1,numprocs-1
				!recieve size of f_center from every other core
				call MPI_RECV(y_array_pos(l3),1,MPI_INTEGER,l3,l3*10,MPI_COMM_WORLD,status,ierr)
			end do
			!make dummy last value for ease of code
			y_array_pos(numprocs)=ntotal+1
		else
			call MPI_ISEND(istart,1,MPI_INTEGER,0,myid*10,MPI_COMM_WORLD,request,ierr)
		end if


		!reciveing y could have used gatherv but prefered to use irecv instead
		if (myid==0) then
			!set y within myid=0
			y(1:iend)=f_center
			!recieve ylocal all info from all procs placing in into y
			do l3=1,numprocs-1
				!
				call MPI_RECV(y(y_array_pos(l3):y_array_pos(l3+1)-1),&
				y_array_pos(l3+1)-y_array_pos(l3),MPI_DOUBLE_PRECISION,l3,l3*10,MPI_COMM_WORLD,status,ierr)
			end do
		else
			!send ylocal if myidd!=0
			call MPI_ISEND(f_center,nlocal,MPI_DOUBLE_PRECISION,0,myid*10,MPI_COMM_WORLD,request,ierr)
		end if


		call MPI_BARRIER(MPI_COMM_WORLD,ierr)
    if (myid==0) print *, 'finished',maxval(abs(y))

end subroutine simulate
!-------------------------
subroutine RHS(nn,t,w,f2,Rpart)
    !called by simulate
    !Rpart = (1/dt)*(f(t+dt)-f(t))
    use params
    implicit none
    integer, intent(in) :: nn !nn is used to calculate where f_center is in f2, could have done this using gloabl params as well
    real(kind=8), intent(in) :: t
!dimensions of variables below must be added
    real(kind=8), dimension(nlocal+1), intent(in) :: w
    real(kind=8), dimension(nfsize), intent(in) :: f2
    real(kind=8), dimension(nlocal+1), intent(out) :: Rpart
		real(kind=8) :: sin_sum
		integer :: i2,k2
		!nn is used to calculate where f_center is in f2

	!iterate over core theta is
		do i2=1,nlocal
			!how many other theta you sum over using ais
			if (ai_copy(i2)>0) then
				do k2=1,ai_copy(i2)
					sin_sum=sin_sum+sin(f2(i2+nn)-f2(i2-k2+nn))+sin(f2(i2+nn)-f2(i2+k2+nn))
				end do
			end if
			!Add code to compute rhs
			Rpart(i2)=w(i2)-coeff*sin_sum
			!reset sin sum for next j
			sin_sum=0
		end do
end subroutine RHS


!--------------------------------------------------------------------
!  (C) 2001 by Argonne National Laboratory.
!      See COPYRIGHT in online MPE documentation.
!  This file contains a routine for producing a decomposition of a 1-d array
!  when given a number of processors.  It may be used in "direct" product
!  decomposition.  The values returned assume a "global" domain in [1:n]
!
subroutine MPE_DECOMP1D( n, numprocs, myid, s, e )
    implicit none
    integer :: n, numprocs, myid, s, e
    integer :: nlocal
    integer :: deficit

    nlocal  = n / numprocs
    s       = myid * nlocal + 1
    deficit = mod(n,numprocs)
    s       = s + min(myid,deficit)
    if (myid .lt. deficit) then
        nlocal = nlocal + 1
    endif
    e = s + nlocal - 1
    if (e .gt. n .or. myid .eq. numprocs-1) e = n

end subroutine MPE_DECOMP1D

!--------------------------------------------------------------------

subroutine random_normal(n,rn)

! Adapted from the following Fortran 77 code
!      ALGORITHM 712, COLLECTED ALGORITHMS FROM ACM.
!      THIS WORK PUBLISHED IN TRANSACTIONS ON MATHEMATICAL SOFTWARE,
!      VOL. 18, NO. 4, DECEMBER, 1992, PP. 434-435.

!  The function random_normal() returns a normally distributed pseudo-random
!  number with zero mean and unit variance.

!  The algorithm uses the ratio of uniforms method of A.J. Kinderman
!  and J.F. Monahan augmented with quadratic bounding curves.

IMPLICIT NONE
integer, intent(in) :: n
real(kind=8), intent(out) :: rn(n)
!     Local variables
integer :: i1
REAL(kind=8)     :: s = 0.449871, t = -0.386595, a = 0.19600, b = 0.25472,           &
            r1 = 0.27597, r2 = 0.27846, u, v, x, y, q

!     Generate P = (u,v) uniform in rectangle enclosing acceptance region
do i1=1,n

DO
  CALL RANDOM_NUMBER(u)
  CALL RANDOM_NUMBER(v)
  v = 1.7156d0 * (v - 0.5d0)

!     Evaluate the quadratic form
  x = u - s
  y = ABS(v) - t
  q = x**2 + y*(a*y - b*x)

!     Accept P if inside inner ellipse
  IF (q < r1) EXIT
!     Reject P if outside outer ellipse
  IF (q > r2) CYCLE
!     Reject P if outside acceptance region
  IF (v**2 < -4.d0*LOG(u)*u**2) EXIT
END DO

!     Return ratio of P's coordinates as the normal deviate
rn(i1) = v/u
end do
RETURN


END subroutine random_normal
