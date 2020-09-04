!Final project part 1
!Module for flow simulations of liquid through tube
!This module contains a few module variables (see comments below)
!and four subroutines:
!jacobi: Uses jacobi iteration to compute solution
! to flow through tube
!sgisolve: To be completed. Use sgi method to
! compute flow through tube
!mvec: To be completed; matrix-vector multiplication z = Ay
!mtvec: To be completed; matrix-vector multiplication z = A^T y
module flow
    implicit none
    real(kind=8), parameter :: pi = acos(-1.d0)
    integer :: numthreads !number of threads used in parallel regions
    integer :: fl_kmax=100000 !max number of iterations
    real(kind=8) :: fl_tol=0.00000001d0 !convergence criterion
    real(kind=8), allocatable, dimension(:) :: fl_deltaw !|max change in w| each iteration
    real(kind=8) :: fl_s0=0.1d0 !deformation magnitude

contains
!-----------------------------------------------------
!Solve 2-d tube flow problem with Jacobi iteration
subroutine jacobi(n,w)
    !input  n: number of grid points (n+2 x n+2) grid
    !output w: final velocity field
    !Should also compute fl_deltaw(k): max(|w^k - w^k-1|)
    !A number of module variables can be set in advance.

    integer, intent(in) :: n
    real(kind=8), dimension(0:n+1,0:n+1), intent(out) :: w
    integer :: i1,j1,k1
    real(kind=8) :: del_r,del_t,del_r2,del_t2
    real(kind=8), dimension(0:n+1) :: s_bc,fac_bc
    real(kind=8), dimension(0:n+1,0:n+1) :: r,r2,t,RHS,w0,wnew,fac,fac2,facp,facm

    if (allocated(fl_deltaw)) then
      deallocate(fl_deltaw)
    end if
    allocate(fl_deltaw(fl_kmax))


    !grid--------------
    del_t = 0.5d0*pi/dble(n+1)
    del_r = 1.d0/dble(n+1)
    del_r2 = del_r**2
    del_t2 = del_t**2


    do i1=0,n+1
        r(i1,:) = i1*del_r
    end do

    do j1=0,n+1
        t(:,j1) = j1*del_t
    end do
    !-------------------

    !Update-equation factors------
    r2 = r**2
    fac = 0.5d0/(r2*del_t2 + del_r2)
    facp = r2*del_t2*fac*(1.d0+0.5d0*del_r/r) !alpha_p/gamma
    facm = r2*del_t2*fac*(1.d0-0.5d0*del_r/r) !alpha_m/gamma
    fac2 = del_r2 * fac !beta/gamma
    RHS = fac*(r2*del_r2*del_t2) !1/gamma
    !----------------------------

    !set initial condition/boundary deformation
    w0 = (1.d0-r2)/4.d0
    w = w0
    wnew = w0
    s_bc = fl_s0*exp(-10.d0*((t(0,:)-pi/2.d0)**2))/del_r
    fac_bc = s_bc/(1.d0+s_bc)


    !Jacobi iteration
    do k1=1,fl_kmax
        wnew(1:n,1:n) = RHS(1:n,1:n) + w(2:n+1,1:n)*facp(1:n,1:n) + w(0:n-1,1:n)*facm(1:n,1:n) + &
                                         (w(1:n,0:n-1) + w(1:n,2:n+1))*fac2(1:n,1:n)

        !Apply boundary conditions
        wnew(:,0) = wnew(:,1) !theta=0
        wnew(:,n+1) = wnew(:,n) !theta=pi/2
        wnew(0,:) = wnew(1,:) !r=0
        wnew(n+1,:) = wnew(n,:)*fac_bc !r=1s

        fl_deltaw(k1) = maxval(abs(wnew-w)) !compute relative error

        w=wnew    !update variable
        if (fl_deltaw(k1)<fl_tol) exit !check convergence criterion
        if (mod(k1,1000)==0) print *, k1,fl_deltaw(k1)
    end do


  !  print *, 'k,error=',k1,fl_deltaw(min(k1,fl_kmax))

end subroutine jacobi
!-----------------------------------------------------

!Solve 2-d tube flow problem with sgi method
subroutine sgisolve(n,w)
  !input  n: number of grid points (n+2 x n+2) grid
  !output w: final velocity field stored in a column vector
  !Should also compute fl_deltaw(k): max(|w^k - w^k-1|)
  !A number of module variables can be set in advance.
  use omp_lib
  integer, intent(in) :: n
  real(kind=8), dimension((n+2)*(n+2)), intent(out) :: w
  real(kind=8) :: del_t,del_r
  !add other variables as needed
  integer :: i1,j1
  real(kind=8), dimension((n+2)*(n+2)) :: x_old,x_new,d,e,b
  real(kind=8) :: n1,step_size,meu,w_tol
  real(kind=8) :: del_r2,del_t2,e_newsum,e_oldsum,Ad_sum
  real(kind=8), dimension(n+2) :: s_bc,fac_bc,r,t,fac,fac2,facp,facm,r2,RHS
  real(kind=8), dimension((n+2)*(n+2)) ::Ad,Md

  if (allocated(fl_deltaw)) then
    deallocate(fl_deltaw)
  end if
  allocate(fl_deltaw(fl_kmax))

  !number of threads
    !$ call omp_set_num_threads(numthreads)

    del_t = 0.5d0*pi/dble(n+1)
    del_r = 1.d0/dble(n+1)
    del_r2 = del_r**2
    del_t2 = del_t**2


    do i1=0,n+1
        r(i1+1) = i1*del_r
        t(i1+1) = i1*del_t
    end do

    !-------------------

    !Update-equation factors------
    r2 = r**2
    fac = 0.5d0/(r2*del_t2 + del_r2)
    facp = r2*del_t2*fac*(1.d0+0.5d0*del_r/r) !alpha_p/gamma
    facm = r2*del_t2*fac*(1.d0-0.5d0*del_r/r) !alpha_m/gamma
    fac2 = del_r2 * fac !beta/gamma
    RHS = fac*(r2*del_r2*del_t2) !1/gamma

    s_bc = fl_s0*exp(-10.d0*((t(:)-pi/2.d0)**2))/del_r
    fac_bc = s_bc/(1.d0+s_bc)

  !set dummy variables
  n1=(n+2)*(n+2)
  !work out b
  b=0.d0
  !$OMP parallel do private(j1)
  do i1=1,n
    do j1=2,n+1
      b(i1*(n+2)+j1)=-RHS(i1+1)
    end do
  end do
  !$OMP end parallel do

  call mtvec(n,fac,fac2,facp,facm,fac_bc,b,d)

!set intial steps in sig solve
  x_old=0.d0
  e=d
  e_oldsum=0.d0

  !$OMP parallel do reduction(+:e_oldsum)
  do j1=1,(n+2)*(n+2)
    e_oldsum=(e(j1))**2+e_oldsum
  end do
  !$OMP end parallel do
  w_tol=fl_tol+1
  i1=0
  !iterate until max tol is reached
  do while  (w_tol>fl_tol)
    !work outstep size
    call mvec(n,fac,fac2,facp,facm,fac_bc,d,Ad)

    !workout dot product of Ad
    Ad_sum=0.d0
    !$OMP parallel do reduction(+:Ad_sum)
    do j1=1,(n+2)*(n+2)
      Ad_sum=(Ad(j1))**2+Ad_sum
    end do
    !$OMP end parallel do

    !workout step size
    step_size=e_oldsum/Ad_sum
    !work estimated solution
    x_new=x_old+step_size*d

    !work out resdual by using mvec result above and mtvec
    call mtvec(n,fac,fac2,facp,facm,fac_bc,Ad,Md)

    !work out new residual
    e=e-step_size*Md

    !work magnitude of new search direction
    e_newsum=0.d0
    !$OMP parallel do reduction(+:e_newsum)
    do j1=1,(n+2)*(n+2)
      e_newsum=(e(j1))**2+e_newsum
    end do
    !$OMP end parallel do

    meu=e_newsum/e_oldsum
    !work out new d
    d=e+meu*d

    !work out max change in iteration
    w_tol=maxval(abs(x_old-x_new))
    !update variables
    x_old=x_new
    e_oldsum=e_newsum
    !record what time step we are on
    i1=i1+1
    !store tolerence
    fl_deltaw(i1)=w_tol

    !have a max number of iterations
    if (i1>fl_kmax) then
      w_tol=0
      print *, "failed to converge"
    end if

  end do

!output result
  w=x_new


end subroutine sgisolve


!Compute matrix-vector multiplication, z = Ay
subroutine mvec(n,fac,fac2,facp,facm,fac_bc,y,z)
    !input n: grid is (n+2) x (n+2)
    ! fac,fac2,facp,facm,fac_bc: arrays that appear in
    !   discretized equations
    ! y: vector multipled by A
    !output z: result of multiplication Ay
    use omp_lib
    implicit none
    integer, intent(in) :: n
    integer :: i1,j1,n1,n2
    real(kind=8), dimension(n+2), intent(in) :: fac,fac2,facp,facm,fac_bc
    real(kind=8), dimension((n+2)*(n+2)), intent(in) :: y
    real(kind=8), dimension((n+2)*(n+2)), intent(out) :: z
    !$ call omp_set_num_threads(numthreads)

    !add other variables as needed
    n1=(n+2)*(n+2)

    do i1=1,n+2
      !boundary condtion for r =1
      !should be other wat round
      z(i1)=y(i1+n+2)-y(i1)
      !boundary at centre
      z((n+2)*(n+1)+i1)=y((n+2)*(n+1)+i1)-fac_bc(i1)*y((n+2)*(n)+i1)
    end do

    !$OMP parallel do private(n2,j1)
    do i1=1,n
      !dummy variable to improve effieceeny
      n2=i1*(n+2)+1
      !boundary condition theta 1
      z(n2)=y(n2+1)-y(n2)
      !boundary condition theta 2
      z(n2+n+1)=y(n2+n+1)-y(n2+n)
      do j1=1,n
        z(j1+n2)=-y(j1+n2)+fac2(i1+1)*(y(j1-1+n2)+y(j1+1+n2))+facm(i1+1)*y(j1+n2-n-2)+facp(i1+1)*y(j1+n2+n+2)
      end do
    end do
    !$OMP end parallel do
end subroutine mvec


!Compute matrix-vector multiplication, z = A^T y
subroutine mtvec(n,fac,fac2,facp,facm,fac_bc,y,z)
    !input n: grid is (n+2) x (n+2)
    ! fac,fac2,facp,facm,fac_bc: arrays that appear in
    !   discretized equations
    ! y: vector multipled by A^T
    !output z: result of multiplication A^T y
    use omp_lib
    implicit none
    integer, intent(in) :: n
    integer :: i1,j1,n1,n2,n3
    real(kind=8), dimension(n+2), intent(in) :: fac,fac2,facp,facm,fac_bc
    real(kind=8), dimension((n+2)*(n+2)), intent(in) :: y
    real(kind=8), dimension((n+2)*(n+2)), intent(out) :: z
    !$ call omp_set_num_threads(numthreads)


    !add other variables as needed
    n1=(n+2)*(n+2)
    n3=(n+2)*(n+1)
    !set first and last element of first sub matrix
    z(1)=-y(1)
    z(n+2)=-y(n+2)

    !work out first n+2 rows
    do i1=2,n+1
      !boundary condtion for r =1
      z(i1)=facm(2)*y(i1+n+2)-y(i1)
    end do


    !dummy variable to improve effieceeny
    n2=(n+2)+1
    !boundary condition theta 1
    z(n2)=fac2(2)*y(n2+1)-y(n2)+y(n2-n-2)
    !first row of sub diagonal matrix effect by boundary condition 1
    z(n2+1)=y(n2)-y(n2+1)+fac2(2)*y(n2+2)+facm(3)*y(n2+n+3)+y(n2-n-1)
    !boundary condition theta 2 last row of sub matrix
    z(n2+n+1)=y(n2+n+1)+fac2(2)*y(n2+n)+y(n2-1)

    do j1=2,n-1
      z(j1+n2)=-y(j1+n2)+fac2(2)*(y(j1-1+n2)+y(j1+1+n2))+facm(3)*y(j1+n2+n+2)+y(j1+n2-n-2)
    end do

    !2nd to last row of sub diagonal matrix effect by boundary condition 2
    z(n2+n)=fac2(2)*y(n2+n-1)-y(n2+n)-y(n2+n+1)+facm(3)*y(n2+n+n+2)+y(n2-2)


    !$OMP parallel do private(j1,n2)
    do i1=2,n-1
      !dummy variable to improve effieceeny
      n2=i1*(n+2)+1
      !boundary condition theta 1
      z(n2)=fac2(i1+1)*y(n2+1)-y(n2)
      !first row of sub diagonal matrix effect by boundary condition 1
      z(n2+1)=y(n2)-y(n2+1)+fac2(i1+1)*y(n2+2)+facm(i1+2)*y(n2+n+3)+facp(i1)*y(n2-n-1)
      !boundary condition theta 2 last row of sub matrix
      z(n2+n+1)=y(n2+n+1)+fac2(i1+1)*y(n2+n)

      do j1=2,n-1
        z(j1+n2)=-y(j1+n2)+fac2(i1+1)*(y(j1-1+n2)+y(j1+1+n2))+facm(i1+2)*y(j1+n2+n+2)+facp(i1)*y(j1+n2-n-2)
      end do
      !2nd to last row of sub diagonal matrix effect by boundary condition 2
      z(n2+n)=fac2(i1+1)*y(n2+n-1)-y(n2+n)-y(n2+n+1)+facm(i1+2)*y(n2+n+n+2)+facp(i1)*y(n2-2)
    end do
    !$OMP end parallel do

!!!!Calculting 2nd to last submatric
    !dummy variable to improve effieceeny
    n2=n*(n+2)+1
    !boundary condition theta 1
    z(n2)=fac2(n+1)*y(n2+1)-y(n2)-fac_bc(1)*y(n2+n+2)
    !first row of sub diagonal matrix effect by boundary condition 1
    z(n2+1)=y(n2)-y(n2+1)+fac2(n+1)*y(n2+2)-fac_bc(2)*y(n2+n+3)+facp(n)*y(n2-n-1)
    !boundary condition theta 2 last row of sub matrix
    z(n2+n+1)=y(n2+n+1)+fac2(n+1)*y(n2+n)-fac_bc(n+2)*y(n2+n+n+3)

    do j1=2,n-1
      z(j1+n2)=-y(j1+n2)+fac2(n+1)*(y(j1-1+n2)+y(j1+1+n2))-fac_bc(j1+1)*y(j1+n2+n+2)+facp(n)*y(j1+n2-n-2)
    end do

    !2nd to last row of sub diagonal matrix effect by boundary condition 2
    z(n2+n)=fac2(n+1)*y(n2+n-1)-y(n2+n)-y(n2+n+1)-fac_bc(n+1)*y(n2+n+n+2)+facp(n)*y(n2-2)

      !last sub matrix
    do i1=2,n+1
      !boundary at centre
      z((n+2)*(n+1)+i1)=y((n+2)*(n+1)+i1)+facp(n+1)*y((n+2)*(n+1)+i1-n-2)
    end do
      !first and last entry of last sub matrix
      z((n+2)*(n+1)+1)=y((n+2)*(n+1)+1)
      z((n+2)*(n+2))=y((n+2)*(n+2))
    end subroutine mtvec



end module flow
