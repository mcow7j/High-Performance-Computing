!MATH96012 Project 2
!This module contains two module variables and three subroutines;
!two of these routines must be developed for this assignment.
!Module variables--
! lr_x: training images, typically n x d with n=784 and d<=15000
! lr_y: labels for training images, d-element array containing 0s and 1s
!   corresponding to images of even and odd integers, respectively.
!lr_lambda: l2-penalty parameter, should be set to be >=0.
!Module routines---
! data_init: allocate lr_x and lr_y using input variables n and d. May be used if/as needed.
! clrmodel: compute cost function and gradient using classical logistic
!   regression model (CLR) with lr_x, lr_y, and
!   fitting parameters provided as input
! mlrmodel: compute cost function and gradient using MLR model with m classes
!   and with lr_x, lr_y, and fitting parameters provided as input

module lrmodel
  implicit none
  real(kind=8), allocatable, dimension(:,:) :: lr_x
  integer, allocatable, dimension(:) :: lr_y
  real(kind=8) :: lr_l !penalty parameter

!same but for test data
  real(kind=8), allocatable, dimension(:,:) :: te_x
  integer, allocatable, dimension(:) :: te_y


contains

!---allocate lr_x and lr_y deallocating first if needed (used by p2_main)--
! ---Use if needed---
subroutine data_init(n,d,d1)
  implicit none
  integer, intent(in) :: n,d,d1
  if (allocated(lr_x)) deallocate(lr_x)
  if (allocated(lr_y)) deallocate(lr_y)
  allocate(lr_x(n,d),lr_y(d))


  if (allocated(te_x)) deallocate(te_x)
  if (allocated(te_y)) deallocate(te_y)
  allocate(te_x(n,d1),te_y(d1))
end subroutine data_init


!Compute cost function and its gradient for CLR model
!for d images (in lr_x) and d labels (in lr_y) along with the
!fitting parameters provided as input in fvec.
!The weight vector, w, corresponds to fvec(1:n) and
!the bias, b, is stored in fvec(n+1)
!Similarly, the elements of dc/dw should be stored in cgrad(1:n)
!and dc/db should be stored in cgrad(n+1)
!Note: lr_x and lr_y must be allocated and set before calling this subroutine.
subroutine clrmodel(fvec,n,d,c,cgrad)
  implicit none
  integer, intent(in) :: n,d !training data size
  real(kind=8), dimension(n+1), intent(in) :: fvec !fitting parameters
  real(kind=8), intent(out) :: c !cost
  real(kind=8), dimension(n+1), intent(out) :: cgrad !gradient of cost
  !dummy variables
  integer :: k,k1
  real(kind=8), dimension(n) :: w
  real(kind=8), dimension(d) :: a,z1
!create w matrix
  w=fvec(1:n)
  !calculates sum of w times lamda
  c=dot_product(w,w)*lr_l
  !Declare other variables as needed
  do k= 1,d
    z1(k)=dot_product(w,lr_x(1:n,k))+fvec(n+1)
    a(k)=exp(z1(k))/(1.d0+exp(z1(k)))
    !calc
  end do
  c=c-dot_product(log(a+1.0d-12),lr_y)+dot_product(log(1.d0-a+1.0d-12),(lr_y-1))
  !Add code to compute c and cgrad
  do k1=1,n
  cgrad(k1)=dot_product((a-1.d0),lr_x(k1,1:d)*lr_y)-dot_product((a),lr_x(k1,1:d)*(lr_y-1.d0))+2.d0*lr_l*fvec(k1)
  end do
  cgrad(n+1)=dot_product((a-1.d0),lr_y)-dot_product(a,(lr_y-1.d0))
end subroutine clrmodel

!calculates error by caluting a for given fvec and then predicting results
subroutine clr_test_error(fvec,n,d1,error)
  implicit none
  integer, intent(in) :: n,d1 !training data size
  real(kind=8), dimension(n+1), intent(in) :: fvec !fitting parameters
  !real(kind=8), intent(out) :: c !cost
!  real(kind=8), dimension(n+1), intent(out) :: cgrad !gradient of cost
  !dummy variables
  integer :: k
  real(kind=8), dimension(n) :: w
  real(kind=8), dimension(d1) :: a,z1,y_guess,y_cor
  real(kind=8), intent(out) :: error
!create w matrix
  w=fvec(1:n)
  !Declare other variables as needed
  do k= 1,d1
    z1(k)=dot_product(w,te_x(1:n,k))+fvec(n+1)
    a(k)=exp(z1(k))/(1.d0+exp(z1(k)))
    !calc
  end do
  !calulate what model assumes y to be by rounding it
  y_guess=a
  where (y_guess<0.5d0)
    y_guess=0.d0
  elsewhere
    y_guess=1.d0
  end where
 !find differences between model results and actual
  y_cor=y_guess-te_y
  !find abosulte value of differnece
  y_cor=y_cor*y_cor
  !calculte perectage error
  error=sum(y_cor)*100.d0/d1
end subroutine clr_test_error


!!Compute cost function and its gradient for MLR model
!for d images (in lr_x) and d labels (in lr_y) along with the
!fitting parameters provided as input in fvec. The labels are integers
! between 0 and m-1.
!fvec contains the elements of the weight matrix w and the bias vector, b
! Code has been provided below to "unpack" fvec
!The elements of dc/dw and dc/db should be stored in cgrad
!and should be "packed" in the same order that fvec was unpacked.
!Note: lr_x and lr_y must be allocated and set before calling this subroutine.
subroutine mlrmodel(fvec,n,d,m,c,cgrad)
  implicit none
  integer, intent(in) :: n,d,m !training data sizes and number of classes
  real(kind=8), dimension((m-1)*(n+1)), intent(in) :: fvec !fitting parameters
  real(kind=8), intent(out) :: c !cost
  real(kind=8), dimension((m-1)*(n+1)), intent(out) :: cgrad !gradient of cost
  integer :: i1,j1,j,k
  real(kind=8), dimension(m-1,n) :: w, wdiv
  real(kind=8), dimension(m-1) :: b, bdiv

  !Declare other variables as needed
  real(kind=8), dimension(m,d) :: z,a,zexp
  real(kind=8), dimension(m,d) :: yhat
  real(kind=8), dimension(d) :: zsum



  !unpack fitting parameters (use if needed)
  do i1=1,n
    j1 = (i1-1)*(m-1)+1
    w(:,i1) = fvec(j1:j1+m-2) !weight matrix
  end do
  b = fvec((m-1)*n+1:(m-1)*(n+1)) !bias vector

  !Add code to compute c and cgrad

  !set up z matrix with z_0=0 and all values 0 by defalut
  z=0.d0
  do k= 1,d
    z(2:m,k)=matmul(w,lr_x(1:n,k))+b
  end do

  !find exp of z matrix and sum columns, in order to calulate a matrix
  zexp=exp(z)
  zsum=sum(zexp,dim=1)

  !calulate a matrix
  do k=1,d
    a(1:m,k)=zexp(1:m,k)/(zsum(k))
  end do

!set yhat to xero, it will select select approrate a_is
  yhat=0.d0
  do i1= 1,d
    yhat(lr_y(i1),i1)=1.d0
  end do

  !calculate cost
  c=lr_l*sum(w*w)-sum(log(yhat*a+1.0d-12))

  !calulate b div vector same format as b
  bdiv=sum(((a(2:m,1:d)*yhat(2:m,1:d))-1.d0),dim=2)-sum((a(2:m,1:d)*(yhat(2:m,1:d)-1.d0)),dim=2)

  !calculate wdiv(with same diminesions as w itetrting over n and m
  do j=1,n
    do k=2,m
      wdiv(k-1,j)=2*lr_l*w(k-1,j)-sum((1.d0-a(k,1:d)*yhat(k,1:d))*lr_x(j,1:d)*yhat(k,1:d))
      wdiv(k-1,j)=wdiv(k-1,j)+sum(a(k,1:d)*(yhat(k,1:d)-1.d0)*lr_x(j,1:d)*(yhat(k,1:d)-1.d0))
    end do
  end do


  !repack cgrad from wdiv
  do i1=1,n
    j1 = (i1-1)*(m-1)+1
    cgrad(j1:j1+m-2)=wdiv(:,i1) !weight matrix
  end do
  cgrad((m-1)*n+1:(m-1)*(n+1)) =bdiv


end subroutine mlrmodel









subroutine mlr_test_error(fvec,n,d1,m,error)
  implicit none
  integer, intent(in) :: n,d1,m !training data sizes and number of classes
  real(kind=8), dimension((m-1)*(n+1)), intent(in) :: fvec !fitting parameters
  real(kind=8), intent(out) :: error !error
  integer :: i1,j1,k
  real(kind=8), dimension(m-1,n) :: w
  real(kind=8), dimension(m-1) :: b

  !Declare other variables as needed
  real(kind=8), dimension(m,d1) :: z,a,zexp
  real(kind=8), dimension(d1) :: zsum,yguess,y_corr



  !unpack fitting parameters (use if needed)
  do i1=1,n
    j1 = (i1-1)*(m-1)+1
    w(:,i1) = fvec(j1:j1+m-2) !weight matrix
  end do
  b = fvec((m-1)*n+1:(m-1)*(n+1)) !bias vector

  !Add code to compute c and cgrad

  !set up z matrix with z_0=0 and all values 0 by defalut
  z=0.d0
  do k= 1,d1
    z(2:m,k)=matmul(w,te_x(1:n,k))+b
  end do

  !find exp of z matrix and sum columns, in order to calulate a matrix
  zexp=exp(z)
  zsum=sum(zexp,dim=1)

  !calulate a matrix
  do k=1,d1
    a(1:m,k)=zexp(1:m,k)/(zsum(k))
  end do
  !find index of max a_value in every column which corresponds to its guess of a
  yguess = maxloc(a, dim=1)-1.d0
  !find how many are correct
  y_corr=yguess-te_y
  !find absoulte value
  y_corr=y_corr*y_corr

  where (y_corr > 0.1d0)
    y_corr=0.d0
  elsewhere
    y_corr=1.d0
  end where

  error=sum(y_corr)*100.d0/d1

end subroutine mlr_test_error




end module lrmodel
