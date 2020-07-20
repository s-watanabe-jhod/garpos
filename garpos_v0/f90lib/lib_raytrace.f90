! Created:
!  07/01/2020 by S. Watanabe
!
subroutine raytrace(n, nlyr, l_dep, l_sv, dist, yd, ys, dsv, ctm, cag)
!$ use omp_lib
use subraytrace

implicit none

  ! call calc_ray_path(distance, y_deep, y_shallow, l_depth, l_sv)
  !
  !<input>
  ! n     : number of data
  ! nlyr  : number of nodes for the sound speed profile
  ! l_dep : depth at each node
  ! l_sv  : sound speed at each node
  ! dist  : horizontal distance between both ends
  ! yd    : height (depth) of deeper end (< 0)
  ! ys    : height (depth) of shallower end (< 0)
  ! dsv   : sound speed variations for each data (options, typically = 0.)
  !
  !<output>
  ! ctm   : one-way travel time
  ! cag   : takeoff angle (in rad. ; Zenith direction = pi)
  !
integer, intent(in) :: n, nlyr
real(8), intent(in) :: l_dep(0:nlyr-1), l_sv(0:nlyr-1)
real(8), intent(in) :: dist(0:n-1), yd(0:n-1), ys(0:n-1), dsv(0:n-1)
real(8), intent(inout) :: ctm(0:n-1), cag(0:n-1)
integer :: i

call omp_set_num_threads(1)
!$omp parallel do default(none) shared(yd,ys,nlyr,cag,ctm,dsv,l_dep,l_sv,dist,n) private(i)
do i = 0, n-1
  call calc_ray_path(dist(i), -yd(i), -ys(i), l_dep(0:nlyr-1), l_sv(0:nlyr-1), nlyr-1, cag(i), ctm(i))
end do
!$omp end parallel do

end
