module subraytrace
! Created by T. Ishikawa
! Modified by S. Watanabe

implicit none
private
public calc_ray_path ! only this subroutine can be called

contains

!*****************************************************************************
subroutine calc_ray_path(distance, y_d, y_s, l_depth, l_sv, nlyr, t_ag, t_tm)
  !------------------------------------------------------
  ! calculate ray path and travel time (one-way)
  !
  !<input>
  ! distance: horizontal distance between both ends
  ! y_d     : height (depth) of deeper end (< 0)
  ! y_s     : height (depth) of shallower end (< 0)
  ! l_depth : depth at each node
  ! l_sv    : sound speed at each node
  ! nlyr    : number of node for sound speed profile
  !
  !<output>
  ! t_ag : takeoff angle (in rad. ; Zenith direction = pi)
  ! t_tm : one-way travel time
  !------------------------------------------------------

  ! Setting paramters
  integer, parameter :: loop1 = 200, loop2 = 20
  real(8), parameter :: eps1 = 1.d-7, eps2 = 1.0d-14 ! conv. criteria
  real(8), parameter :: pi = 4.0d0*datan(1.0d0)

  ! input/output parameters
  real(8), intent(in)  :: y_d, y_s, distance
  real(8), intent(in)  :: l_depth(0:nlyr), l_sv(0:nlyr)
  integer, intent(in)  :: nlyr
  real(8), intent(out) :: t_ag, t_tm

  integer :: i, j, k, r_nm
  real(8) :: t0, diff1, diff2, x0, x1, x2, x_diff, t_angle0, t_angle1, t_angle2
  real(8) :: t_angle
  real(8) :: a0, diff_true0, delta_angle, diff_true1
  real(8) :: ta_rough(6), x_hori(6), tadeg(6)

  integer :: layer_d, layer_s
  real(8) :: sv_d, sv_s
  real(8) :: layer_sv_trend(1:nlyr), layer_thickness(1:nlyr)

  ! ==========================================================================
  ! setup gradient and thickness of sound speed layeres
  layer_thickness(1:nlyr) = l_depth(1:nlyr) - l_depth(0:nlyr-1)
  layer_sv_trend(1:nlyr) = (l_sv(1:nlyr) - l_sv(0:nlyr-1)) / layer_thickness(1:nlyr)

  ! ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  ! setup layer_numbers and sound speed at both ends
  call layer_setting(nlyr, y_d, l_depth(0:nlyr), l_sv(0:nlyr), layer_sv_trend(1:nlyr), sv_d, layer_d)
  call layer_setting(nlyr, y_s, l_depth(0:nlyr), l_sv(0:nlyr), layer_sv_trend(1:nlyr), sv_s, layer_s)

  x_hori   = 0.0D0
  r_nm = 0
  tadeg = (/ 0.d0, 20.d0, 40.d0, 60.d0, 70.d0, 70.d0 /)
  ta_rough = pi * (180.d0 - tadeg) / 180.d0

  ! Rough scan for take-off angle
  do i = 1, 5

    if(i == 5) exit

    j = i + 1
    ! calculate horizontal distance (x_hori) for given takeoff angle (ta_rough(j))
    call ray_path(ta_rough(j), nlyr, layer_d, layer_s, sv_d, sv_s, y_d, y_s, l_depth(0:nlyr), l_sv(0:nlyr), x_hori(j), a0)

    ! negative distance does not make sense
    if(x_hori(j) < 0.0D0) cycle

    ! Only if x_hori(i) <= distance <= x_hori(i+1),
    ! takeoff angle should be between ta_rough(i) and ta_rough(i+1)
    diff1 = distance - x_hori(i)
    diff2 = x_hori(j) - distance
    if (diff1*diff2 < 0.0D0) cycle
    r_nm = 1

    ! rename
    x1 = x_hori(i)
    x2 = x_hori(j)
    t_angle1 = ta_rough(i)
    t_angle2 = ta_rough(j)

    ! detailed search for takeoff angle (conv. criteria is "eps1")
    do k = 1, loop1
        x_diff = x1 - x2 ! horizontal diff. of 2 paths
        if(dabs(x_diff) < eps1) exit

        ! calculate horizontal distance (x0) for averaged takeoff angle (t_angle0)
        !  x0 should be x1 or x2 (depending on the sign of x0-distance)
        t_angle0 = (t_angle1 + t_angle2)/2.0D0
        call ray_path(t_angle0, nlyr, layer_d, layer_s, sv_d, sv_s, y_d, y_s, l_depth(0:nlyr), l_sv(0:nlyr), x0, a0)

        a0 = -a0
        t0 = distance - x0
        if(t0*diff1 > 0.0D0)then
           x1 = x0
           t_angle1 = t_angle0
        else
           x2 = x0
           t_angle2 = t_angle0
        endif
    enddo

    diff_true0 = dabs((distance - x0)/distance)

    do k = 1, loop2
        !
        ! delta_angle is the angle for arc-length of (x0-distance)
        delta_angle = (distance - x0)/a0
        t_angle = t_angle0 + delta_angle

        ! check convergence ("eps2")
        if(dabs(delta_angle) < eps2) exit

        call ray_path(t_angle, nlyr, layer_d, layer_s, sv_d, sv_s, y_d, y_s, l_depth(0:nlyr), l_sv(0:nlyr), x0, a0)
        a0 = -a0

        diff_true1 = dabs((distance - x0)/distance)
        if(diff_true0 <= diff_true1)then
           t_angle = t_angle0
           exit
        endif
        ! check convergence ("eps2")
        if(diff_true1 < eps2) exit
    enddo
    exit
  enddo

  if(r_nm == 0) then
   print*, "Distance: ", distance
   print*, "X_hori: ", x_hori
   print*, "TA_rough: ", ta_rough
   print*, "Depth, Sound Speed: "
   do k = 0, nlyr
     print*, l_depth(k), l_sv(k)
   end do
  stop
  endif

  ! After the determination of ray path, calculate the travel time
  t_ag = t_angle
  t_tm = calc_travel_time(t_angle,nlyr,layer_d,layer_s,sv_d,sv_s,y_d,y_s,&
   l_depth(0:nlyr),l_sv(0:nlyr),layer_sv_trend(1:nlyr),layer_thickness(1:nlyr))

end subroutine calc_ray_path

!*****************************************************************************
subroutine layer_setting(nlyr, depth, l_depth, l_sv, l_sv_trend, sv, layer_n)
  ! extract sound speed at "depth" and layer number "layer_n"
  !  (l_depth(layer_n-1) < depth <= l_depth(layer_n))
  ! "sv" is sound speed at "depth"
  integer :: i
  real(8), intent(in)  :: depth, l_depth(0:nlyr), l_sv(0:nlyr), l_sv_trend(1:nlyr)
  integer, intent(in)  :: nlyr
  real(8), intent(out) :: sv
  integer, intent(out) :: layer_n
  !***************************************************
  do i = 1, nlyr
    if(l_depth(i) >= depth) then
        layer_n = i
        exit
     endif
  enddo
  if(i==nlyr+1) print*,"b", depth, l_depth
  sv = l_sv(layer_n) + (depth - l_depth(layer_n))*l_sv_trend(layer_n)

end subroutine layer_setting

!*******************************************************
subroutine ray_path(t_angle, nlyr, layer_d, layer_s, sv_d, sv_s, y_d, y_s, l_dep, l_sv, x, ray_length)
  ! Calculation for ray path
  !output:
  ! x: horizontal distance
  ! ray_length : length of ray path
  !------------------------------------------------------

  real(8), intent(in)  :: t_angle
  integer, intent(in)  :: layer_d, layer_s
  integer, intent(in)  :: nlyr
  real(8), intent(in)  :: sv_d, sv_s
  real(8), intent(in)  :: y_d, y_s
  real(8), intent(in)  :: l_dep(0:nlyr), l_sv(0:nlyr)
  real(8), intent(out) :: x, ray_length

  integer :: i, j, k1, lm
  real(8) :: pp, dx
  real(8), allocatable :: sn(:), scn(:), rn(:), yd(:)

  ! IF id=0, y_d TO y_s
  ! IF id=1, y_s TO y_d. AT y_d, RAY IS HORIZONTAL

  ! initialize
  x = 0.0d0
  ray_length = 0.0d0

  ! PP = sin(theta)/SV_D
  pp = dsin(t_angle)/sv_d

  if(y_d == y_s) then ! if the depths are the same
     x=-1.0d0
     print *, "Warning A in ray_path"
     return
  endif

  k1 = layer_s - 1
  lm = layer_d ! layer for the deeper end

  ! Layer structure>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
  !
  ! Depth                Sound speed                       Layer index
  ! layer_depth(k1)      layer_sv(k1)--------------------- k1(= layer_s - 1)
  !
  ! y_s                  sv_s            <depth for the shallower end>
  !
  ! layer_depth(layer_s) layer_sv(layer_s)---------------- layer_s
  !
  ! layer_depth(k?)      leyse_sv(k?)--------------------- layer_?
  !
  ! layer_depth(k?)      layer_sv(k?)--------------------- layer_?
  !
  !
  ! y_s                  sv_d            <depth for the deeper end>
  !
  !
  ! layer_depth(layer_d) layer_sv(layer_d)---------------- layer_d(= lm)
  ! >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

  allocate(sn(k1:layer_d), scn(k1:layer_d), rn(k1:layer_d), yd(k1:layer_d))
  sn(k1:layer_d) = pp*l_sv(k1:layer_d) ! angles of ray for each layer
  yd(k1:layer_d) = l_dep(k1:layer_d) ! depths for each node

  yd(k1) = y_s
  sn(k1) = pp*sv_s
  yd(layer_d) = y_d
  sn(layer_d) = pp*sv_d

  if(maxval(sn(k1:layer_d)) > 1.0d0 .or. minval(sn(k1:layer_d)) < -1.0d0) then
    !print *, "Warning B in ray_path"
    x = -1.0d0 ! error returns (x = -1)
    return
  endif

  scn(k1:layer_d) = dsqrt(1.0d0 - sn(k1:layer_d)**2.d0)
  rn(k1:layer_d-1) = scn(lm)/scn(k1:layer_d-1)
  rn(layer_d) = 1.d0

  ! loop from layer_s to layer_d
  do i = layer_s, layer_d
     j = i - 1
     ! horizontal distance for each layer
     dx = (yd(i) - yd(j))*(sn(i) + sn(j))/(scn(i) + scn(j))
     x = x + dx
     ray_length = ray_length + rn(i)*dx/scn(j)
  enddo
  ray_length = ray_length/sn(lm)

end subroutine ray_path

!*****************************************************************************
function calc_travel_time(t_angle, nl, layer_d, layer_s, sv_d, sv_s, y_d, y_s, l_dep, l_sv, l_sv_trend, l_th) result(travel_time)
!------------------------------------------------------
  real(8), intent(in)  :: t_angle
  integer, intent(in)  :: nl, layer_d, layer_s
  real(8), intent(in)  :: sv_d, sv_s
  real(8), intent(in)  :: y_d, y_s
  real(8), intent(in)  :: l_dep(0:nl), l_sv(0:nl), l_sv_trend(1:nl), l_th(1:nl)
  real(8) :: travel_time

  integer, parameter :: lmax = 55
  real(8), parameter :: epg = 1.0d-2
  real(8), parameter :: eang = 80.0*3.14159265358979d0/180.0d0
  real(8), parameter :: epr = 1.0d-12

  integer :: k1, ls, i, j
  real(8) :: epa, aatra, bbtra, cctra, d2, d1, tmp, tmpdd
  real(8) :: sn1, sn2, snm, cn1, cn2, tinc
  real(8) :: xa, za, xx, zz, xxc, zzc
  real(8) :: pp
  real(8), allocatable :: vn(:), tn(:)
  !------------------------------------------------------

  epa = dsin(eang)
  travel_time = 0.0D0
  pp = dsin(t_angle)/sv_d
  tinc = 0.d0

  if(y_d /= y_s)then
     k1 = layer_s-1
     allocate(vn(0:layer_d), tn(0:layer_d))

     vn(k1+1:layer_d-1) = l_sv(k1+1:layer_d-1)
     vn(k1)      = sv_s
     vn(layer_d) = sv_d

     tn(layer_s:layer_d) = l_th(layer_s:layer_d)
     tn(layer_s) = tn(layer_s) - (y_s - l_dep(k1))
     tn(layer_d) = tn(layer_d) - (l_dep(layer_d) - y_d)

     do i = layer_s, layer_d
        j = i - 1
        sn1 = pp*vn(i)
        cn1 = dsqrt(1.0d0 - sn1**2.d0)
        sn2 = pp*vn(j)
        cn2 = dsqrt(1.0d0 - sn2**2.d0)

        if(dabs(l_sv_trend(i)) > epg) then
           tinc = (dlog((1.0d0 + cn2)/(1.0d0 + cn1)) + dlog(vn(i)/vn(j)))/l_sv_trend(i)
        elseif(dabs(l_sv_trend(i)) <= epg) then
           snm = dmin1(sn1, sn2)

           if(snm > epa) then
              aatra = 1.0d0
              bbtra = 1.0d0
              cctra = cn1*(cn2+cn1)
              d2 = cn2**2.d0
              d1 = cn1**2.d0
              tmp = 0.0d0
              ls = 1

              do
                 tmpdd = aatra/dfloat(ls)
                 tmp = tmp + tmpdd
                 if(tmpdd >= epr .and. ls <= lmax) then
                    aatra = aatra*d2 + bbtra*cctra
                    bbtra = bbtra*d1
                    ls = ls + 2
                    cycle
                 else
                    tinc = tn(i)*tmp*pp*(sn1 + sn2)/(cn1 + cn2)
                    exit
                 endif
              enddo

           elseif(snm <= epa) then
              zzc = pp*(sn1 + sn2)/(1.0d0 + cn1)/(cn1 + cn2)
              xxc = 1.0d0/vn(i)
              zz = (cn1 - cn2)/(1.0d0 + cn1)
              xx = xxc*(vn(i) - vn(j))
              za = 1.0D0
              xa = 1.0D0
              tmp = 0.0D0
              ls = 1

              do
                 tmpdd = (za*zzc + xa*xxc)/dfloat(ls)
                 tmp = tmp + tmpdd
                 if(tmpdd >= epr .and. ls <= lmax) then
                    za = za*zz
                    xa = xa*xx
                    ls = ls + 1
                    cycle
                 else
                    tinc = tn(i)*tmp
                    exit
                 endif
              enddo

           endif

        endif

        travel_time = travel_time + tinc
     enddo

     deallocate(vn,tn)
  endif

end function calc_travel_time

end
