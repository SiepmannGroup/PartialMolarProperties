MODULE energy_kspace
  use var_type,only:dp
  use const_math,only:onepi,twopi
  use const_phys,only:qqfact
  use util_runtime,only:err_exit
  use util_mp,only:mp_sum,mp_allgather,mp_set_displs
  use sim_system,only:lsolid,lrect,boxlx,boxly,boxlz,nchain,moltyp,lelect,nboxi,ntype,lqchg,rxu,ryu,rzu,qqu,myid,numprocs,moltion&
   ,nunit,rxuion,ryuion,rzuion,qquion,xcm,ycm,zcm,groupid,rcut
  use sim_cell
  implicit none
  private
  save
  public::recipsum,recip,recip_atom,ee_recip,recippress,calp,sself,correct,save_kvector,restore_kvector,allocate_kspace
  public::k_max_l, k_max_m, k_max_n, compute_kmax

  integer,parameter::vectormax=100000 !< the maximum number of reciprocal vectors for Ewald sum
  integer,allocatable::numvect(:)& !< the total number of reciprocal vectors
   ,numvecto(:),k_max_l(:),k_max_m(:),k_max_n(:)
  real,allocatable::kx(:,:),ky(:,:),kz(:,:),prefact(:,:),ssumr(:,:),ssumi(:,:),ssumrn(:,:),ssumin(:,:),ssumro(:,:),ssumio(:,:)&
   ,kxo(:,:),kyo(:,:),kzo(:,:),prefacto(:,:),calpo(:)
  real,allocatable,target::calp(:) !< calp = kalp / boxlen; kalp is a parameter to control the real space sum
  real::sself,correct

contains
!> \brief calculates the total reciprocal space ewald-sum term for volume
!> moves
!> \par History
!> written in 1998 by Bin Chen \n
!> rewritten in 2001 by Bin Chen \n
!> rewritten again, probably by Bin
  subroutine recipsum(ibox,vrecip)
    real(kind=dp)::vrecip
    integer::ibox,i,ii,imolty,ncount

    ! from h-matrix formulation
    integer::l,m,n,m_min,n_min,kmaxl,kmaxm,kmaxn

    real::alpsqr4,vol,ksqr,sumr,sumi,arg,bx1,by1,bz1,hmatik(9),kx1,ky1,kz1,hmaxsq,calpi
    ! real::sum_sumr,sum_sumi

    ! RP added for calculating time for communication step
    integer::mystart,myend,blocksize
    integer::rcounts(numprocs),displs(numprocs)
    real::my_kx(vectormax),my_ky(vectormax),my_kz(vectormax),my_ssumr(vectormax),my_ssumi(vectormax),my_prefact(vectormax)

    ! Set up the reciprocal space vectors ***
    ncount = 0
    vrecip = 0.0E0_dp

    calpi = calp(ibox)

    if ( (.not. lsolid(ibox)) .or. lrect(ibox) )  then
       bx1 = boxlx(ibox)
       by1 = boxly(ibox)
       bz1 = boxlz(ibox)
       hmat(ibox,1) = bx1
       hmat(ibox,5) = by1
       hmat(ibox,9) = bz1
       do i = 1,9
          hmatik(i) = 0.0E0_dp
       end do
       hmatik(1) = twopi/bx1
       hmatik(5) = twopi/by1
       hmatik(9) = twopi/bz1
    else
       do i = 1,9
          hmatik(i) = twopi*hmati(ibox,i)
       end do
    end if
    call compute_kmax(ibox)
    kmaxl = k_max_l(ibox)
    kmaxm = k_max_m(ibox)
    kmaxn = k_max_n(ibox)

    alpsqr4 = 4.0E0_dp*calpi*calpi

    vol = hmat(ibox,1)* (hmat(ibox,5)*hmat(ibox,9) - hmat(ibox,8)*hmat(ibox,6))&
     + hmat(ibox,4)* (hmat(ibox,8)*hmat(ibox,3) - hmat(ibox,2)*hmat(ibox,9))&
     + hmat(ibox,7)* (hmat(ibox,2)*hmat(ibox,6) - hmat(ibox,5)*hmat(ibox,3))

    vol = vol/(4.0E0_dp*onepi)

    hmaxsq = alpsqr4*calpi*rcut(ibox)*calpi*rcut(ibox)

    ! RP added for MPI
    blocksize = kmaxl/numprocs
    mystart = myid * blocksize
    if (myid .eq. (numprocs-1)) then
       myend = kmaxl
    else
       myend = (myid + 1) * blocksize - 1
    end if
    ! generate the reciprocal-space
    ! here -kmaxl,-kmaxl+1,...,-1 are skipped, so no need to divide by 2 for the prefactor
    do l = mystart,myend
       ! do l = 0,kmaxl
       if ( l .eq. 0 ) then
          m_min = 0
       else
          m_min = -kmaxm
       end if
       do m = m_min, kmaxm
          if (l .eq. 0 .and. m .eq. 0) then
             n_min = 1
          else
             n_min = -kmaxn
          end if
          do n = n_min, kmaxn
             kx1 = real(l,dp)*hmatik(1)+real(m,dp)*hmatik(2)+real(n,dp)*hmatik(3)
             ky1 = real(l,dp)*hmatik(4)+real(m,dp)*hmatik(5)+real(n,dp)*hmatik(6)
             kz1 = real(l,dp)*hmatik(7)+real(m,dp)*hmatik(8)+real(n,dp)*hmatik(9)
             ksqr = kx1*kx1+ky1*ky1+kz1*kz1
             ! if ( ksqr .lt. hmaxsq ) then
             ! sometimes these are about equal, which can cause different
             ! behavior on 32 and 64 bit machines without this .and. statement
             if ( ksqr .lt. hmaxsq .and. abs(ksqr-hmaxsq) .gt. 1E-9_dp ) then
                ncount = ncount + 1
                my_kx(ncount) = kx1
                my_ky(ncount) = ky1
                my_kz(ncount) = kz1
                my_prefact(ncount) = exp(-ksqr/alpsqr4)/(ksqr*vol)
                ! sum up q*cos and q*sin ***
                sumr = 0.0E0_dp
                sumi = 0.0E0_dp
                ! do i = myid+1,nchain,numprocs
                do i = 1,nchain
                   imolty = moltyp(i)
                   if (.not.lelect(imolty).or.nboxi(i).ne.ibox ) cycle
                   do ii = 1,nunit(imolty)
                      if ( lqchg(ntype(imolty,ii)) ) then
                         arg=kx1*rxu(i,ii)+ky1*ryu(i,ii)+kz1*rzu(i ,ii)
                         sumr = sumr + cos(arg)*qqu(i,ii)
                         sumi = sumi + sin(arg)*qqu(i,ii)
                      end if
                   end do
                end do

                my_ssumr(ncount) = sumr
                my_ssumi(ncount) = sumi
                ! Potential energy ***
                vrecip = vrecip + (sumr*sumr + sumi*sumi) * my_prefact(ncount)
             end if
          end do
       end do
    end do

    call mp_sum(vrecip,1,groupid)
    call mp_allgather(ncount,rcounts,groupid)
    call mp_set_displs(rcounts,displs,numvect(ibox),numprocs)
    if ( numvect(ibox) .gt. vectormax ) call err_exit(__FILE__,__LINE__,'choose a larger vectormax',myid+1)
    call mp_allgather(my_kx,kx(:,ibox),rcounts,displs,groupid)
    call mp_allgather(my_ky,ky(:,ibox),rcounts,displs,groupid)
    call mp_allgather(my_kz,kz(:,ibox),rcounts,displs,groupid)
    call mp_allgather(my_ssumr,ssumr(:,ibox),rcounts,displs,groupid)
    call mp_allgather(my_ssumi,ssumi(:,ibox),rcounts,displs,groupid)
    call mp_allgather(my_prefact,prefact(:,ibox),rcounts,displs,groupid)

    return
  end subroutine recipsum

!> \brief calculates the reciprocal ewald-sum term for trans, rot, flucq,
!> swatch and swap moves, and update the reciprocal ewald-sum.
!> \par History
!> rewritten on June 25/99 by Bin Chen
  subroutine recip(ibox,vrecipnew,vrecipold,type)
    integer::ic,izz,ii,imolty,ibox,ncount,type
    real::vrecipnew,vrecipold,sumr(2),sumi(2),arg

    ! RP added for MPI
    integer::rcounts(numprocs),displs(numprocs),my_start,my_end,blocksize
    real::my_ssumrn(vectormax),my_ssumin(vectormax)

! if (LSOLPAR.and.(ibox.eq.2))then
! return
! end if

    ncount = numvect(ibox)
    if ( type .eq. 1 ) then
! recalculate the reciprocal space part for one-particle move, translation,
! rotation, swap, flucq, and swatch.
! old conformation izz = 1 (which is 0 for swap inserted molecule)
! new conformation izz = 2 (which is 0 for swap removed molecule)

#ifdef __DEBUG_KSPACE__
       write(io_output,*) myid,' in recip:',moltion(1),moltion(2)
       do izz = 1,2
          imolty = moltion(izz)
          do ii = 1, nunit(imolty)
             write(io_output,*) rxuion(ii,izz),ryuion(ii,izz),rzuion(ii,izz), qquion(ii,izz)
          end do
       end do
#endif

       ! RP added for MPI
       blocksize = ncount/numprocs
       rcounts = blocksize
       blocksize = ncount - blocksize * numprocs
       if (blocksize.gt.0) rcounts(1:blocksize) = rcounts(1:blocksize) + 1
       call mp_set_displs(rcounts,displs,blocksize,numprocs)
       my_start = displs(myid+1) + 1
       my_end = my_start + rcounts(myid+1) - 1

       ! do 30 ic = 1,ncount
       do ic = my_start,my_end
          do izz = 1,2
             ! izz = 1: old configuration
             ! izz = 2: new configuration
             sumr(izz) = 0.0E0_dp
             sumi(izz) = 0.0E0_dp
             imolty = moltion(izz)
             do ii = 1, nunit(imolty)
                if ( lqchg(ntype(imolty,ii)) ) then
                   arg = kx(ic,ibox)*rxuion(ii,izz) + ky(ic,ibox)*ryuion(ii,izz) + kz(ic,ibox)*rzuion(ii,izz)
                   sumr(izz) = sumr(izz) +  qquion(ii,izz)*cos(arg)
                   sumi(izz) = sumi(izz) +  qquion(ii,izz)*sin(arg)
                end if
             end do
          end do

          ! ssumrn(ic,ibox) = ssumr(ic,ibox) - sumr(1) + sumr(2)
          ! ssumin(ic,ibox) = ssumi(ic,ibox) - sumi(1) + sumi(2)
          ! RP added for MPI
          my_ssumrn(ic-my_start + 1) = ssumr(ic,ibox) - sumr(1) + sumr(2)
          my_ssumin(ic-my_start + 1) = ssumi(ic,ibox) - sumi(1) + sumi(2)
       end do

       call mp_allgather(my_ssumrn,ssumrn(:,ibox),rcounts,displs,groupid)
       call mp_allgather(my_ssumin,ssumin(:,ibox),rcounts,displs,groupid)
!----------------------------------------------------------------------
       vrecipnew = 0.0E0_dp
       vrecipold = 0.0E0_dp
       do ic = my_start,my_end
          vrecipnew = vrecipnew + (ssumrn(ic,ibox)*ssumrn(ic,ibox) + ssumin(ic,ibox)*ssumin(ic,ibox))*prefact(ic,ibox)
          vrecipold = vrecipold + (ssumr(ic,ibox)*ssumr(ic,ibox) + ssumi(ic,ibox)*ssumi(ic,ibox))*prefact(ic,ibox)
       end do
       vrecipnew = vrecipnew*qqfact
       vrecipold = vrecipold*qqfact
       call mp_sum(vrecipnew,1,groupid)
       call mp_sum(vrecipold,1,groupid)
    else if (type .eq. 2) then
       ! update the reciprocal space k vectors
       do ic = 1, ncount
          ssumr(ic,ibox) = ssumrn(ic,ibox)
          ssumi(ic,ibox) = ssumin(ic,ibox)
       end do
    else if (type .eq. 3) then
       ! store the reciprocal space k vectors
       do ic = 1, ncount
          ssumro(ic,ibox) = ssumr(ic,ibox)
          ssumio(ic,ibox) = ssumi(ic,ibox)
       end do
    else if (type .eq. 4) then
       ! restore the reciprocal space k vectors
       do ic = 1, ncount
          ssumr(ic,ibox) = ssumro(ic,ibox)
          ssumi(ic,ibox) = ssumio(ic,ibox)
       end do
    end if

#ifdef __DEBUG_KSPACE__
    write(io_output,*) myid,' in recip:',ssumr(100,ibox),ibox,ssumrn(100,ibox)
#endif
    return
  end subroutine recip

!> \brief store old k vectors and reciprocal sum
  subroutine save_kvector(ibox)
    integer,intent(in)::ibox
    integer::ic,ncount
    calpo(ibox) = calp(ibox)
    numvecto(ibox) = numvect(ibox)
    ncount = numvect(ibox)
    do ic = 1,ncount
       kxo(ic,ibox) = kx(ic,ibox)
       kyo(ic,ibox) = ky(ic,ibox)
       kzo(ic,ibox) = kz(ic,ibox)
       prefacto(ic,ibox) = prefact(ic,ibox)
    end do
  end subroutine save_kvector

!> \brief restore old k vectors and reciprocal sum and calp
  subroutine restore_kvector(ibox)
    integer,intent(in)::ibox
    integer::ic,ncount

    calp(ibox) = calpo(ibox)
    numvect(ibox) = numvecto(ibox)
    ncount = numvecto(ibox)
    do ic = 1,ncount
       kx(ic,ibox) = kxo(ic,ibox)
       ky(ic,ibox) = kyo(ic,ibox)
       kz(ic,ibox) = kzo(ic,ibox)
       prefact(ic,ibox) = prefacto(ic,ibox)
    end do
  end subroutine restore_kvector

  subroutine recip_atom(ibox,vrecipnew,vrecipold,type,ii)
      integer::ic,izz,ii,imolty,ibox,ncount,type
      real::vrecipnew,vrecipold,sumr(2),sumi(2) ,arg

      ncount = numvect(ibox)

      if ( type .eq. 1 ) then
! recalculate the reciprocal space part for one-particle move, translation,
! rotation, swap, flucq, and swatch.
! old conformation izz = 1 (which is 0 for swap inserted molecule)
! new conformation izz = 2 (which is 0 for swap removed molecule)

#ifdef __DEBUG_KSPACE__
         write(io_output,*) myid,' in recip_atom:',moltion(1),moltion(2)
         do izz = 1,2
            imolty = moltion(izz)
            do ii = 1, nunit(imolty)
               write(io_output,*) rxuion(ii,izz),ryuion(ii,izz),rzuion(ii,izz),qquion(ii,izz)
            end do
         end do
#endif

         do 30 ic = 1, ncount
            do 20 izz = 1,2
! izz = 1: old configuration
! izz = 2: new configuration

               sumr(izz) = 0.0E0_dp
               sumi(izz) = 0.0E0_dp
               imolty = moltion(izz)
                  if ( lqchg(ntype(imolty,ii)) ) then
                     arg = kx(ic,ibox)*rxuion(ii,izz) + ky(ic,ibox)*ryuion(ii,izz) + kz(ic,ibox)*rzuion(ii,izz)
                     sumr(izz) = sumr(izz) +  qquion(ii,izz)*cos(arg)
                     sumi(izz) = sumi(izz) +  qquion(ii,izz)*sin(arg)
                  end if
 20         continue
            ssumrn(ic,ibox) = ssumr(ic,ibox) - sumr(1) + sumr(2)
            ssumin(ic,ibox) = ssumi(ic,ibox) - sumi(1) + sumi(2)
 30      continue
         vrecipnew = 0.0E0_dp
         vrecipold = 0.0E0_dp
         do ic = 1,ncount
            vrecipnew = vrecipnew + (ssumrn(ic,ibox)* ssumrn(ic,ibox) + ssumin(ic,ibox)* ssumin(ic,ibox))*prefact(ic,ibox)
            vrecipold = vrecipold + (ssumr(ic,ibox)* ssumr(ic,ibox) + ssumi(ic,ibox)* ssumi(ic,ibox))*prefact(ic,ibox)
         end do

         vrecipnew = vrecipnew*qqfact
         vrecipold = vrecipold*qqfact

      else if (type .eq. 2) then

! update the reciprocal space k vectors

         do ic = 1, ncount
            ssumr(ic,ibox) = ssumrn(ic,ibox)
            ssumi(ic,ibox) = ssumin(ic,ibox)
         end do

      else if (type .eq. 3) then

! store the reciprocal space k vectors

         do ic = 1, ncount
            ssumro(ic,ibox) = ssumr(ic,ibox)
            ssumio(ic,ibox) = ssumi(ic,ibox)
         end do

      else if (type .eq. 4) then

! restore the reciprocal space k vectors

         do ic = 1, ncount
            ssumr(ic,ibox) = ssumro(ic,ibox)
            ssumi(ic,ibox) = ssumio(ic,ibox)
         end do

      end if

#ifdef __DEBUG_KSPACE__
      write(io_output,*) myid,' in recip_atom:',ssumr(100,ibox),ibox,ssumrn(100,ibox)
#endif

      return
  end subroutine recip_atom

  subroutine ee_recip(ibox,vrecipnew,vrecipold,type)
      integer::ic,zzz,ii,imolty,ibox,ncount,type
      real::vrecipnew,vrecipold,sumr(2),sumi(2) ,arg

      ncount = numvect(ibox)

      if ( type .eq. 1 ) then

! recalculate the reciprocal space part for one-particle move, translation,
! rotation, swap, flucq, and swatch.
! old conformation zzz = 1 (which is 0 for swap inserted molecule)
! new conformation zzz = 2 (which is 0 for swap removed molecule)

#ifdef __DEBUG_KSPACE__
         write(io_output,*) myid,' in ee_recip:',moltion(1),moltion(2)
         do zzz = 1,2
            imolty = moltion(zzz)
            do ii = 1, nunit(imolty)
               write(io_output,*) rxuion(ii,zzz),ryuion(ii,zzz),rzuion(ii,zzz),qquion(ii,zzz)
            end do
         end do
#endif

         do 30 ic = 1, ncount
            do 20 zzz = 1,2
! zzz = 1: old configuration
! zzz = 2: new configuration

               sumr(zzz) = 0.0E0_dp
               sumi(zzz) = 0.0E0_dp
               imolty = moltion(zzz)
               do ii = 1, nunit(imolty)
! if ( lqchg(ntype(imolty,ii)) ) then
                     arg = kx(ic,ibox)*rxuion(ii,zzz) + ky(ic,ibox)*ryuion(ii,zzz) + kz(ic,ibox)*rzuion(ii,zzz)
                     sumr(zzz) = sumr(zzz) +  qquion(ii,zzz)*cos(arg)
                     sumi(zzz) = sumi(zzz) +  qquion(ii,zzz)*sin(arg)
! end if
               end do
 20         continue
            ssumrn(ic,ibox) = ssumr(ic,ibox) - sumr(1) + sumr(2)
            ssumin(ic,ibox) = ssumi(ic,ibox) - sumi(1) + sumi(2)
 30      continue
         vrecipnew = 0.0E0_dp
         vrecipold = 0.0E0_dp
         do ic = 1,ncount
            vrecipnew = vrecipnew + (ssumrn(ic,ibox)* ssumrn(ic,ibox) + ssumin(ic,ibox)* ssumin(ic,ibox))*prefact(ic,ibox)
            vrecipold = vrecipold + (ssumr(ic,ibox)* ssumr(ic,ibox) + ssumi(ic,ibox)* ssumi(ic,ibox))*prefact(ic,ibox)
         end do

         vrecipnew = vrecipnew*qqfact
         vrecipold = vrecipold*qqfact

      else if (type .eq. 2) then

! update the reciprocal space k vectors

         do ic = 1, ncount
            ssumr(ic,ibox) = ssumrn(ic,ibox)
            ssumi(ic,ibox) = ssumin(ic,ibox)
         end do

      else if (type .eq. 3) then

! store the reciprocal space k vectors

         do ic = 1, ncount
            ssumro(ic,ibox) = ssumr(ic,ibox)
            ssumio(ic,ibox) = ssumi(ic,ibox)
         end do

      else if (type .eq. 4) then

! restore the reciprocal space k vectors

         do ic = 1, ncount
            ssumr(ic,ibox) = ssumro(ic,ibox)
            ssumi(ic,ibox) = ssumio(ic,ibox)
         end do

      end if

#ifdef __DEBUG_KSPACE__
      write(io_output,*) myid,' in ee_recip:',ssumr(100,ibox),ibox,ssumrn(100,ibox)
#endif
      return
  end subroutine ee_recip

!> \brief Calculates the reciprocal space contribution to pressure using
!> thermodynamic definition.
!> \see J. Chem. Phys. Vol. 109 P2791.
!> \par History
!> written in 1998 by Bin Chen \n
!> modified to calculate surface tension, 11/24/03 JMS
  subroutine recippress(ibox,repress,pxx,pyy,pzz,pxy,pyx,pxz,pzx, pyz,pzy)
    integer::ncount,ibox,i,ii,imolty
    real::factor,repress,repressx,repressy,repressz,recipintra,piix,piiy,piiz,xcmi,ycmi,zcmi,arg
    real::pxx,pyy,pzz,intraxx,intrayy,intrazz,intraxy,intraxz,intrazy,intrayz,intrayx,intrazx,pxy,pyx,pyz,pzy,pxz,pzx

    repress  = 0.0E0_dp
    repressx = 0.0E0_dp
    repressy = 0.0E0_dp
    repressz = 0.0E0_dp
    recipintra = 0.0E0_dp
    pxy = 0.0E0_dp
    pxz = 0.0E0_dp
    pyx = 0.0E0_dp
    pyz = 0.0E0_dp
    pzx = 0.0E0_dp
    pzy = 0.0E0_dp

    intraxx = 0.0E0_dp
    intrayy = 0.0E0_dp
    intrazz = 0.0E0_dp
    intraxy = 0.0E0_dp
    intrazy = 0.0E0_dp
    intraxz = 0.0E0_dp
    intrazx = 0.0E0_dp
    intrayz = 0.0E0_dp
    intrayx = 0.0E0_dp

    ! RP for MPI
    do ncount = myid+1,numvect(ibox),numprocs
       ! do ncount = 1, numvect(ibox)
       factor = prefact(ncount,ibox)*(ssumr(ncount,ibox)*ssumr(ncount,ibox) + ssumi(ncount,ibox)* ssumi(ncount,ibox))
       repressx = repressx + factor*(1.0E0_dp - (1.0E0_dp/(4.0E0_dp*calp(ibox) *calp(ibox))&
        + 1.0E0_dp/(kx(ncount,ibox)*kx(ncount,ibox)+ ky(ncount,ibox)*ky(ncount,ibox)+kz(ncount,ibox)* kz(ncount,ibox)))&
        *2.0E0_dp*kx(ncount,ibox)*kx(ncount,ibox))
       repressy = repressy + factor*(1.0E0_dp - (1.0E0_dp/(4.0E0_dp*calp(ibox) *calp(ibox))&
        + 1.0E0_dp/(kx(ncount,ibox)*kx(ncount,ibox)+ ky(ncount,ibox)*ky(ncount,ibox)+kz(ncount,ibox)* kz(ncount,ibox)))&
        *2.0E0_dp*ky(ncount,ibox)*ky(ncount,ibox))
       repressz = repressz + factor*(1.0E0_dp - (1.0E0_dp/(4.0E0_dp*calp(ibox) *calp(ibox))&
        + 1.0E0_dp/(kx(ncount,ibox)*kx(ncount,ibox)+ ky(ncount,ibox)*ky(ncount,ibox)+kz(ncount,ibox)* kz(ncount,ibox)))&
        *2.0E0_dp*kz(ncount,ibox)*kz(ncount,ibox))
       pxy = pxy + factor*(0.0E0_dp - (1.0E0_dp/(4.0E0_dp*calp(ibox) *calp(ibox))&
        + 1.0E0_dp/(kx(ncount,ibox)*kx(ncount,ibox)+ ky(ncount,ibox)*ky(ncount,ibox)+kz(ncount,ibox)* kz(ncount,ibox)))&
        *2.0E0_dp*kx(ncount,ibox)*ky(ncount,ibox))
       pxz = pxz + factor*(0.0E0_dp - (1.0E0_dp/(4.0E0_dp*calp(ibox) *calp(ibox))&
        + 1.0E0_dp/(kx(ncount,ibox)*kx(ncount,ibox)+ ky(ncount,ibox)*ky(ncount,ibox)+kz(ncount,ibox)* kz(ncount,ibox)))&
        *2.0E0_dp*kx(ncount,ibox)*kz(ncount,ibox))
       pyz = pyz + factor*(0.0E0_dp - (1.0E0_dp/(4.0E0_dp*calp(ibox) *calp(ibox))&
        + 1.0E0_dp/(kx(ncount,ibox)*kx(ncount,ibox)+ ky(ncount,ibox)*ky(ncount,ibox)+kz(ncount,ibox)* kz(ncount,ibox)))&
        *2.0E0_dp*ky(ncount,ibox)*kz(ncount,ibox))
    end do

    ! RP added for MPI
    call mp_sum(repressx,1,groupid)
    call mp_sum(repressy,1,groupid)
    call mp_sum(repressz,1,groupid)
    call mp_sum(pxy,1,groupid)
    call mp_sum(pxz,1,groupid)
    call mp_sum(pyz,1,groupid)

    repress = repressx + repressy + repressz
    ! keep x,y,z separate for surface tension calculation
    pxx = repressx
    pyy = repressy
    pzz = repressz
    pyx = pxy
    pzx = pxz
    pzy = pyz

    ! the intramolecular part should be substracted
    ! RP for MPI
    do i = myid+1, nchain, numprocs
       ! check if i is in relevant box ###
       if ( nboxi(i) .eq. ibox ) then
          imolty = moltyp(i)
          if ( .not. lelect(imolty) ) cycle
          xcmi = xcm(i)
          ycmi = ycm(i)
          zcmi = zcm(i)

          ! loop over all beads ii of chain i
          do ii = 1, nunit(imolty)
             ! compute the vector of the bead to the COM (p)
             piix = rxu(i,ii) - xcmi
             piiy = ryu(i,ii) - ycmi
             piiz = rzu(i,ii) - zcmi

             do ncount = 1,numvect(ibox)
                ! compute the dot product of k and r
                arg = kx(ncount,ibox)*rxu(i,ii) + ky(ncount,ibox)*ryu(i,ii) + kz(ncount,ibox)*rzu(i,ii)
                factor = prefact(ncount,ibox)*2.0E0_dp* (-ssumr(ncount,ibox)*sin(arg) +ssumi(ncount,ibox)*cos(arg))*qqu(i,ii)
                recipintra = recipintra + factor* (kx(ncount,ibox)*piix+ky(ncount,ibox)*piiy +kz(ncount,ibox)*piiz)
                ! keep x,y and z separate for surface tension calculation
                intraxx = intraxx + factor*(kx(ncount,ibox)*piix)
                intrayy = intrayy + factor*(ky(ncount,ibox)*piiy)
                intrazz = intrazz + factor*(kz(ncount,ibox)*piiz)
                intraxy = intraxy + factor*(kx(ncount,ibox)*piiy)
                intraxz = intraxz + factor*(kx(ncount,ibox)*piiz)
                intrayx = intrayx + factor*(ky(ncount,ibox)*piix)
                intrayz = intrayz + factor*(ky(ncount,ibox)*piiz)
                intrazx = intrazx + factor*(kz(ncount,ibox)*piix)
                intrazy = intrazy + factor*(kz(ncount,ibox)*piiy)

             end do
          end do
       end if
    end do

    call mp_sum(recipintra,1,groupid)
    call mp_sum(intraxx,1,groupid)
    call mp_sum(intrayy,1,groupid)
    call mp_sum(intrazz,1,groupid)
    call mp_sum(intraxy,1,groupid)
    call mp_sum(intraxz,1,groupid)
    call mp_sum(intrayx,1,groupid)
    call mp_sum(intrayz,1,groupid)
    call mp_sum(intrazx,1,groupid)
    call mp_sum(intrazy,1,groupid)

    repress = (repress + recipintra)*qqfact

    pxx = (pxx + intraxx)*qqfact
    pyy = (pyy + intrayy)*qqfact
    pzz = (pzz + intrazz)*qqfact

    pxy = pxy + intraxy
    pyx = pyx + intrayx
    pxz = pxz + intraxz
    pzx = pzx + intrazx
    pyz = pyz + intrayz
    pzy = pzy + intrazy

#ifdef __DEBUG_KSPACE__
    write(io_output,*) myid,' in recippress. Internal part:',intraxx,intrayy,intrazz
#endif
    return
  end subroutine recippress

  subroutine allocate_kspace()
    use sim_system,only:nbxmax
    integer::jerr
    if (allocated(kx)) deallocate(kx,ky,kz,prefact,ssumr,ssumi,ssumrn,ssumin,ssumro,ssumio,kxo,kyo,kzo,prefacto,calpo,calp,numvect,numvecto,&
                                  k_max_l,k_max_m,k_max_n,stat=jerr)
    allocate(kx(vectormax,nbxmax),ky(vectormax,nbxmax),kz(vectormax,nbxmax),prefact(vectormax,nbxmax)&
     ,ssumr(vectormax,nbxmax),ssumi(vectormax,nbxmax),ssumrn(vectormax,nbxmax),ssumin(vectormax,nbxmax)&
     ,ssumro(vectormax,nbxmax),ssumio(vectormax,nbxmax),kxo(vectormax,nbxmax),kyo(vectormax,nbxmax)&
     ,kzo(vectormax,nbxmax),prefacto(vectormax,nbxmax),calpo(nbxmax),calp(nbxmax),numvect(nbxmax),numvecto(nbxmax)&
     ,k_max_l(nbxmax),k_max_m(nbxmax),k_max_n(nbxmax),stat=jerr)
    if (jerr.ne.0) call err_exit(__FILE__,__LINE__,'allocate_kspace: allocation failed',jerr)
    numvect=0
  end subroutine allocate_kspace

  subroutine compute_kmax(ibox)
    integer, intent(in)::ibox
    if ((.not.lsolid(ibox)).or.lrect(ibox)) then
       k_max_l(ibox) = aint(calp(ibox)*calp(ibox)*boxlx(ibox)*rcut(ibox)/onepi)+1
       k_max_m(ibox) = aint(calp(ibox)*calp(ibox)*boxly(ibox)*rcut(ibox)/onepi)+1
       k_max_n(ibox) = aint(calp(ibox)*calp(ibox)*boxlz(ibox)*rcut(ibox)/onepi)+1
    else
       k_max_l(ibox) = aint(calp(ibox)*calp(ibox)*hmat(ibox,1)*rcut(ibox)/onepi)+2
       k_max_m(ibox) = aint(calp(ibox)*calp(ibox)*hmat(ibox,5)*rcut(ibox)/onepi)+2
       k_max_n(ibox) = aint(calp(ibox)*calp(ibox)*hmat(ibox,9)*rcut(ibox)/onepi)+2
    end if
  end subroutine compute_kmax
end MODULE energy_kspace
