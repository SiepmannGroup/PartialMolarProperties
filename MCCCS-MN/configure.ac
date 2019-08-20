AC_PREREQ([2.63])
AC_INIT([topmon],[0.1],[siepmanngroup@gmail.com])
AM_INIT_AUTOMAKE
AC_CONFIG_SRCDIR([src/topmon.f])

AC_ARG_ENABLE(mpi,
   [AS_HELP_STRING([--enable-mpi],
       [compile with mpi if possible (default: yes)])],
   [set_use_mpi=1
    if   test "$enableval" = "yes" ; then
      use_mpi=1
   else
      use_mpi=0
   fi],
   [set_use_mpi=0 use_mpi=1])

AC_ARG_ENABLE(openmp,
   [AS_HELP_STRING([--enable-openmp],
       [compile with openmp if possible (default: yes)])],
   [if   test "$enableval" = "yes" ; then
      use_openmp=1
   else
      use_openmp=0
   fi],
   [use_openmp=1])

# debug flags are implemented only for a few cases
AC_ARG_ENABLE(debug,
   [AS_HELP_STRING([--enable-debug],
       [compile with debug flags "-g" (default: no)])],
   [if   test "$enableval" = "yes" ; then
      use_debug=1
   else
      use_debug=0
   fi],
   [use_debug=0])

topdir=$TOPDIR     # current directory
arch=$ARCH         # see below for recognized architectures, plus
                   # "ppc64-mn" for MareNostrum IBM parallel machine
env_cc=$CC         # C compiler (must be in the execution path)
cpp=$CPP           # C preprocessor (as above)
cflags=$CFLAGS     # Flags for C compiler
cppflags=$CPPFLAGS # Flags for C preprocessor
dflags=$DFLAGS     # Fortran file preprocessing options, e.g. -D__DEFINE_THIS
iflags=$IFLAGS     # Location of include files - shouldn't be needed
f77=$F77           # Fortran 77 serial compiler (must be in execution path)
f90=$F90           # Fortran 90 serial compiler (must be in execution path)
mpif90=$MPIF90     # Fortran 90 parallel compiler (must be in execution path)
fflags=$FFLAGS     # Flags for Fortran 77 and 90 compilers
f90flags=$F90FLAGS # Flags for Fortran 90 compiler only
ld=$LD             # Loader (must be in the execution path)
ldflags=$LDFLAGS   # Flags for loader
ld_libs=$LD_LIBS   # Additional libraries
mpi_libs=$MPI_LIBS # MPI libraries - shouldn't be needed
ar=$AR             # ar (shouldn't be needed)
arflags=$ARFLAGS   # Flags for ar (as above)

# configure for current directory by default
if test "$topdir" = "" ; then topdir="`pwd`" ; fi

# check system type (no cross-compilation for now)
AC_CANONICAL_BUILD

# many HPC systems are configured so that running parallel programs
# interactively is disabled: on those systems, AC_PROG_F77 and AC_PROG_CC
# would fail because they can't run the compiled executables.
# to work around that, let's pretend we are cross-compiling even if we aren't
# !!! this relies on undocumented Autoconf behavior !!!

# This is used to distinguish between true and fake cross compilation
# (only on NEC SX8 actually)
if test "$host" != "" ; then ranlib=echo; fi

cross_compiling=yes
if test "$host" = "" ; then host=$build; fi

# identify architecture
if test "$arch" = ""
then
	case $host in
		ia64-*-linux-gnu )      arch=ia64   ;;
		x86_64-*-linux-gnu )    arch=x86_64 ;;
		*-pc-linux-gnu )        arch=ia32   ;;
		*-ibm-aix* )            arch=aix    ;;
		sparc-sun-* )           arch=sparc  ;;
		i386-pc-solaris* )      arch=solaris;;
		powerpc-apple-darwin* ) arch=macppc ;;
		i686-apple-darwin* )    arch=mac686 ;;
		*-pc-cygwin )           arch=cygwin ;;
		sx*-nec* )              arch=necsx  ;;
		powerpc64-*-linux-gnu ) arch=ppc64  ;;
                * )                     AC_MSG_WARN(incorrect host name ?)
        ;;

	esac
fi
echo checking architecture... $arch

# candidate archiver and archiver flags
try_ar="ar"
try_arflags="ruv"
try_iflags=""

# candidate fortran compilers good for all cases
try_mpif90="mpif90"
try_f90="g95 gfortran f90"

# add all needed -D options to try_dflags
try_dflags=""

# candidate compilers and flags based on architecture
case $arch in
ia32 | ia64 | x86_64 )
	try_f90="ifort pgf90 pathf95 sunf95 openf95 $try_f90"
	;;
solaris )
	try_f90="sunf95 $try_f90"
	;;
aix )
	try_mpif90="mpxlf90_r mpxlf90"
	try_f90="xlf90_r xlf90 $try_f90"
	try_arflags="-X64 ruv"
	try_dflags="-D__AIX -D__XLF"
	;;
crayxt | crayxt4 )
	try_f90="ftn $try_f90"
	;;
macppc )
	try_mpif90="mpxlf90 mpf90"
	try_f90="xlf90 $try_f90"
	try_dflags="-D__XLF"
	;;
mac686 | cygwin )
	try_f90="ifort $try_f90"
	;;
necsx )
	sxopt=`echo $host|awk '{print substr($1,1,3)}'`
	echo $sxopt $host
	try_mpif90="sxmpif90"
	try_f90="sxf90"
	try_ar="sxar"
	try_arflags="rv"
        try_dflags='-D__SX6 '
# default for Nec: no parallel unless explicitly required
        if test "$set_use_mpi" -ne 1 ; then use_mpi=0 ; fi
	;;
ppc64 )
	try_mpif90="mpxlf90_r mpf90_r"
	try_f90="xlf90_r $try_f90"
	try_dflags="-D__XLF"
	try_arflags="ruv"
	;;
# PowerPC MareNostrum
ppc64-mn )
	try_f90="xlf90_r"
	try_dflags="-D__XLF"
	try_arflags="ruv"
	;;
* )
	AC_MSG_WARN(unsupported architecture?)
	;;
esac

# check serial Fortran 90 compiler. This must be done before performing
# the check for the parallel compiler (section below) because option
# --disable-mpi will do very strange things otherwise. The reason
# seems to be that autoconf does not repeat all tests for the second
# occurrence of AC_PROG_F77. So the first occurrence is the one that
# must always be performed, the second is optional. PG & CC sep.2006

# use F90 if set, ignore contents of F77 (is stored in f77 for later usage)
unset F77
if test "$f90" = "" ; then f90="$try_f90" ; fi
AC_PROG_F77($f90)
f90=$F77

# check parallel Fortran 90 compiler
if test "$use_mpi" -eq 0 ;
then
	mpif90=$f90
else
        # clear cached values (not sure when and why this is needed)
        unset F77 ac_cv_prog_ac_ct_F77 ac_cv_f77_compiler_gnu ac_cv_prog_f77_g
	if test "$mpif90" = "" ; then mpif90="$try_mpif90 $f90" ; fi
	AC_PROG_F77($mpif90)
	mpif90=$F77
fi

# check which compiler does mpif90 wrap

case "$arch" in
	ia32 | ia64 | x86_64 | mac686 )
	echo $ECHO_N "checking version of $mpif90... $ECHO_C"
	ifort_version=`$mpif90 -V 2>&1 | grep "Intel(R)"`
	sunf95_version=`$mpif90 -V 2>&1 | grep "^sunf95"`
	openf95_version=`$mpif90 -V 2>&1 | grep "^Open64"`
	pgf_version=`$mpif90 -V 2>&1 | grep "^pgf"`
	eko_version=`$mpif90 -v 2>&1 | grep "EKOPath"`
	g95_version=`$mpif90 -v 2>&1 | grep "g95"`
	gfortran_version=`$mpif90 -v 2>&1 | grep "gcc version"`
        #
	if test "$ifort_version" != ""
	then
		version=`$mpif90 -V 2>&1 | grep Version |
			 sed 's/.*Version//' | awk '{print $1}'`
		ifort_version=`echo $version | sed 's/\..*//'`
		echo "${ECHO_T}ifort $version"
                f90_in_mpif90="ifort"
	elif test "$sunf95_version" != ""
	then
		version=`echo $sunf95_version | awk '{print $5}'`
		echo "${ECHO_T}sunf95 $version"
                f90_in_mpif90="sunf95"
	elif test "$openf95_version" != ""
	then
		version=`echo $openf95_version | awk '{print $5}'`
		echo "${ECHO_T}openf95 $version"
                f90_in_mpif90="openf95"
	elif test "$pgf_version" != ""
	then
		version=`echo $pgf_version | awk '{print $2}'`
		echo "${ECHO_T}pgf90 $version"
                f90_in_mpif90="pgf90"
	elif test "$eko_version" != ""
	then
		version=`echo $eko_version | awk '{print $6}'`
		echo "${ECHO_T}pathf95 $version"
                f90_in_mpif90="pathf95"
	elif test "$g95_version" != ""
	then
		version=`echo $g95_version | awk '{print $3}'`
		echo "${ECHO_T}g95 $version"
                f90_in_mpif90="g95"
	elif test "$gfortran_version" != ""
	then
		version=`echo $gfortran_version | awk '{print $3}'`
		echo "${ECHO_T}gfortran $version"
                f90_in_mpif90="gfortran"
	else
		echo "${ECHO_T}unknown, assuming gfortran"
                f90_in_mpif90="gfortran"
	fi
        # check if serial and parallel compiler are the same
        if test "$f90" != "$f90_in_mpif90"; then
           AC_MSG_WARN([parallel compiler $mpif90 uses $f90_in_mpif90, but serial compiler $f90 was detected])
           AC_MSG_WARN([assuming F90=$f90_in_mpif90, discarding $f90])
        fi
        f90=$f90_in_mpif90
	;;
esac

echo setting F90... $f90
echo setting MPIF90... $mpif90

# candidate C and f77 compilers good for all cases
try_cc="cc gcc"
try_f77="$f90"

case "$arch:$f90" in
*:ifort* )
	try_cc="icc ecc $try_cc"
	;;
*:pgf90 )
	try_cc="pgcc $try_cc"
	try_f77="pgf77 $f90"
	;;
*:pathf95 )
	try_cc="pathcc $try_cc"
	;;
*:sunf95 )
	try_cc="suncc $try_cc"
	;;
*:openf95 )
	try_cc="opencc $try_cc"
	;;
aix:*xlf*_r )
	try_cc="xlc_r $try_cc"
	try_f77="xlf_r $f90"
	;;
aix:*xlf* )
	try_cc="xlc $try_cc"
	try_f77="xlf $f90"
	;;
sparc:* | solaris:sunf95 )
	try_f77="f77 $f90"
	;;
cray*:ftn )
	try_cc="cc"
	;;
macppc:xlf90 )
	try_f77="xlf $f90"
	;;
necsx:* )
	try_cc="sxcc"
	;;
ppc64:*xlf* | ppc64-mn:*xlf* )
	try_cc="xlc_r $try_cc"
	try_f77="xlf_r $f90"
	;;
esac

# check serial C compiler
if test "$env_cc" = "" ; then cc="$try_cc" ; else cc="$env_cc"; fi
AC_PROG_CC($cc)
cc=$CC

echo setting CC... $cc

# clear cached values
unset F77 ac_cv_prog_ac_ct_F77 ac_cv_f77_compiler_gnu ac_cv_prog_f77_g

# check serial Fortran 77 compiler (use F77 if it was set)
if test "$f77" = "" ; then f77="$try_f77" ; fi
AC_PROG_F77($f77)
f77=$F77

echo setting F77... $f77

# check Fortran compiler flags
# have_cpp=0: use external C preprocessing for fortran code
# have_cpp=1: use C-like preprocessing in fortran compiler
have_cpp=1
xlf_flags=0

echo using F90... $f90

case "$arch:$f90" in
ia32:ifort* | ia64:ifort* | x86_64:ifort* | mac686:ifort*  )
    if test "$use_debug" -eq 1; then
	try_fflags="-free -O0 -g -traceback -check all -warn unused -warn declarations -warn truncated_source -warn uncalled -warn ignore_loc -D__DEBUG__"
    else
	try_fflags="-free -O3 -xHost -ipo -no-prec-div"
    fi
	try_fflags_openmp="-openmp"
	try_f90flags="\$(FFLAGS)"
	try_fflags_noopt="-O0 -g -traceback"
	try_ldflags=""
	try_ldflags_static="-static"
	try_dflags="$try_dflags -D__INTEL"
	pre_fdflags="-fpp "
	;;
ia32:pgf* | ia64:pgf* | x86_64:pgf* )
# in try_fflags we use -O3 instead of -fast because some *.f files in Multigrid
# confuse the optimizer when option -fast is enabled
    if test "$use_debug" -eq 1; then
	try_fflags="-tp k8-64 -Mbounds -g -Mchkptr -check_bounds"
    else
	try_fflags="-O2 -tp k8-64 -Mvect=sse"
    fi
	try_fflags_openmp="-mp"
	try_f90flags="-fast -Mcache_align -r8"
	try_fflags_noopt="-O0"
	try_ldflags=""
	try_ldflags_static="-Bstatic"
	try_dflags="$try_dflags -D__PGI"
	have_cpp=0
	;;
ia32:path* | ia64:path* | x86_64:path* )
	try_fflags="-march=auto -O2"
	try_f90flags="\$(FFLAGS)"
	try_fflags_noopt="-O0"
	try_ldflags=""
	try_ldflags_static="-static"
	have_cpp=0
	;;
*:g95 )
    if test "$use_debug" -eq 1; then
	try_fflags="-ftrace=full -pg -fbounds-check"
    else
	try_fflags="-O3 -fomit-frame-pointer -finline-functions -ffast-math" #-mcpu=i686 -march=i686
    fi
	try_f90flags="\$(FFLAGS)"
	try_fflags_noopt="-O0 -cpp"
	try_ldflags=""
	try_ldflags_static="-static"
	;;
*:gfortran )
    if test "$use_debug" -eq 1; then
	try_fflags="-ftrace=full -pg -fbounds-check"
    else
	try_fflags="-O3 -fomit-frame-pointer -finline-functions -ffast-math" #-mcpu=i686 -march=i686
    fi
	try_fflags_openmp="-fopenmp"
	try_f90flags="\$(FFLAGS) -x f95-cpp-input"
	try_fflags_noopt="-O0 -g"
	try_ldflags="-g"
	try_dflags="$try_dflags -D__GFORTRAN -D__STD_F95"
	try_ldflags_static="-static"
        ;;
*:sunf95 )
	try_fflags="-O4"
	try_fflags_openmp="-openmp"
	try_f90flags="\$(FFLAGS) -fpp"
	try_fflags_noopt="-O0"
	try_ldflags="-fast"
	try_ldflags_static="-Bstatic"
	imod="-M"
	;;
*:openf95 )
	try_fflags="-O3"
	try_f90flags="\$(FFLAGS) -ftpp"
	try_fflags_noopt="-O0"
	try_ldflags=""
	imod="-I"
	;;
aix:*xlf* )
    if test "$use_debug" -eq 1; then
	try_fflags="-q64 -qalias=noaryovrlp -g -C \
-qarch=auto -qtune=auto -qdpc -Q -qalias=nointptr"
    else
	try_fflags=" -O3 -qstrict -bmaxdata:2000000000" #-check_bounds -g
    fi
        try_fflags_openmp="-qsmp=omp"
	try_f90flags="\$(FFLAGS) -qsuffix=cpp=f90 -qfree=f90"
	try_fflags_noopt="-q64 -O0"
	try_ldflags="-q64"
	# try_ldflags_static="-bstatic"
	pre_fdflags="-WF,"
	xlf_flags=1
	;;
solaris:sunf95 )
	try_fflags="-fast -O2 -fpp"
	try_f90flags="\$(FFLAGS)"
	try_fflags_noopt="-O0 "
	try_ldflags=""
	imod="-M"
	;;
sparc:f90 )
	try_fflags="-fast -O1 -nodepend -xvector=no -xchip=ultra3 \
-xarch=v8plusb -xlic_lib=sunperf"
	try_f90flags="\$(FFLAGS)"
	try_fflags_noopt="-O0 -xlic_lib=sunperf"
	try_ldflags=""
	imod="-M"
        have_cpp=0
	;;
crayxt*:ftn )
# see comment above for pgf*
	try_fflags_openmp="-mp"
        try_fflags="-O3 -r8"
	try_f90flags="-fast -Mcache_align -r8"
        try_fflags_noopt="-O0"
        try_ldflags_openmp="-mp"
        try_ldflags="-v"
	try_dflags="$try_dflags -D__PGI"
	have_cpp=0
	;;
macppc:* )
	try_fflags="-O4 -qarch=auto -qtune=auto -qsuffix=cpp=f90 \
-qdpc -qalias=nointptr"
	try_f90flags="\$(FFLAGS) -qfree=f90"
	try_fflags_noopt="-O0"
	try_ldflags=""
	pre_fdflags="-WF,"
	xlf_flags=1
	;;

necsx:* )
        try_fflags='      -float0 -Cvopt -eab -R5 -Wf,-Ncont,-A dbl4,-P nh,-ptr byte,-pvctl noifopt loopcnt=9999999 expand=12 fullmsg vwork=stack,-fusion,-O noif,-init stack=nan heap=nan'
        try_f90flags='  -f2003  -float0 -Cvopt -eab -R5 -Wf,-Ncont,-A dbl4,-P nh,-ptr byte,-pvctl noifopt loopcnt=9999999 expand=12 fullmsg vwork=stack,-fusion,-O noif,-init stack=nan heap=nan'
        try_f90flags="-$sxopt $try_f90flags"
        try_fflags_noopt='-float0   '
        try_f90flags_noopt='-f2003 -float0 -eab -R5 -C debug  -Wf,-Ncont,-A dbl4,-P nh ,ptr byte,-init stack=nan heap=nan'
        try_f90flags_noopt="$try_f90flags_noopt"
        try_f90flags_inline='-f2003  -float0 -Cvopt -eab -R5 -pi noauto incdir exp=w0gauss -Wf,-Ncont,-A dbl4,-P nh,-ptr byte,-pvctl noifopt loopcnt=9999999 expand=12 fullmsg vwork=stack,-fusion,-O noif,-init stack=nan heap=nan'
        try_f90flags_inline="$try_f90flags_inline"
        try_ldflags_static='-P static'
        try_ldflags='-Wl,-f zero'
        try_ldflags="-p $try_ldflags"
        pre_fdflags=""
	;;

ppc64:*xlf* )
    if test "$use_debug" -eq 1; then
	try_fflags="-g -C -qsuffix=cpp=f90 -qdpc -qalias=nointptr -Q"
    else
	try_fflags="-O4 -qsuffix=cpp=f90 -qdpc -qalias=nointptr -Q"
    fi
	try_f90flags="\$(FFLAGS) -qfree=f90"
	try_fflags_noopt="-O0"
	try_ldflags=""
	pre_fdflags="-WF,"
	xlf_flags=1
	;;

ppc64-mn:* )
    if test "$use_debug" -eq 1; then
	try_fflags="-g -C -q64 -qstrict -qsuffix=cpp=f90 -qdpc -qalias=nointptr -Q -qtune=ppc970 -qarch=ppc970 -qcache=auto -qhot=vector,simd -qenablevmx"
    else
	try_fflags="-O3 -q64 -qstrict -qsuffix=cpp=f90 -qdpc -qalias=nointptr -Q -qtune=ppc970 -qarch=ppc970 -qcache=auto -qhot=vector,simd -qenablevmx"
    fi
	try_f90flags="\$(FFLAGS) -qfree=f90"
	try_fflags_noopt="-O0 -q64"
	try_ldflags=""
	pre_fdflags="-WF,"
	xlf_flags=1
	;;

* )
	# unknown, try these
	try_fflags="-O"
	try_f90flags="\$(FFLAGS)"
	try_fflags_noopt="-O0"
	try_ldflags=""
        have_cpp=0
	;;
#########################
### Compiler definitions
#########################
### NAG Compiler for OSX (f95)
#try_fflags     = -O0 -gline -f77 -mismatch_all -C=all -C=undefined -nan
## only for debugging
#try_fflags     = -O2

### SGI Altix: Intel Compiler for ia64 architecture (ifort)
#try_fflags     = -O2 -tpp2 -w90 -w95 -cm
# only for debugging
#try_fflags     =  -check all -traceback -w90 -w95 -cm -CB -CU -g

### SGI Compiler for Origin (f77)
#try_fflags     = -O2 #-march=opteron #-mips4 -r12000 #-check_bounds -g
#############################
### End compiler definitions
#############################
esac

if test "$use_openmp" -eq 1 ; then
  try_fflags="$try_fflags $try_fflags_openmp"
  try_ldlags="$try_ldlags $try_ldlags_openmp"
fi

if test "$fflags" = ""   ; then fflags=$try_fflags     ; fi
if test "$f90flags" = "" ; then f90flags=$try_f90flags ; fi
if test "$fflags_noopt" = ""   ; then fflags_noopt=$try_fflags_noopt     ; fi
echo setting FFLAGS... $fflags
echo setting F90FLAGS... $f90flags
echo setting FFLAGS_NOOPT... $fflags_noopt
if test "$imod" = "" ; then imod="-I" ; fi

# tentative C and loader flags, good for many cases
try_cflags="-O3"
c_ldflags=""
try_cpp="cpp"

case "$arch:$cc" in
*:pgcc )
	try_cflags="-fast"
	;;
# pathcc -E seems to give problems when preprocessing iotk
#*:pathcc )
#	try_cpp="pathcc -E"
#	;;
aix:xlc* | aix:cc )
	try_cflags="-q64 -O2"
	c_ldflags="-q64"
	;;
*:suncc  )
	try_cflags="-fast -O"
	;;
sparc:cc )
	try_cflags="-fast -dalign -xchip=ultra3 -xarch=v8plusb \
-xlic_lib=sunperf"
	try_cpp="fpp"
	;;
crayxt*:cc )
        try_cflags="-fast"
        ;;
macppc:xlc* )
	try_cflags="-O4"
	;;
necsx:* )
	#try_cflags="-D__SX6 \$(IFLAGS) \$(MODFLAGS)"
        try_cflags=""
	try_ar="sxar"
	;;
ppc64-mn:* )
        try_cflags="-O3 -q64"
	;;
esac
if test "$cflags" = "" ; then cflags=$try_cflags ; fi
echo setting CFLAGS... $cflags

# preprocessor - try cpp in all cases; the preprocessor returned by
# AC_PROG_CPP
# may sometimes refuse to preprocess fortran files
if test "$cpp" = "" ; then cpp=$try_cpp; fi
# if test "$cpp" = "" ; then cpp=$CPP;     fi
echo setting CPP... $cpp

echo $ECHO_N "setting CPPFLAGS... $ECHO_C"
case $cpp in
	cpp)  try_cppflags="-P -traditional" ;;
	fpp)  try_cppflags="-P"              ;;
	*)    try_cppflags=""                ;;
esac
if test "$cppflags" = "" ; then cppflags=$try_cppflags ; fi
echo "${ECHO_T}$cppflags"

# linker and archiver
if test "$ld" = ""       ; then ld="$mpif90"           ; fi
if test "$ldflags" = ""  ; then ldflags="$try_ldflags" ; fi
if test "$ar" = ""       ; then ar="$try_ar"           ; fi
if test "$arflags" = ""  ; then arflags="$try_arflags" ; fi
echo setting LD... $ld
echo setting LDFLAGS... $ldflags
echo setting AR... $ar
echo setting ARFLAGS... $arflags

# compilation rules
AC_PROG_MAKE_SET
echo $ECHO_N "checking whether Fortran files must be preprocessed... $ECHO_C"
if test "$have_cpp" -ne 0
then
	f90rule="\$(MPIF90) \$(F90FLAGS) -c \$<"
	echo "${ECHO_T}no"
else
	f90rule="\$(CPP) \$(CPPFLAGS) \$< -o \$*.F90 ; \\
	\$(MPIF90) \$(F90FLAGS) -c \$*.F90 -o \$*.o"
	echo "${ECHO_T}yes"
fi

# compilation flags for all subsequent tests
# remove all $(...) because at least one compiler doesn't like them
# but if f90flags contains $(FFLAGS), substitute it
test_cflags="`echo $cflags | sed 's/\$([[^)]]*)//g'`"
test_cppflags="$test_cflags"
if test "`echo $f90flags | grep '$(FFLAGS)'`" != ""
then
	test_fflags="`echo $fflags $f90flags | sed 's/\$([[^)]]*)//g'`"
else
	test_fflags="`echo $f90flags | sed 's/\$([[^)]]*)//g'`"
fi
test_ldflags="`echo $ldflags | sed 's/\$([[^)]]*)//g'`"

have_mpi=0
F77=$f90
FFLAGS="$test_fflags"
LDFLAGS="$test_ldflags"

# check for mpi
# some architectures require to link mpi libraries explicitly
F77=$mpif90 # use parallel compiler
if test "$mpi_libs" = ""
then
	# check directories in LD_LIBRARY_PATH too
	# (maybe they are already searched by default, but I'm not sure)
	ld_library_path=`echo $LD_LIBRARY_PATH | sed 's/:/ /g'`

	if test "$use_mpi" -ne 0
	then
		if test "$have_mpi" -eq 0
			# check for mpi
		then
			unset ac_cv_search_mpi_init # clear cached value
			LDFLAGS="$test_ldflags"
			LIBS="$mpi_libs"
			AC_SEARCH_LIBS(mpi_init, mpi,
				       have_mpi=1 mpi_libs="$LIBS")
		fi
	fi
fi
echo setting MPI_LIBS... $mpi_libs


# final check on availability of parallel environment
parallel=0
if test "$use_mpi" -ne 0
then
	F77=$mpif90
	LIBS="$mpi_libs"
	AC_SEARCH_LIBS(mpi_init, "", parallel=1
		       try_dflags="$try_dflags -D__MPI -D__PARA")
fi

# preprocessing flag for openmp (experimental)
if test "$use_openmp" -eq 1 ; then try_dflags="$try_dflags -D__OPENMP" ; fi

if test "$dflags" = "" ; then dflags="$try_dflags" ; fi
echo setting DFLAGS... $dflags

if test "$iflags" = "" ; then iflags="$try_iflags" ; fi
echo setting IFLAGS... $iflags

# xlf compilers (AIX and powerpc) want comma-separated -D directives
if test "$xlf_flags" -ne 0
then
	fdflags="`echo $dflags | sed 's/  */,/g'`"
else
	fdflags="\$(DFLAGS)"
fi
echo setting FDFLAGS... $fdflags

if test "$ranlib" != "echo"
then
   AC_CHECK_PROG(ranlib,ranlib,ranlib,echo)
fi

# MacOs work around for ranlib
if test "$arch" == "mac686"
then
 if test "$ranlib" == "ranlib"
 then
  ranlib="ranlib -c"
 fi
fi

echo setting RANLIB... $ranlib

if test "$mpi_libs" != "" ; then
   mpi_line="MPI_LIBS=$mpi_libs"
else
   mpi_line="@delete@"
fi

if test "$use_mpi" -ne 0
then
	if test "$parallel" -ne 0
	then
		parallel_report="Parallel environment detected successfully.\\
Configured for compilation of parallel executables."
	else
		parallel_report="Parallel environment not detected(is this a parallel machine?).\\
Configured for compilation of serial executables."
	fi
else
	parallel_report="Configured for compilation of serial executables."
fi

# export settings to generated files
AC_SUBST(cc)
AC_SUBST(cflags)
AC_SUBST(dflags)
AC_SUBST(fdflags)
AC_SUBST(cpp)
AC_SUBST(cppflags)
AC_SUBST(f90)
AC_SUBST(mpif90)
AC_SUBST(f90flags)
AC_SUBST(f77)
AC_SUBST(fflags)
AC_SUBST(fflags_noopt)
AC_SUBST(pre_fdflags)
AC_SUBST(imod)
AC_SUBST(iflags)
AC_SUBST(ld)
AC_SUBST(ldflags)
AC_SUBST(ld_libs)
AC_SUBST(mpi_libs)
AC_SUBST(ar)
AC_SUBST(arflags)
AC_SUBST(ranlib)
AC_SUBST(f90rule)
AC_SUBST(mpi_line)
AC_SUBST(parallel_report)
AC_SUBST(topdir)

AC_CONFIG_FILES([Makefile src/Makefile])

AC_OUTPUT

echo \
"------------------------------------------------
${PACKAGE_NAME} Version ${PACKAGE_VERSION}

Prefix: '${prefix}'

Compiler: ${cpp}    @<:@MPI Fortran Compiler Wrapper@:>@
          ${mpif90} @<:@MPI Fortran Compiler Wrapper@:>@

Flags: ${f90flags} ${pre_fdflags}${fdflags} ${iflags} ${modflags}

MPI: ${mpi_line}
     ${parallel_report}

Now type 'make @<:@<target>@:>@'
  where the optional <target> is:
    all      --  build all binaries
    install  --  install everything
------------------------------------------------"
