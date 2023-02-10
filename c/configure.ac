#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.63])
AC_INIT([MPI_STREAM], [0.3], [laurent.nguyen@cea.fr])
AM_INIT_AUTOMAKE([-Wall])
AC_CONFIG_SRCDIR([src/])
AC_CONFIG_HEADERS([config.h])

# use the C compiler for the following checks
AC_LANG([C])

# Checks for programs.
AC_CONFIG_MACRO_DIR([m4])
AC_LANG_PUSH(C)
AC_LANG_POP(C)

AM_PROG_AR
AM_PROG_LIBTOOL
AC_PROG_INSTALL

# find OpenMP flag
AC_ARG_WITH([openmp],
    [AS_HELP_STRING([--with-openmp],
        [Use OpenMP])],
    [ENABLE_OPENMP=1],
    [ENABLE_OPENMP=0])

if test $ENABLE_OPENMP = 1 ; then
AC_OPENMP
AC_SUBST(AM_CFLAGS,"$OPENMP_CFLAGS")
AC_SUBST(AM_LDFLAGS,"$OPENMP_LDFLAGS")
fi

# find MPI flag
AC_ARG_WITH([mpi],
    [AS_HELP_STRING([--with-mpi],
        [Use MPI])],
    [ENABLE_MPI=1],
    [ENABLE_MPI=0])

if test $ENABLE_MPI = 1 ; then
AC_PROG_CC([mpicc])
AC_CHECK_HEADERS([mpi.h],[],[AC_MSG_ERROR([Unable to use MPI])])
AC_DEFINE([HAVE_MPI],[],[Enable MPI])
else
AC_PROG_CC([gcc icc clang nvc cc])
fi

# Checks for libraries.

# Checks for header files.
AC_HEADER_STDC
AC_CHECK_HEADERS([stdio.h stdlib.h string.h])
AC_CHECK_HEADERS([getopt.h math.h float.h])
AC_CHECK_HEADERS([sys/time.h limits.h])

# distribute additional compiler and linker flags among Makefiles
# --> set and change these variables instead of CXXFLAGS or LDFLAGS (for user only)
AC_SUBST([AM_LDFLAGS])

# Checks for typedefs, structures, and compiler characteristics.
AC_TYPE_SIZE_T

# Checks for library functions.
AC_FUNC_MALLOC
AC_CHECK_FUNCS([printf])
AC_CHECK_FUNCS([atoi])
AC_CHECK_FUNCS([atol])
AC_CHECK_FUNCS([atof])
AC_CHECK_FUNCS([gettimeofday])
AC_CHECK_FUNCS([getopt])

AC_CHECK_LIB([m],[sqrt],[],[AC_MSG_ERROR([Unable to use -lm])])

AC_CONFIG_FILES([Makefile
	src/Makefile])
AC_OUTPUT
