/*-----------------------------------------------------------------------*/
/* Program: Stream                                                       */
/* Revision: $Id: stream.c,v 5.8 2007/02/19 23:57:39 mccalpin Exp mccalpin $ */
/* Revision: MPI Version 2011/06/22                                      */
/*           Modified by L. Nguyen CS:SI (laurent.nguyen@c-s.fr)         */
/* Revision: Command line options 2014                                   */
/*           Modified by L. Nguyen CEA (laurent.nguyen@cea.fr)           */ 
/* Original code developed by John D. McCalpin                           */
/* Programmers: John D. McCalpin                                         */
/*              Joe R. Zagar                                             */
/*                                                                       */
/* This program measures memory transfer rates in MB/s for simple        */
/* computational kernels coded in C.                                     */
/*-----------------------------------------------------------------------*/
/* Copyright 1991-2005: John D. McCalpin                                 */
/*-----------------------------------------------------------------------*/
/* License:                                                              */
/*  1. You are free to use this program and/or to redistribute           */
/*     this program.                                                     */
/*  2. You are free to modify this program for your own use,             */
/*     including commercial use, subject to the publication              */
/*     restrictions in item 3.                                           */
/*  3. You are free to publish results obtained from running this        */
/*     program, or from works that you derive from this program,         */
/*     with the following limitations:                                   */
/*     3a. In order to be referred to as "STREAM benchmark results",     */
/*         published results must be in conformance to the STREAM        */
/*         Run Rules, (briefly reviewed below) published at              */
/*         http://www.cs.virginia.edu/stream/ref.html                    */
/*         and incorporated herein by reference.                         */
/*         As the copyright holder, John McCalpin retains the            */
/*         right to determine conformity with the Run Rules.             */
/*     3b. Results based on modified source code or on runs not in       */
/*         accordance with the STREAM Run Rules must be clearly          */
/*         labelled whenever they are published.  Examples of            */
/*         proper labelling include:                                     */
/*         "tuned STREAM benchmark results"                              */
/*         "based on a variant of the STREAM benchmark code"             */
/*         Other comparable, clear and reasonable labelling is           */
/*         acceptable.                                                   */
/*     3c. Submission of results to the STREAM benchmark web site        */
/*         is encouraged, but not required.                              */
/*  4. Use of this program or creation of derived works based on this    */
/*     program constitutes acceptance of these licensing restrictions.   */
/*  5. Absolutely no warranty is expressed or implied.                   */
/*-----------------------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include <sys/time.h>
#ifdef __MPI__
#include "mpi.h"
#endif

/* INSTRUCTIONS:
 *
 *	1) Stream requires a good bit of memory to run.  Adjust the
 *          value of 'N' (below) to give a 'timing calibration' of 
 *          at least 20 clock-ticks.  This will provide rate estimates
 *          that should be good to about 5% precision.
 */

/*
 *	3) Compile the code with full optimization.  Many compilers
 *	   generate unreasonably bad code before the optimizer tightens
 *	   things up.  If the results are unreasonably good, on the
 *	   other hand, the optimizer might be too smart for me!
 *
 *         Try compiling with:
 *               cc -O stream_omp.c -o stream_omp
 *
 *         This is known to work on Cray, SGI, IBM, and Sun machines.
 *
 *
 *	4) Mail the results to mccalpin@cs.virginia.edu
 *	   Be sure to include:
 *		a) computer hardware model number and software revision
 *		b) the compiler flags
 *		c) all of the output from the test case.
 * Thanks!
 *
 */

# define HLINE "-------------------------------------------------------------\n"

# ifndef MIN
# define MIN(x,y) ((x)<(y)?(x):(y))
# endif
# ifndef MAX
# define MAX(x,y) ((x)>(y)?(x):(y))
# endif

double** malloc_double_2d(int row, int column)
{
  double** array;
  double* data;
  int i;
 
  if(row <= 0 || column <= 0)
    {
      return NULL;
    }
 
  array = (double**) malloc((size_t)(row*column*sizeof(double) + row*sizeof(double*)));
  if(array == NULL)
    {
      return NULL;
    }
 
  for(i=0,data=(double*)(array+row); i<row; i++,data+=column)
    {
      array[i] = data;
    }
 
  return array;
}
 

long N = 100000;
int NTIMES = 10;
int OFFSET = 0;
double*              a;
double*              b;
double*              c;

#ifdef __MPI__
char *myHostName;
char *hostNames;
#endif

static double	avgtime[4] = {0}, maxtime[4] = {0},
	    mintime[4] = {FLT_MAX,FLT_MAX,FLT_MAX,FLT_MAX};

static char	*label[4] = {"Copy:      ", "Scale:     ",
			     "Add:       ", "Triad:     "};

static double	bytes[4];

extern double mysecond();
extern void checkSTREAMresults();
#ifdef TUNED
extern void tuned_STREAM_Copy();
extern void tuned_STREAM_Scale(double scalar);
extern void tuned_STREAM_Add();
extern void tuned_STREAM_Triad(double scalar);
#endif
#ifdef _OPENMP
extern int omp_get_num_threads();
#endif

int main(int argc, char* argv[])
{
  int			quantum, checktick();
  int			BytesPerWord=sizeof(double);
  long 	j, k;
  double		scalar, t;
  double                **times;
  int opt = 0;
  double mem = -1.0;
  
#ifdef __MPI__
  double                bw=0.0;
  double *              all_bw;
  double                bw_max=0.0, bw_min=0.0, bw_avg=0.0, bw_stdd=0.0, bw_sum=0.0;
  int                   nthreads=1;
  int j_max, j_min;

  MPI_Comm comm = MPI_COMM_WORLD;
  int rank, size, len;
 

  MPI_Init(&argc, &argv);
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);

  hostNames = (char*) malloc(size*MPI_MAX_PROCESSOR_NAME*sizeof(char));
  myHostName = (char*) malloc(MPI_MAX_PROCESSOR_NAME*sizeof(char));
  
  all_bw = (double*) malloc(size*sizeof(double));

  MPI_Get_processor_name(myHostName, &len);
  
  MPI_Allgather(myHostName, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, hostNames, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, MPI_COMM_WORLD);

  free(myHostName);


  if(rank == 0) 
    {
#endif
      while ((opt = getopt(argc, argv, "hn:m:t:o:")) != -1)
	{
	  switch (opt)
	    {
	    case 'h':
	      printf("STREAM CCRT MPI/OpenMP version $Revision: 5.8 $\n");
	      printf("Usage: ./stream.exe [-h] [-n N] [-m mem] [-t ntimes] [-o offset]\n\n");
	      printf("Options:\n");
	      printf("   -n N         Size of a vector\n");
	      printf("   -m mem       Memory (kB) used per process\n");
	      printf("   -t ntimes    Number of times the computation will run\n");
	      printf("   -o offset    Offset\n");
	      printf("   -h           Print this help\n\n");
	      exit(EXIT_SUCCESS);
	      break;
	    case 'n':
	      N=atol(optarg);
	      break;
	    case 'm':
	      mem=atof(optarg);
	      break;
	    case 't':
	      NTIMES=atoi(optarg);
	      break;
	    case 'o':
	      OFFSET=atoi(optarg);
	      break;
	    default:
	      break;
	    }
	  if(mem < 0)
	    {
	      mem =  (3.0 * BytesPerWord) * ( (double) N / 1024.0);
	    }
	  else
	    {
	      N = (long) (1024.0*mem)/(3.0*BytesPerWord);
	    }
	}
#ifdef __MPI__
    }
  MPI_Bcast(&N, 1, MPI_LONG, 0, comm);
  MPI_Bcast(&NTIMES, 1, MPI_INT, 0, comm);
  MPI_Bcast(&OFFSET, 1, MPI_INT, 0, comm);

#endif

  /* --- SETUP --- determine precision and check timing --- */
#ifdef __MPI__
  if(rank == 0)
    {
#endif
      printf(HLINE);
      printf("STREAM CCRT MPI/OpenMP version $Revision: 5.8 $\n");
      printf(HLINE);
      
      printf("This system uses %d bytes per DOUBLE PRECISION word.\n",
	     BytesPerWord);
      
      printf(HLINE);
      printf("Array size = %ld, Offset = %ld\n" , N, (long) OFFSET);
      printf("Total memory required = %.2f KB.\n",
	     (3.0 * BytesPerWord) * ( (double) N / 1024.0));
      printf("Each test is run %d times, but only\n", NTIMES);
      printf("the *best* time for each is used.\n");

#ifdef _OPENMP
      printf(HLINE);
#pragma omp parallel 
      {
#pragma omp master
	{
	  k = (long) omp_get_num_threads();
#ifdef __MPI__
	  nthreads=(int) k;
#endif
	  printf ("Number of Threads requested = %ld\n",k);
	}
      }
#endif
      
      printf(HLINE);
#ifdef __MPI__
      if(size>1)
	{
	  printf("Number of MPI Processes = %d\n", size);
	}
    }
#endif

  bytes[0] = 2 * sizeof(double) * N;
  bytes[1] = 2 * sizeof(double) * N;
  bytes[2] = 3 * sizeof(double) * N;
  bytes[3] = 3 * sizeof(double) * N;

  times = malloc_double_2d(4,NTIMES);
  for(j=0;j<4;j++)
    {
      avgtime[j] = 0;
      maxtime[j] = 0;
      mintime[j] = FLT_MAX;
    }

  /* Allocate buffers */
    
  a = (double *) malloc((N+OFFSET)*sizeof(double));
  b = (double *) malloc((N+OFFSET)*sizeof(double));
  c = (double *) malloc((N+OFFSET)*sizeof(double));

  /* Get initial value for system clock. */
#pragma omp parallel for
  for (j=0; j<N; j++) {
    a[j] = 1.0;
    b[j] = 2.0;
    c[j] = 0.0;
  }

#ifdef __MPI__
  if(rank == 0)    
#endif
    {
      printf(HLINE);
    }

  if  ( (quantum = checktick()) >= 1) 
#ifdef __MPI__
    if(rank == 0)
      {
#endif
	printf("Your clock granularity/precision appears to be "
	       "%d microseconds.\n", quantum);
#ifdef __MPI__
      }
#endif
    else {
#ifdef __MPI__
      if(rank == 0)
	{
#endif
	  printf("Your clock granularity appears to be "
		 "less than one microsecond.\n");
#ifdef __MPI__
	}
#endif
      quantum = 1;
    }

  t = mysecond();
#pragma omp parallel for
  for (j = 0; j < N; j++)
    a[j] = 2.0E0 * a[j];
  t = 1.0E6 * (mysecond() - t);
  
#ifdef __MPI__
  if(rank == 0)
    {
#endif
      printf("Each test below will take on the order"
	     " of %d microseconds.\n", (int) t  );
      printf("   (= %d clock ticks)\n", (int) (t/quantum) );
      printf("Increase the size of the arrays if this shows that\n");
      printf("you are not getting at least 20 clock ticks per test.\n");
      
      printf(HLINE);
      
      printf("WARNING -- The above is only a rough guideline.\n");
      printf("For best results, please be sure you know the\n");
      printf("precision of your system timer.\n");
      printf(HLINE);
#ifdef __MPI__
    }
  MPI_Barrier(comm);
#endif
    
  /*	--- MAIN LOOP --- repeat test cases NTIMES times --- */

  scalar = 3.0;
  for (k=0; k<NTIMES; k++)
    {
      times[0][k] = mysecond();
#ifdef TUNED
      tuned_STREAM_Copy();
#else
#pragma omp parallel for
      for (j=0; j<N; j++)
	c[j] = a[j];
#endif
      times[0][k] = mysecond() - times[0][k];
	
      times[1][k] = mysecond();
#ifdef TUNED
      tuned_STREAM_Scale(scalar);
#else
#pragma omp parallel for
      for (j=0; j<N; j++)
	b[j] = scalar*c[j];
#endif
      times[1][k] = mysecond() - times[1][k];
	
      times[2][k] = mysecond();
#ifdef TUNED
      tuned_STREAM_Add();
#else
#pragma omp parallel for
      for (j=0; j<N; j++)
	c[j] = a[j]+b[j];
#endif
      times[2][k] = mysecond() - times[2][k];
	
      times[3][k] = mysecond();
#ifdef TUNED
      tuned_STREAM_Triad(scalar);
#else
#pragma omp parallel for
      for (j=0; j<N; j++)
	a[j] = b[j]+scalar*c[j];
#endif
      times[3][k] = mysecond() - times[3][k];
    }

#ifdef __MPI__
  MPI_Barrier(comm);
#endif

  /*	--- SUMMARY --- */

  for (k=1; k<NTIMES; k++) /* note -- skip first iteration */
    {
      for (j=0; j<4; j++)
	{
	  avgtime[j] = avgtime[j] + times[j][k];
	  mintime[j] = MIN(mintime[j], times[j][k]);
	  maxtime[j] = MAX(maxtime[j], times[j][k]);
	}
    }

#ifdef __MPI__
  if(rank == -1)
    { 
#endif
      printf("Function      Rate (MB/s)   Avg time     Min time     Max time\n");
#ifdef __MPI__
    }
#endif
  
  for (j=0; j<4; j++) {
    avgtime[j] = avgtime[j]/(double)(NTIMES-1);   
#ifdef __MPI__
    if(rank == -1)
      { 
#endif
	printf("%s%11.4f  %11.4f  %11.4f  %11.4f\n", label[j],
	       1.0E-06 * bytes[j]/mintime[j],
	       avgtime[j],
	       mintime[j],
	       maxtime[j]);
#ifdef __MPI__
      }
#endif
  }

#ifdef __MPI__
  bw = 1.0E-06 * bytes[3]/mintime[3];

  MPI_Allgather(&bw, 1, MPI_DOUBLE, all_bw, 1, MPI_DOUBLE, comm);

  if(rank == -1)
    { 
#endif
      printf(HLINE);

      /* --- Check Results --- */
      checkSTREAMresults();
      printf(HLINE);
#ifdef __MPI__
    }
  if(rank == 0)
    { 
      printf("Triad Rate (MB/s):\n");
      
      bw_max=all_bw[0];
      bw_min=bw_max;
      bw_sum=0.0;
      j_max = 0 ;
      j_min = 0 ;

      for(j=0;j<size;j++)
	{
	  printf("%s\t\t%lf\n", &(hostNames[j*MPI_MAX_PROCESSOR_NAME]),all_bw[j]);
	  if(bw_max < all_bw[j])
	    {
	      j_max = j;
	      bw_max = all_bw[j];
	    }
	  if(bw_min > all_bw[j])
	    {
	      j_min = j;
	      bw_min = all_bw[j];
	    }
	  bw_sum += all_bw[j];  
	}
      bw_avg = bw_sum / size;
      bw_sum = 0.0;
      for(j=0;j<size;j++)
	{
	  bw_sum += (all_bw[j]-bw_avg)*(all_bw[j]-bw_avg);
	}
      if(size<2)
	{
	  bw_stdd=0.0;
	}
      else
	{
	  bw_stdd = sqrt(bw_sum/((double)(size-1)));
	}
      printf("\n\n============SUMMARY============\n\n");
      printf("TRIAD_MAX          = %lf\t\t MB/s on %s\n", bw_max, &(hostNames[j_max*MPI_MAX_PROCESSOR_NAME]));
      printf("TRIAD_MIN          = %lf\t\t MB/s on %s\n", bw_min, &(hostNames[j_min*MPI_MAX_PROCESSOR_NAME]));
      printf("TRIAD_AVG          = %lf\t\t MB/s\n", bw_avg);
      printf("TRIAD_AVG_per_proc = %lf\t\t MB/s\n", bw_avg/nthreads);
      printf("TRIAD_STDD         = %lf\t\t MB/s", bw_stdd);
      printf("\n\n==========END SUMMARY==========\n\n");
    }
#endif
  free(a);
  free(b);
  free(c);
  free(times);
#ifdef __MPI__
  free(hostNames);
  free(all_bw);

  MPI_Finalize();
#endif
  return 0;
}

# define	M	20

int
checktick()
{
  int		i, minDelta, Delta;
  double	t1, t2, timesfound[M];

  /*  Collect a sequence of M unique time values from the system. */

  for (i = 0; i < M; i++) {
    t1 = mysecond();
    while( ((t2=mysecond()) - t1) < 1.0E-6 )
      ;
    timesfound[i] = t1 = t2;
  }

  /*
   * Determine the minimum difference between these M values.
   * This result will be our estimate (in microseconds) for the
   * clock granularity.
   */

  minDelta = 1000000;
  for (i = 1; i < M; i++) {
    Delta = (int)( 1.0E6 * (timesfound[i]-timesfound[i-1]));
    minDelta = MIN(minDelta, MAX(Delta,0));
  }

  return(minDelta);
}

/* A gettimeofday routine to give access to the wall
   clock timer on most UNIX-like systems.  */

#include <sys/time.h>

double mysecond()
{
  struct timeval tp;
  struct timezone tzp;
  int i;

  i = gettimeofday(&tp,&tzp);
  return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}

void checkSTREAMresults ()
{
  double aj,bj,cj,scalar;
  double asum,bsum,csum;
  double epsilon;
  long	j,k;

  /* reproduce initialization */
  aj = 1.0;
  bj = 2.0;
  cj = 0.0;
  /* a[] is modified during timing check */
  aj = 2.0E0 * aj;
  /* now execute timing loop */
  scalar = 3.0;
  for (k=0; k<NTIMES; k++)
    {
      cj = aj;
      bj = scalar*cj;
      cj = aj+bj;
      aj = bj+scalar*cj;
    }
  aj = aj * (double) (N);
  bj = bj * (double) (N);
  cj = cj * (double) (N);

  asum = 0.0;
  bsum = 0.0;
  csum = 0.0;
  for (j=0; j<N; j++) {
    asum += a[j];
    bsum += b[j];
    csum += c[j];
  }
#ifdef VERBOSE
  printf ("Results Comparison: \n");
  printf ("        Expected  : %f %f %f \n",aj,bj,cj);
  printf ("        Observed  : %f %f %f \n",asum,bsum,csum);
#endif

#ifndef abs
#define abs(a) ((a) >= 0 ? (a) : -(a))
#endif
  epsilon = 1.e-8;

  if (abs(aj-asum)/asum > epsilon) {
    printf ("Failed Validation on array a[]\n");
    printf ("        Expected  : %f \n",aj);
    printf ("        Observed  : %f \n",asum);
  }
  else if (abs(bj-bsum)/bsum > epsilon) {
    printf ("Failed Validation on array b[]\n");
    printf ("        Expected  : %f \n",bj);
    printf ("        Observed  : %f \n",bsum);
  }
  else if (abs(cj-csum)/csum > epsilon) {
    printf ("Failed Validation on array c[]\n");
    printf ("        Expected  : %f \n",cj);
    printf ("        Observed  : %f \n",csum);
  }
  else {
    printf ("Solution Validates\n");
  }
}

void tuned_STREAM_Copy()
{
  int j;
#pragma omp parallel for
  for (j=0; j<N; j++)
    c[j] = a[j];
}

void tuned_STREAM_Scale(double scalar)
{
  int j;
#pragma omp parallel for
  for (j=0; j<N; j++)
    b[j] = scalar*c[j];
}

void tuned_STREAM_Add()
{
  int j;
#pragma omp parallel for
  for (j=0; j<N; j++)
    c[j] = a[j]+b[j];
}

void tuned_STREAM_Triad(double scalar)
{
  int j;
#pragma omp parallel for
  for (j=0; j<N; j++)
    a[j] = b[j]+scalar*c[j];
}
