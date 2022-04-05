# MPI_STREAM

This program measures memory transfer rates in MB/s for simple computational kernels coded in C.

## Context

Since 2007, Stream benchmark is used to test and to check nodes in clusters managed by the CEA/DAM during system updates or system maintenances. The benchmark is useful to detect :
* Memory module failure
* Lack of memory on nodes
* Performance problems on memory modules
* Regression on OS
* Regression on compilers (OpenMP)

Over the years, CEA adds features to be more efficient to detect these problems :
* The MPI version can test all a cluster (more than 8000 nodes) with only a single run.
* An option can be use to indicate an amount of memory instead to compute an array size.
* The output was updated to give a list of nodes sorted by their mesured bandwidth.

To keep a reference code, CEA publishes its modified version on Github in 2022.

## Legacy code

McCalpin, John D., 1995: "Memory Bandwidth and Machine Balance in Current High Performance Computers", IEEE Computer Society Technical Committee on Computer Architecture (TCCA) Newsletter, December 1995.

[Stream benchmark](https://www.cs.virginia.edu/stream/)

[paper](https://www.researchgate.net/publication/213876927_Memory_Bandwidth_and_Machine_Balance_in_Current_High_Performance_Computers)

## Prerequisites

Main program:

* C++ compiler
* OpenMP
* MPI

## Getting started

```
$ ./autogen.sh
$ ./configure                                   # Sequential mode
$ ./configure --with-openmp                     # OpenMP mode
$ ./configure --with-mpi CC=mpicc               # MPI mode
$ ./configure --with-openmp --with-mpi CC=mpicc # MPI/OpenMP mode
$ make
```

## Running

```
$ ./stream.exe -h
MPI_STREAM CEA MPI/OpenMP version $Revision: X.Y $
Usage: ./stream.exe [-h] [-n N] [-m mem] [-t ntimes] [-o offset]

Options:
   -n N         Size of a vector
   -m mem       Memory (kB) used per process
   -t ntimes    Number of times the computation will run
   -o offset    Offset
   -h           Print this help
```

## Examples

You can launch the sequential mode with 1GB of memory:
```
$ ./stream.exe -m 1048576
-------------------------------------------------------------
MPI_STREAM CEA MPI/OpenMP version $Revision: X.Y $
-------------------------------------------------------------
This system uses 8 bytes per DOUBLE PRECISION word.
-------------------------------------------------------------
Array size = 44739242, Offset = 0
Total memory required = 1048575.98 KB.
Each test is run 10 times, but only
the *best* time for each is used.
-------------------------------------------------------------
Number of Threads requested = 1
-------------------------------------------------------------
-------------------------------------------------------------
Your clock granularity/precision appears to be 1 microseconds.
Each test below will take on the order of 20594 microseconds.
   (= 20594 clock ticks)
Increase the size of the arrays if this shows that
you are not getting at least 20 clock ticks per test.
-------------------------------------------------------------
WARNING -- The above is only a rough guideline.
For best results, please be sure you know the
precision of your system timer.
-------------------------------------------------------------
Function      Rate (MB/s)   Avg time     Min time     Max time
Copy:       42299.2351       0.0171       0.0169       0.0172
Scale:      42512.4562       0.0171       0.0168       0.0171
Add:        42376.8484       0.0255       0.0253       0.0256
Triad:      42440.3442       0.0255       0.0253       0.0257
-------------------------------------------------------------
Solution Validates
-------------------------------------------------------------
```

To test 4 nodes with 128 cores and 256GB per node, you can launch the MPI/OpenMP mode with 220GB per node:
```
$ OMP_NUM_THREADS=128 mpirun -n 4 -cpus-per-rank 128 ./stream.exe -m 230686720
-------------------------------------------------------------
MPI_STREAM CEA MPI/OpenMP version $Revision: X.Y $
-------------------------------------------------------------
This system uses 8 bytes per DOUBLE PRECISION word.
-------------------------------------------------------------
Array size = 9842633386, Offset = 0
Total memory required = 230686719.98 KB.
Each test is run 10 times, but only
the *best* time for each is used.
-------------------------------------------------------------
Number of Threads requested = 128
-------------------------------------------------------------
Number of MPI Processes = 4
-------------------------------------------------------------
Your clock granularity/precision appears to be 1 microseconds.
Each test below will take on the order of 487067 microseconds.
   (= 487067 clock ticks)
Increase the size of the arrays if this shows that
you are not getting at least 20 clock ticks per test.
-------------------------------------------------------------
WARNING -- The above is only a rough guideline.
For best results, please be sure you know the
precision of your system timer.
-------------------------------------------------------------
Triad Rate (MB/s):
node6201		336870.222291
node6202		336899.087888
node6203		336828.994303
node6204		336809.300049


============SUMMARY============

TRIAD_MAX          = 336899.087888		 MB/s on node6202
TRIAD_MIN          = 336809.300049		 MB/s on node6204
TRIAD_AVG          = 336851.901133		 MB/s
TRIAD_AVG_per_proc = 2631.655478		 MB/s
TRIAD_STDD         = 40.422064		 MB/s

==========END SUMMARY==========
```

## Contributing

## Authors

See the list of [AUTHORS](AUTHORS) who participated in this project.

## Contact

Laurent Nguyen - <laurent.nguyen@cea.fr>

## Website

[CEA-HPC](http://www-hpc.cea.fr/)

## License

Copyright 2007-2022 CEA/DAM/DIF<br />
<br />
MPI_STREAM is distributed under the original license of STREAM benchmark.<br />
See the included files LICENSE.txt (English version).

## Acknowledgments
