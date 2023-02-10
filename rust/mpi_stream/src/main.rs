//-----------------------------------------------------------------------
// Program: MPI_Stream                                                   
// Revision: see README.md                                               
// This program measures memory transfer rates in MB/s for simple        
// computational kernels coded in Rust                                   
//-----------------------------------------------------------------------
// Copyright 1991-2005: John D. McCalpin                                 
// Copyright 2007-2023: CEA/DAM/DIF                                      
//-----------------------------------------------------------------------
// License: see License.txt or                                           
//          https://github.com/cea-hpc/mpi_stream/blob/main/LICENSE.txt  
//-----------------------------------------------------------------------

use clap::Parser;
use std::mem;
use std::time::{Instant};


#[derive(Parser, Debug)]
#[command(author, version, about="Stream memory benchmark", long_about = None)]
struct Args {
    /// Size of a vector
    #[arg(short = 'n', long, default_value_t = 128)]
    nsize: u32,

    /// Memory in kB used per process
    #[arg(short = 'm', long, default_value_t = 1.0)]
    memory: f64,

    /// Number of times the computation will run
    #[arg(short = 't', long, default_value_t = 10)]
    ntimes: usize,

    /// Offset
    #[arg(short = 'o', long, default_value_t = 00)]
    offset: u8,
}

fn main() {
    const STREAM_VERSION: &str = env!("CARGO_PKG_VERSION");
    const HLINE: &str = "-------------------------------------------------------------";
    const BYTES_PER_WORD: usize = mem::size_of::<f64>() as usize;
         
    let args = Args::parse();
    let ntimes: usize = args.ntimes;
    let offset: u8 = args.offset;
    let nsize: u32 = if args.memory > 1.0 {((1024.0 * args.memory as f64) / (3.0 * BYTES_PER_WORD as f64)) as u32} else {args.nsize};
    let n = nsize as usize;
    let memory: f64 = (3.0 * BYTES_PER_WORD as f64) * (n as f64/ 1024.0) as f64;
    

    println!("{}", HLINE);
    println!("MPI_STREAM CEA MPI/OpenMP version {}", STREAM_VERSION);
    println!("{}", HLINE);
    println!("This system uses {} bytes per DOUBLE PRECISION word.", BYTES_PER_WORD);
    println!("Array size = {}, Offset = {}", n, offset);
    println!("Total memory required = {} KB.", memory);
    println!("Each test is run {} times, but only", ntimes);
    println!("the *best* time for each is used.");

    // Allocate timer buffers
    let mut bytes: [usize;4] = [0;4];
    bytes[0] = 2 * BYTES_PER_WORD * n;
    bytes[1] = 2 * BYTES_PER_WORD * n;
    bytes[2] = 3 * BYTES_PER_WORD * n;
    bytes[3] = 3 * BYTES_PER_WORD * n;

    let mut times = vec![vec![0.0; ntimes]; 4];
    let mut avgtime: [f64;4] = [0.0;4];
    let mut maxtime: [f64;4] = [0.0;4];
    let mut mintime: [f64;4] = [f64::MAX;4];
    
    // Allocate buffers
    let mut a = vec![1.0;n];
    let mut b = vec![2.0;n];
    let mut c = vec![0.0;n];

  
    let quantum = checktick();
    if quantum >= 1000 {
        println!("Your clock granularity/precision appears to be ");
        println!("{} microseconds.", (quantum as f64/ 1000000.0) as f64);
    } else {
        println!("Your clock granularity/precision appears to be ");
        println!("less than microseconds.");
    }

    // Time a test
    let t_test_start = Instant::now();
    for i in 0..n {
	a[i] = 2.0 * a[i];
    }
    let t_test_stop = t_test_start.elapsed().as_nanos();

    println!("Each test below will take on the order of {} microseconds.", (t_test_stop / 1000000) as u32);
    println!("   (= {} clock ticks)", t_test_stop / quantum);
    println!("Increase the size of the arrays if this shows that");
    println!("you are not getting at least 20 clock ticks per test.");
    println!("{}", HLINE);

    // Main loop --- repeat test cases NTIMES times ---
    let scalar: f64 = 3.0;
    for k in 0..ntimes {
	// Stream_Copy
	let time_copy = Instant::now();
	for j in 0..n {
	    c[j] = a[j];
	}
	times[0][k] = time_copy.elapsed().as_nanos() as f64 / 1.0e9;

	// Stream_Scale
	let time_scale = Instant::now();
	for j in 0..n {
	    b[j] = scalar * c[j];
	}
	times[1][k] = time_scale.elapsed().as_nanos() as f64 / 1.0e9;

	// Stream_Add
	let time_add = Instant::now();
	for j in 0..n {
	    c[j] = a[j] + b[j];
	}
	times[2][k] = time_add.elapsed().as_nanos() as f64 / 1.0e9;

	// Stream_Triad
	let time_triad = Instant::now();
	for j in 0..n {
	    a[j] = b[j] + scalar * c[j];
	}
	times[3][k] = time_triad.elapsed().as_nanos() as f64 / 1.0e9;
    }

    // Summary
    for k in 1..ntimes {
	for j in 0..4 {
	    avgtime[j] = avgtime[j] + times[j][k];
            mintime[j] = f64::min(mintime[j], times[j][k]);
            maxtime[j] = f64::max(maxtime[j], times[j][k]);
	}
    }
    let label: [&str; 4] = ["Copy:      ", "Scale:     ", "Add:       ", "Triad:     "];
    println!("Function      Rate (MB/s)   Avg time     Min time     Max time\n");
    for j in 0..4 {
	avgtime[j] = avgtime[j] / ((ntimes - 1) as f64);
	println!("{}{:11.4}  {:11.4}  {:11.4}  {:11.4}", label[j], 1.0E-06 * (bytes[j] as f64) / mintime[j], avgtime[j], mintime[j], maxtime[j]);
    }
    println!("{}", HLINE);
    check_stream_results(n, ntimes, &a, &b, &c);
}

fn checktick() -> u128 {
    const M: usize = 20;
    let mut timesfound: [u128;M] = [0; M];

    // Collect a sequence of M unique time values from the system.
    for i in 0..M {
        let t1 = Instant::now();
        let mut t2: u128 = t1.elapsed().as_nanos();
        while t2 < 1000 {
            t2 = t1.elapsed().as_nanos();
        };

        timesfound[i] = t2;
    }
    
    // Determine the minimum difference between these M values.
    // This result will be our estimate (in microseconds) for the
    // clock granularity.
    
    let mut min_delta: u128 = 1000000;
    for i in 0..M {
        min_delta = std::cmp::min(min_delta, timesfound[i]);
    }
    return min_delta;
}

fn check_stream_results(n: usize, ntimes: usize, a: &[f64], b: &[f64], c: &[f64]) {
    // Reproduce initialization
    let mut aj: f64 = 1.0;
    let mut bj: f64 = 2.0;
    let mut cj: f64 = 0.0;
    // a[] is modified during timing check
    aj = 2.0 * aj;
    // Now execute timing loop
    let scalar = 3.0;
    for _k in 0..ntimes {
	cj = aj;
	bj = scalar * cj;
	cj = aj + bj;
	aj = bj + scalar * cj;
    }
    aj = aj * n as f64;
    bj = bj * n as f64;
    cj = cj * n as f64;

    let mut asum: f64 = 0.0;
    let mut bsum: f64 = 0.0;
    let mut csum: f64 = 0.0;
    for j in 0..n {
	asum += a[j];
	bsum += b[j];
	csum += c[j];
    }
    println!("Results Comparison:");
    println!("        Expected  : {} {} {}", aj, bj, cj);
    println!("        Observed  : {} {} {}", asum, bsum, csum);
    
    const EPSILON: f64 = 1.0e-8;
    //const EPSILON: f64 = f64::EPSILON; // Rust Epsilon
	
    if f64::abs(aj - asum) / asum > EPSILON {
	println!("Failed Validation on array a[]");
	println!("        Expected  : {}", aj);
	println!("        Observed  : {}", asum);
    } else if f64::abs(bj - bsum) / bsum > EPSILON {
	println!("Failed Validation on array b[]");
	println!("        Expected  : {}", bj);
	println!("        Observed  : {}", bsum);
    }
    else if f64::abs(cj - csum) / csum > EPSILON {
	println!("Failed Validation on array c[]");
	println!("        Expected  : {}", cj);
	println!("        Observed  : {}", csum);
    } else {
	println!("Solution Validates")
    }
}
