# Assignment 3

## Overview

This project implements adaptive numerical integration using **Adaptive Simpson’s Rule** in C with **MPI** and **OpenMP**.

The program supports three execution modes

#### Mode 0 – Serial Baseline

A single process computes the integral using recursive adaptive Simpson’s Rule.

#### Mode 1 – MPI Dynamic Master/Worker

- Rank 0 acts as the master
- Workers request tasks dynamically
- When a task is split, one half may be returned to the master
- This provides dynamic load balancing for irregular workloads

#### Mode 2 – Hybrid MPI + OpenMP

- Rank 0 acts as the coordinator
- The interval [0,1] is split into coarse intervals statically
- Worker ranks process their assigned intervals locally
- Recursive sub-intervals are not sent back to the master
- OpenMP tasks are used to parallelize local recursive work

---

## Supported Functions

The program supports the following function IDs:

- **Function 0:**  
  `f(x) = sin(x) + 0.5 cos(3x)`

- **Function 1:**  
  `f(x) = 1 / (1 + 100(x - 0.3)^2)`

- **Function 2:**  
  `f(x) = sin(200x)e^(-x)`

---

## Compilation

```bash
mpicc -fopenmp integration.c -lm -o integration
```

As on my machine, it uses uses `clang` by default, which does not support `-fopenmp`.  
So the program must be compiled by forcing OpenMPI to use GCC.

```bash
OMPI_CC=gcc-15 mpicc -fopenmp integration.c -lm -o integration
```

## Running the program

### General format

    mpirun -np P ./integration func_id mode tol

### Hybrid Format

    OMP_NUM_THREADS=T mpirun -np P ./integration func_id 2 tol

#### Parameters:

    P = number of MPI processes
    T = number of OpenMP threads per worker process
    func_id = function ID (0, 1, or 2)
    mode = execution mode
        0 = serial
        1 = MPI dynamic master/worker
        2 = hybrid MPI + OpenMP
    tol = error tolerance( 1e-6, 1e-8)

### Run commands example

#### Serial

    mpirun -np 1 ./integration 1 0 1e-6
    mpirun -np 1 ./integration 2 0 1e-8

#### MPI dynamic

    mpirun -np 2 ./integration 1 1 1e-6
    mpirun -np 4 ./integration 1 1 1e-6
    mpirun -np 8 ./integration 2 1 1e-8

#### Hybrid

    OMP_NUM_THREADS=1 mpirun -np 4 ./integration 1 2 1e-6
    OMP_NUM_THREADS=2 mpirun -np 4 ./integration 1 2 1e-6
    OMP_NUM_THREADS=4 mpirun -np 4 ./integration 1 2 1e-6
    OMP_NUM_THREADS=2 mpirun -np 8 ./integration 2 2 1e-8

## Machine Specification

    Machine: MacBook Pro (16-inch, 2019)
    CPU: 2.3 GHz 8-Core Intel Core i9
    RAM: 16 GB 2667 MHz DDR4
    Operating System: macOS

## MPI Version

    mpirun (Open MPI) 5.0.9

## Github Link

https://github.com/Pax-02/Assignment3_IP.git

## Authorship Statement

I Ishimwe Pacis Hanyurwimfura confirm that :

- All code in this repository was written and understood by me
- I implemented each mode (0–2) incrementally, and used git commits to track my progress
- Any external resources I consulted were for general MPI and C language reference, and no code was directly copied from other students or online solutions.
