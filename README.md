# Exploring-Chebyshev-Spectral-Method-with-Quantum-Annealing-via-Carleman-Linearization
Exploring Chebyshev Spectral Method with Quantum Annealing via Carleman Linearization

This repository contains the supplementary material for the manuscript entitled "Exploring Chebyshev Spectral Method with Quantum Annealing via Carleman Linearization" accepted for the IEEE Control Systems Letters

Abstract: This letter explores the integration of Carleman linearization and quantum annealing to solve nonlinear ordinary differential equations (ODEs) with the Chebyshev spectral method. Using Carleman linearization, the computation of optimal Chebyshev coefficients is reformulated as a quadratic program, which can be further transformed into a Quadratic Unconstrained Binary Optimization (QUBO) problem via binary expansion. This QUBO formulation is well-suited for quantum annealers, where solutions are obtained by minimizing the Hamiltonian of the corresponding Ising model. To address hardware limitations, we employ an iterative quantum annealing strategy with dynamically updated parameters. Case studies on the D-Wave platform demonstrate the potential of our approach to harness the strengths of Carleman linearization and Chebyshev approximation theory, offering continuous and differentiable solutions to nonlinear ODEs—especially promising as quantum resources become more economically viable.

In the repsitory, the code of following sections are provided:
- An implementation of Chevbyshev Spectral Method
- Implementations of three ODE examples on the D-Wave Advantage system 5.4 quantum processor unit:
  * Van-der-Pol Model
  * Single Machine Infinite Bus (SMIB) Power System
  * Lotka–Volterra Model
