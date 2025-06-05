# paper-2025-perrk
[![License: MIT](https://img.shields.io/badge/License-MIT-success.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/996624801.svg)](https://doi.org/10.5281/zenodo.15601890)

Reproducibility Repository for the paper  
_"Paired Explicit Relaxation Runge-Kutta Methods: Entropy Conservative/Stable High-Order Optimized Multirate Time Integration"_

If you use the implementations provided here, please also cite this repository as
```bibtex
@misc{doehring2025PERRK_ReproRepo,
  title={Reproducibility repository for "Paired Explicit Relaxation Runge-Kutta Methods: Entropy Conservative/Stable High-Order Optimized Multirate Time Integration"},
  author={Doehring, Daniel and Ranocha, Hendrik, and Torrilhon, Manuel},
  year={2025},
  howpublished={\url{https://github.com/DanielDoehring/paper-2025-perrk}},
  doi={https://doi.org/10.5281/zenodo.15601890}
}
```

## Abstract

We present novel entropy conservative and entropy stable multirate Runge-Kutta methods based on Paired Explicit Runge-Kutta (P-ERK) with relaxation for conservation laws and related systems of partial differential equations.
Optimized schemes up to fourth-order of accuracy are derived and validated in terms of order of consistency, conservation of linear invariants, and entropy conservation/stability.

We demonstrate the effectiveness of these P-ERRK methods when combined with a high-order, entropy-conservative/stable discontinuous Galerkin spectral element method on unstructured meshes.
The Paired Explicit _Relaxation_ Runge-Kutta methods (P-ERRK) are readily implemented for partitioned semidiscretizations arising from problems with equation-based scale separation such as non-uniform meshes.
We highlight that the relaxation approach acts as a time-limiting technique which improves the nonlinear stability and thus robustness of the multirate schemes.

The P-ERRK methods are applied to a range of problems, ranging from compressible Euler over compressible Navier-Stokes to the visco-resistive magnetohydrodynamics equations in two and three spatial dimensions.
For each test case, we compare computational load and runtime to standalone relaxed Runge-Kutta methods which are outperformed by factors up to four.
All results can be reproduced using a publicly available repository.
## Reproducing the results

### Installation

To download the code using `git`, use 

```bash
git clone git@github.com:DanielDoehring/paper-2025-perrk.git
``` 

If you do not have git installed you can obtain a `.zip` and unpack it:
```bash
wget git@github.com:DanielDoehring/paper-2025-perrk.git/archive/main.zip
unzip paper-2025-perrk
```

To instantiate the Julia environment execute the following two commands:
```bash
cd paper-2025-perrk
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Note that the results are obtained using Julia 1.10.9, which is also set in the `Manifest.toml`.
Thus, you might need to install the [Julia 1.10.9 LTS release](https://julialang.org/downloads/) first
and *replace* the `julia` calls from this README with
`/YOUR/PATH/TO/julia-1.10.9/bin/julia`

### Project initialization

If you installed Trixi.jl this way, you always have to start Julia with the `--project` flag set to your `paper-2025-perrk` directory, e.g.,
```bash
julia --project=.
```
if already inside the `paper-2025-perrk` directory.

If you do not execute from the `paper-2025-perrk` directory, you have to call `julia` with
```bash
julia --project=/YOUR/PATH/TO/paper-2025-perrk
```

### Running the code

The scripts for validations and applications are located in the `3_PERRK_Methods`, `4_Validation`, and `5_Applications` directory, respectively.

To execute them provide the respective path:

```bash
julia --project=. ./4_Validation/4_1_EntropyConservation/4_1_1_WeakBlastWave_Euler_MHD/elixir_euler_weak_blast_er.jl
```

For all cases in the `5_Applications` directory the solution has been computed using a specific number of 
threads.
To specify the number of threads the [`--threads` flag](https://docs.julialang.org/en/v1/manual/multi-threading/#Starting-Julia-with-multiple-threads) needs to be given, i.e., 
```bash
julia --project=. --threads 6 ./5_Applications/5_3_NACA0012_AMR/elixir_euler_NACA0012airfoil_mach08.jl
```
The number of threads used for the examples are given in the `README.md` in `5_Applications`.

## Authors

* [Daniel Doehring](https://www.acom.rwth-aachen.de/the-lab/team-people/name:daniel_doehring) (Corresponding Author)
* [Hendrik Ranocha](https://ranocha.de/home#gsc.tab=0)
* [Manuel Torrilhon](https://www.acom.rwth-aachen.de/the-lab/team-people/name:manuel_torrilhon)

Note that the Trixi authors are listed separately [here](https://github.com/DanielDoehring/paper-2025-perrk/blob/main/Trixi.jl-v0.12.2%2Bmod/AUTHORS.md).

## Disclaimer

Everything is provided as is and without warranty. Use at your own risk!
