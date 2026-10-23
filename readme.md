# FETI-SAW-CPP

The C++ implementation accompanying the paper "A Novel Two-Level Domain Decomposition Method with Physically Motivated Coarse Space for Surface Acoustic Wave Device Simulations".

## Third-party dependencies

This project uses the following third-party libraries and runtimes:

| Dependency | Purpose |
| --- | --- |
| [Eigen](https://eigen.tuxfamily.org/) | Dense and sparse matrix operations, eigenvalue computations, and the Eigen PARDISO solver interface. |
| [Intel oneMKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html) | PARDISO sparse direct solving and LAPACKE complex LU factorization/solving. The code includes `mkl.h` and uses MKL routines directly. |
| [OpenMP](https://www.openmp.org/) | Shared-memory parallelism for mesh assembly, matrix operations, and subproblem calculations. |

### Build notes

- Install Eigen and make its headers available to the compiler.
- Configure Intel oneMKL headers and libraries, including the components needed by PARDISO and LAPACKE.
- Build with OpenMP support enabled in the compiler and linker.
- Ensure the Eigen version and oneMKL installation are compatible with the compiler and platform being used. This repository does not currently include a build-system configuration that pins dependency versions or provides platform-specific link flags.

The C++ standard library is also required; it is not a separate third-party dependency.

> Note: To protect the copyright, the core source code for the FETI-SAW domain decomposition iterative method is not currently available in this repository. The complete source code will be released once the paper is formally published.