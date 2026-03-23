#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <vector>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

double compute_spatial_entropy_cpp(py::array_t<double> X_in) {
    py::buffer_info buf = X_in.request();
    if (buf.ndim != 2) throw std::runtime_error("Number of dimensions must be 2");

    size_t N = buf.shape[0];
    size_t dim = buf.shape[1];
    double* X = static_cast<double*>(buf.ptr);

    std::vector<double> d_matrix(N * N, 0.0);
    double sum_d = 0.0;
    long long count = 0;

    // Euclidean distance matrix
    #pragma omp parallel for reduction(+:sum_d, count)
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = i + 1; j < N; ++j) {
            double dist_sq = 0.0;
            for (size_t k = 0; k < dim; ++k) {
                double diff = X[i * dim + k] - X[j * dim + k];
                dist_sq += diff * diff;
            }
            double dist = std::sqrt(dist_sq);
            d_matrix[i * N + j] = dist;
            d_matrix[j * N + i] = dist;
            sum_d += 2.0 * dist;
            count += 2;
        }
    }

    if (count == 0 || sum_d == 0.0) return std::log((double)N);

    double sigma = sum_d / count;
    double sigma_sq = std::max(sigma * sigma, 1e-15);

    std::vector<double> rho(N, 0.0);
    double sum_rho = 0.0;

    // Gaussian Kernel Density
    #pragma omp parallel for reduction(+:sum_rho)
    for (size_t i = 0; i < N; ++i) {
        double r_i = 0.0;
        for (size_t j = 0; j < N; ++j) {
            if (i != j) {
                double d = d_matrix[i * N + j];
                r_i += std::exp(-(d * d) / (2.0 * sigma_sq));
            }
        }
        rho[i] = r_i;
        sum_rho += r_i;
    }

    if (sum_rho < 1e-15) return std::log((double)N);

    double entropy = 0.0;
    
    // Shannon Entropy
    #pragma omp parallel for reduction(-:entropy)
    for (size_t i = 0; i < N; ++i) {
        double p_i = rho[i] / sum_rho;
        if (p_i > 0) {
            entropy -= p_i * std::log(p_i);
        }
    }

    return entropy;
}

PYBIND11_MODULE(sede_core, m) {
    m.doc() = "C++ OpenMP implementation of SEDE spatial entropy";
    m.def("compute_spatial_entropy_cpp", &compute_spatial_entropy_cpp, "Compute Spatial Entropy (OpenMP Array)", py::arg("X_in"));
}
