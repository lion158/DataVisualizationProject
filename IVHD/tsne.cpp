/*
 *
 * Copyright (c) 2014, Laurens van der Maaten (Delft University of Technology)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *    This product includes software developed by the Delft University of Technology.
 * 4. Neither the name of the Delft University of Technology nor the names of
 *    its contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY LAURENS VAN DER MAATEN ''AS IS'' AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL LAURENS VAN DER MAATEN BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
 * IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
 * OF SUCH DAMAGE.
 *
 */

#include "tsne.h"
#include "sptree.h"
#include "vptree.h"
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
//#include <format>

using namespace std;

static double sign(double x) { return (x == .0 ? .0 : (x < .0 ? -1.0 : 1.0)); }
static double randn();
static void zeroMean(std::vector<double>& X, int N, int D);
static void computeSquaredEuclideanDistance(std::vector<double>& X, int N, int D, vector<double>& DD_vector);
static void symmetrizeMatrix(std::vector<unsigned int>& _row_P, std::vector<unsigned int>& _col_P, std::vector<double>& _val_P, int N);

static void computeGaussianPerplexity(vector<double>& X, int N, int D, vector<double>& P, double perplexity);

static void computeGaussianPerplexity(const viskit::embed::cast::TSNEParams& params, vector<double>& X, std::vector<unsigned int>& _row_P, std::vector<unsigned int>& _col_P, std::vector<double>& _val_P);
static void computeGaussianPerplexityWithNNGraph(const viskit::embed::cast::TSNEParams& params, std::vector<unsigned int>& row_P, std::vector<unsigned int>& col_P, std::vector<double>& val_P, Graph& graph);

static void computeExactGradient(vector<double>& P, vector<double>& Y, int N, int D, vector<double>& dC);
static void computeGradient(std::vector<unsigned int>& inp_row_P, std::vector<unsigned int>& inp_col_P, std::vector<double>& inp_val_P, std::vector<double>& Y, int N, int D, vector<double>& dC, double theta);

static double evaluateError(vector<double>& P, std::vector<double>& Y, int N, int D);
static double evaluateError(std::vector<unsigned int>& row_P, std::vector<unsigned int>& col_P, std::vector<double>& val_P, std::vector<double>& Y, int N, int D, double theta);

std::unique_ptr<viskit::embed::cast::TSNEState> viskit::embed::cast::tsne::initialize(const viskit::embed::cast::TSNEParams& params, std::vector<double>& X, Graph& graph, const core::System& m_ext_system)
{
    int N = params.N;
    int D = params.input_dims;
    int no_dims = params.output_dims;
    double theta = params.theta;
    double perplexity = params.perplexity;
    bool exact = params.is_exact();
    int rand_seed = params.rand_seed;
    bool skip_random_init = params.skip_random_init;

    clock_t start;
    double momentum = .5;
    double final_momentum = .8;
    double eta = 200.0;
    vector<double> dY(N * no_dims);
    vector<double> uY(N * no_dims);
    vector<double> gains(N * no_dims);
    vector<double> P;
    vector<unsigned int> row_P;
    vector<unsigned int> col_P;
    vector<double> val_P;
    vector<double> Y(N * params.output_dims);

    if (exact && params.use_nn_graph) {
        m_ext_system.logger().logInfo("Graph mode can be used with non exact, sparse table mode (theta > 0.0)");
        exit(1);
    }

    if (!skip_random_init) {
        if (rand_seed >= 0) {
            m_ext_system.logger().logInfo("Using random seed: " + std::to_string(rand_seed));
            srand((unsigned int)rand_seed);
        } else {
            m_ext_system.logger().logInfo("[t-SNE Caster] Using current time as random seed...");
            srand(time(nullptr));
        }
    }

    // Determine whether we are using an exact algorithm
    if (N - 1 < 3 * perplexity) {
        m_ext_system.logger().logInfo("Perplexity too large for the number of data points!");
        exit(1);
    }

    m_ext_system.logger().logInfo("Using no_dims = " + std::to_string(no_dims) +
    ", perplexity = " + std::to_string(perplexity) +", and theta = " + std::to_string(theta));

    for (int i = 0; i < N * no_dims; i++)
        uY[i] = .0;
    for (int i = 0; i < N * no_dims; i++)
        gains[i] = 1.0;

    // Normalize input data (to prevent numerical problems)
    m_ext_system.logger().logInfo("Computing input similarities...");
    start = clock();
    zeroMean(X, N, D);
    double max_X = .0;
    for (int i = 0; i < N * D; i++) {
        if (fabs(X[i]) > max_X)
            max_X = fabs(X[i]);
    }
    for (int i = 0; i < N * D; i++)
        X[i] /= max_X;

    // Compute input similarities for exact t-SNE
    if (exact) {
        // Compute similarities
        m_ext_system.logger().logInfo("Exact?");
        P.resize(N * N);
        computeGaussianPerplexity(X, N, D, P, perplexity);

        // Symmetrize input similarities
        m_ext_system.logger().logInfo("Symmetrizing...");
        int nN = 0;
        for (int n = 0; n < N; n++) {
            int mN = (n + 1) * N;
            for (int m = n + 1; m < N; m++) {
                P[nN + m] += P[mN + n];
                P[mN + n] = P[nN + m];
                mN += N;
            }
            nN += N;
        }
        double sum_P = .0;
        for (int i = 0; i < N * N; i++)
            sum_P += P[i];
        for (int i = 0; i < N * N; i++)
            P[i] /= sum_P;
    }

    // Compute input similarities for approximate t-SNE
    else {

        // Compute asymmetric pairwise input similarities
        if (params.use_nn_graph) {
            computeGaussianPerplexityWithNNGraph(params, row_P, col_P, val_P, graph);
        } else {
            computeGaussianPerplexity(params, X, row_P, col_P, val_P);
        }

        // Symmetrize input similarities
        symmetrizeMatrix(row_P, col_P, val_P, N);
        double sum_P = .0;
        for (int i = 0; i < row_P[N]; i++)
            sum_P += val_P[i];
        for (int i = 0; i < row_P[N]; i++)
            val_P[i] /= sum_P;
    }
    clock_t end = clock();

    // Lie about the P-values
    if (exact) {
        for (int i = 0; i < N * N; i++)
            P[i] *= 12.0;
    } else {
        for (int i = 0; i < row_P[N]; i++)
            val_P[i] *= 12.0;
    }

    // Initialize solution (randomly)
    if (!skip_random_init) {
        for (int i = 0; i < N * no_dims; i++)
            Y[i] = randn() * .0001;
    }

    if (exact)
    {
        m_ext_system.logger().logInfo("Input similarities computed in " + std::to_string((float)(end - start) / CLOCKS_PER_SEC) + "seconds.");
        m_ext_system.logger().logInfo("Learning embedding...");
    }
    else {
        m_ext_system.logger().logInfo("Input similarities computed in " + std::to_string((float)(end - start) / CLOCKS_PER_SEC) + "seconds, "
        + "sparsity=" + std::to_string((double) row_P[N] / ((double) N * (double) N)));
        m_ext_system.logger().logInfo("Learning embedding...");
    }
    start = clock();

    return std::make_unique<TSNEState>(start, momentum, final_momentum, eta, dY, uY, gains, P, row_P, col_P, val_P, Y, 0);
}

void viskit::embed::cast::tsne::loop(const TSNEParams& params, TSNEState& state, const core::System& m_ext_system)
{
    int stop_lying_iter = params.stop_lying_iter;
    int mom_switch_iter = params.mom_switch_iter;
    int iter = state.iter;

    clock_t start = clock();
    int N = params.N;
    int no_dims = params.output_dims;
    int max_iter = params.max_iter;
    double theta = params.theta;

    bool exact = params.is_exact();
    if (exact)
        computeExactGradient(state.P, state.Y, N, no_dims, state.dY);
    else
        computeGradient(state.row_P, state.col_P, state.val_P, state.Y, N, no_dims, state.dY, theta);

    // Update gains
    for (int i = 0; i < N * no_dims; i++)
        state.gains[i] = (sign(state.dY[i]) != sign(state.uY[i])) ? (state.gains[i] + .2) : (state.gains[i] * .8);
    for (int i = 0; i < N * no_dims; i++)
        if (state.gains[i] < .01)
            state.gains[i] = .01;

    // Perform gradient update (with momentum and gains)
    for (int i = 0; i < N * no_dims; i++)
        state.uY[i] = state.momentum * state.uY[i] - state.eta * state.gains[i] * state.dY[i];
    for (int i = 0; i < N * no_dims; i++)
        state.Y[i] = state.Y[i] + state.uY[i];

    // Make solution zero-mean
    zeroMean(state.Y, N, no_dims);

    // Stop lying about the P-values after a while, and switch momentum
    if (iter == stop_lying_iter) {
        if (exact) {
            for (int i = 0; i < N * N; i++)
                state.P[i] /= 12.0;
        } else {
            for (int i = 0; i < state.row_P[N]; i++)
                state.val_P[i] /= 12.0;
        }
    }
    if (iter == mom_switch_iter)
        state.momentum = state.final_momentum;

    // Print out progress
    if (iter > 0 && (iter % 50 == 0 || iter == max_iter - 1)) {
        clock_t end = clock();
        double C;
        if (exact)
            C = evaluateError(state.P, state.Y, N, no_dims);
        else
            C = evaluateError(state.row_P, state.col_P, state.val_P, state.Y, N, no_dims, theta); // doing approximate computation here!
        if (iter == 0)
            m_ext_system.logger().logInfo("[t-SNE Caster] Iteration: " + std::to_string(iter + 1) + ", error is" + std::to_string(C) + "\n");
        else {
            printf("Iteration %d: error is %f (50 iterations in approx %4.2f seconds)\n", iter, C, 50 * (float)(end - start) / CLOCKS_PER_SEC);
        }
        start = clock();
    }

    state.iter++;
}

void viskit::embed::cast::tsne::finalize(TSNEState& state)
{
    clock_t end = clock();
    float total_time = (float)(end - state.start) / CLOCKS_PER_SEC;

    printf("Fitting performed in %4.2f seconds.\n", total_time);
}

static void computeGradient(std::vector<unsigned int>& inp_row_P, std::vector<unsigned int>& inp_col_P, std::vector<double>& inp_val_P, std::vector<double>& Y, int N, int D, vector<double>& dC, double theta)
{

    // Construct space-partitioning tree on current map
    auto tree = std::make_unique<SPTree>(D, Y.data(), N);

    // Compute all terms required for t-SNE gradient
    double sum_Q = .0;
    std::vector<double> pos_f(N * D);
    std::vector<double> neg_f(N * D);

    tree->computeEdgeForces(inp_row_P.data(), inp_col_P.data(), inp_val_P.data(), N, pos_f.data());
    for (int n = 0; n < N; n++)
        tree->computeNonEdgeForces(n, theta, neg_f.data() + n * D, &sum_Q);

    // Compute final t-SNE gradient
    for (int i = 0; i < N * D; i++) {
        dC[i] = pos_f[i] - (neg_f[i] / sum_Q);
    }
}

// Compute gradient of the t-SNE cost function (exact)
static void computeExactGradient(vector<double>& P, vector<double>& Y, int N, int D, vector<double>& dC)
{

    // Make sure the current gradient contains zeros
    for (int i = 0; i < N * D; i++)
        dC[i] = 0.0;

    // Compute the squared Euclidean distance matrix
    std::vector<double> DD(N * N);

    computeSquaredEuclideanDistance(Y, N, D, DD);

    // Compute Q-matrix and normalization sum
    std::vector<double> Q(N * N);
    double sum_Q = .0;
    int nN = 0;
    for (int n = 0; n < N; n++) {
        for (int m = 0; m < N; m++) {
            if (n != m) {
                Q[nN + m] = 1 / (1 + DD[nN + m]);
                sum_Q += Q[nN + m];
            }
        }
        nN += N;
    }

    // Perform the computation of the gradient
    nN = 0;
    int nD = 0;
    for (int n = 0; n < N; n++) {
        int mD = 0;
        for (int m = 0; m < N; m++) {
            if (n != m) {
                double mult = (P[nN + m] - (Q[nN + m] / sum_Q)) * Q[nN + m];
                for (int d = 0; d < D; d++) {
                    dC[nD + d] += (Y[nD + d] - Y[mD + d]) * mult;
                }
            }
            mD += D;
        }
        nN += N;
        nD += D;
    }
}

static double evaluateError(vector<double>& P, std::vector<double>& Y, int N, int D)
{
    // Compute the squared Euclidean distance matrix
    std::vector<double> DD(N * N);
    std::vector<double> Q(N * N);
    computeSquaredEuclideanDistance(Y, N, D, DD);

    // Compute Q-matrix and normalization sum
    int nN = 0;
    double sum_Q = DBL_MIN;
    for (int n = 0; n < N; n++) {
        for (int m = 0; m < N; m++) {
            if (n != m) {
                Q[nN + m] = 1 / (1 + DD[nN + m]);
                sum_Q += Q[nN + m];
            } else
                Q[nN + m] = DBL_MIN;
        }
        nN += N;
    }
    for (int i = 0; i < N * N; i++)
        Q[i] /= sum_Q;

    // Sum t-SNE error
    double C = .0;
    for (int n = 0; n < N * N; n++) {
        C += P[n] * log((P[n] + FLT_MIN) / (Q[n] + FLT_MIN));
    }

    return C;
}

static double evaluateError(std::vector<unsigned int>& row_P, std::vector<unsigned int>& col_P, std::vector<double>& val_P, std::vector<double>& Y, int N, int D, double theta)
{
    // Get estimate of normalization term
    auto tree = std::make_unique<SPTree>(D, Y.data(), N);
    std::vector<double> buff(D);

    double sum_Q = .0;
    for (int n = 0; n < N; n++)
        tree->computeNonEdgeForces(n, theta, buff.data(), &sum_Q);

    // Loop over all edges to compute t-SNE error
    int ind1, ind2;
    double C = .0, Q;
    for (int n = 0; n < N; n++) {
        ind1 = n * D;
        for (int i = row_P[n]; i < row_P[n + 1]; i++) {
            Q = .0;
            ind2 = col_P[i] * D;
            for (int d = 0; d < D; d++)
                buff[d] = Y[ind1 + d];
            for (int d = 0; d < D; d++)
                buff[d] -= Y[ind2 + d];
            for (int d = 0; d < D; d++)
                Q += buff[d] * buff[d];
            Q = (1.0 / (1.0 + Q)) / sum_Q;
            C += val_P[i] * log((val_P[i] + FLT_MIN) / (Q + FLT_MIN));
        }
    }

    return C;
}

// Compute input similarities with a fixed perplexity
static void computeGaussianPerplexity(vector<double>& X, int N, int D, vector<double>& P, double perplexity)
{

    // Compute the squared Euclidean distance matrix
    vector<double> DD(N * N);
    computeSquaredEuclideanDistance(X, N, D, DD);

    // Compute the Gaussian kernel row by row
    int nN = 0;
    for (int n = 0; n < N; n++) {

        // Initialize some variables
        bool found = false;
        double beta = 1.0;
        double min_beta = -DBL_MAX;
        double max_beta = DBL_MAX;
        double tol = 1e-5;
        double sum_P;

        // Iterate until we found a good perplexity
        int iter = 0;
        while (!found && iter < 200) {

            // Compute Gaussian kernel row
            for (int m = 0; m < N; m++)
                P[nN + m] = exp(-beta * DD[nN + m]);
            P[nN + n] = DBL_MIN;

            // Compute entropy of current row
            sum_P = DBL_MIN;
            for (int m = 0; m < N; m++)
                sum_P += P[nN + m];
            double H = 0.0;
            for (int m = 0; m < N; m++)
                H += beta * (DD[nN + m] * P[nN + m]);
            H = (H / sum_P) + log(sum_P);

            // Evaluate whether the entropy is within the tolerance level
            double Hdiff = H - log(perplexity);
            if (Hdiff < tol && -Hdiff < tol) {
                found = true;
            } else {
                if (Hdiff > 0) {
                    min_beta = beta;
                    if (max_beta == DBL_MAX || max_beta == -DBL_MAX)
                        beta *= 2.0;
                    else
                        beta = (beta + max_beta) / 2.0;
                } else {
                    max_beta = beta;
                    if (min_beta == -DBL_MAX || min_beta == DBL_MAX) {
                        if (beta < 0) {
                            beta *= 2;
                        } else {
                            beta = beta <= 1.0 ? -0.5 : beta / 2.0;
                        }
                    } else {
                        beta = (beta + min_beta) / 2.0;
                    }
                }
            }

            // Update iteration counter
            iter++;
        }

        // Row normalize P
        for (int m = 0; m < N; m++)
            P[nN + m] /= sum_P;
        nN += N;
    }
}

// Compute input similarities with a fixed perplexity using ball trees
static void computeGaussianPerplexity(const viskit::embed::cast::TSNEParams& params, vector<double>& X, std::vector<unsigned int>& row_P, std::vector<unsigned int>& col_P, std::vector<double>& val_P)
{
    double perplexity = params.perplexity;
    int N = params.N;
    int D = params.input_dims;
    int K = params.neighbor_count;
    if (params.perplexity > K)
        printf("Perplexity should be lower than K!\n");

    row_P.resize(N + 1);
    col_P.resize(N * K);
    val_P.resize(N * K);
    vector<double> cur_P(N - 1);

    row_P[0] = 0;
    for (int n = 0; n < N; n++)
        row_P[n + 1] = row_P[n] + (unsigned int)K;

    // Build ball tree on data set
    auto tree = std::make_unique<VpTree<DataPoint, euclidean_distance>>();
    vector<DataPoint> obj_X(N, DataPoint(D, -1, X.data()));
    for (int n = 0; n < N; n++)
        obj_X[n] = DataPoint(D, n, X.data() + n * D);
    tree->create(obj_X);

    // Loop over all points to find nearest neighbors
    printf("Building tree...\n");
    vector<DataPoint> indices;
    vector<double> distances;
    for (int n = 0; n < N; n++) {

        if (n % 10000 == 0)
            printf(" - point %d of %d\n", n, N);

        // Find nearest neighbors
        indices.clear();
        distances.clear();
        tree->search(obj_X[n], K + 1, &indices, &distances);

        // Initialize some variables for binary search
        bool found = false;
        double beta = 1.0;
        double min_beta = -DBL_MAX;
        double max_beta = DBL_MAX;
        double tol = 1e-5;

        // Iterate until we found a good perplexity
        int iter = 0;
        double sum_P;
        while (!found && iter < 200) {

            // Compute Gaussian kernel row
            for (int m = 0; m < K; m++)
                cur_P[m] = exp(-beta * distances[m + 1] * distances[m + 1]);

            // Compute entropy of current row
            sum_P = DBL_MIN;
            for (int m = 0; m < K; m++)
                sum_P += cur_P[m];
            double H = .0;
            for (int m = 0; m < K; m++)
                H += beta * (distances[m + 1] * distances[m + 1] * cur_P[m]);
            H = (H / sum_P) + log(sum_P);

            // Evaluate whether the entropy is within the tolerance level
            double Hdiff = H - log(perplexity);
            if (Hdiff < tol && -Hdiff < tol) {
                found = true;
            } else {
                if (Hdiff > 0) {
                    min_beta = beta;
                    if (max_beta == DBL_MAX || max_beta == -DBL_MAX)
                        beta *= 2.0;
                    else
                        beta = (beta + max_beta) / 2.0;
                } else {
                    max_beta = beta;
                    if (min_beta == -DBL_MAX || min_beta == DBL_MAX)
                        beta /= 2.0;
                    else
                        beta = (beta + min_beta) / 2.0;
                }
            }

            // Update iteration counter
            iter++;
        }

        // Row-normalize current row of P and store in matrix
        for (unsigned int m = 0; m < K; m++)
            cur_P[m] /= sum_P;
        for (unsigned int m = 0; m < K; m++) {
            col_P[row_P[n] + m] = (unsigned int)indices[m + 1].index();
            val_P[row_P[n] + m] = cur_P[m];
        }
    }
}

// Compute input similarities using supplied NN graph.
static void computeGaussianPerplexityWithNNGraph(const viskit::embed::cast::TSNEParams& params, std::vector<unsigned int>& row_P, std::vector<unsigned int>& col_P, std::vector<double>& val_P, Graph& graph)
{
    int N = params.N;
    int K = params.neighbor_count;
    double perplexity = params.perplexity;

    row_P.resize(N + 1);
    col_P.resize(N * K);
    val_P.resize(N * K);
    std::vector<double> cur_P(N - 1);
    vector<double> distances;

    for (int n = 0; n < N; n++) {
        row_P[n] = n * K;

        distances.clear();

        if (graph.getNeighbors(n)->empty()) {
            break;
        }

        auto neighbors = graph.getNeighbors(n).value();

        int i = 0;
        for (auto neighbor : neighbors) {
            col_P[n * K + i] = neighbor.j;

            auto dist = neighbor.r;
            distances.push_back(dist);
            i++;
        }

        // Initialize some variables for binary search
        bool found = false;
        double beta = 1.0;
        double min_beta = -DBL_MAX;
        double max_beta = DBL_MAX;
        double tol = 1e-5;

        // Iterate until we found a good perplexity
        int iter = 0;
        double sum_P;
        while (!found && iter < 200) {

            // Compute Gaussian kernel row
            for (int m = 0; m < K; m++)
                cur_P[m] = exp(-beta * distances[m] * distances[m]);

            // Compute entropy of current row
            sum_P = DBL_MIN;
            for (int m = 0; m < K; m++)
                sum_P += cur_P[m];
            double H = .0;
            for (int m = 0; m < K; m++)
                H += beta * (distances[m] * distances[m] * cur_P[m]);
            H = (H / sum_P) + log(sum_P);

            // Evaluate whether the entropy is within the tolerance level
            double Hdiff = H - log(perplexity);
            if (Hdiff < tol && -Hdiff < tol) {
                found = true;
            } else {
                if (Hdiff > 0) {
                    min_beta = beta;
                    if (max_beta == DBL_MAX || max_beta == -DBL_MAX)
                        beta *= 2.0;
                    else
                        beta = (beta + max_beta) / 2.0;
                } else {
                    max_beta = beta;
                    if (min_beta == -DBL_MAX || min_beta == DBL_MAX)
                        beta /= 2.0;
                    else
                        beta = (beta + min_beta) / 2.0;
                }
            }

            // Update iteration counter
            iter++;
        }

        // Row-normalize current row of P and store in matrix
        for (unsigned int m = 0; m < K; m++)
            cur_P[m] /= sum_P;
        for (unsigned int m = 0; m < K; m++) {
            val_P[row_P[n] + m] = cur_P[m];
        }
    }
}

// Symmetrizes a sparse matrix
static void symmetrizeMatrix(std::vector<unsigned int>& row_P, std::vector<unsigned int>& col_P, std::vector<double>& val_P, int N)
{
    // Count number of elements and row counts of symmetric matrix
    std::vector<int> row_counts(N);
    for (int n = 0; n < N; n++) {
        for (int i = row_P[n]; i < row_P[n + 1]; i++) {

            // Check whether element (col_P[i], n) is present
            bool present = false;
            for (int m = row_P[col_P[i]]; m < row_P[col_P[i] + 1]; m++) {
                if (col_P[m] == n)
                    present = true;
            }
            if (present)
                row_counts[n]++;
            else {
                row_counts[n]++;
                row_counts[col_P[i]]++;
            }
        }
    }
    int no_elem = 0;
    for (int n = 0; n < N; n++)
        no_elem += row_counts[n];

    // Allocate memory for symmetric matrix
    std::vector<unsigned int> sym_row_P(N + 1);
    std::vector<unsigned int> sym_col_P(no_elem);
    std::vector<double> sym_val_P(no_elem);

    // Construct new row indices for symmetric matrix
    sym_row_P[0] = 0;
    for (int n = 0; n < N; n++)
        sym_row_P[n + 1] = sym_row_P[n] + (unsigned int)row_counts[n];

    // Fill the result matrix
    vector<int> offset(N);
    for (int n = 0; n < N; n++) {
        for (unsigned int i = row_P[n]; i < row_P[n + 1]; i++) { // considering element(n, col_P[i])

            // Check whether element (col_P[i], n) is present
            bool present = false;
            for (unsigned int m = row_P[col_P[i]]; m < row_P[col_P[i] + 1]; m++) {
                if (col_P[m] == n) {
                    present = true;
                    if (n <= col_P[i]) { // make sure we do not add elements twice
                        sym_col_P[sym_row_P[n] + offset[n]] = col_P[i];
                        sym_col_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = n;
                        sym_val_P[sym_row_P[n] + offset[n]] = val_P[i] + val_P[m];
                        sym_val_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = val_P[i] + val_P[m];
                    }
                }
            }

            // If (col_P[i], n) is not present, there is no addition involved
            if (!present) {
                sym_col_P[sym_row_P[n] + offset[n]] = col_P[i];
                sym_col_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = n;
                sym_val_P[sym_row_P[n] + offset[n]] = val_P[i];
                sym_val_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = val_P[i];
            }

            // Update offsets
            if (!present || n <= col_P[i]) {
                offset[n]++;
                if (col_P[i] != n)
                    offset[col_P[i]]++;
            }
        }
    }

    // Divide the result by two
    for (int i = 0; i < no_elem; i++)
        sym_val_P[i] /= 2.0;

    // Return symmetric matrices
    row_P.assign(sym_row_P.begin(), sym_row_P.end());
    col_P.assign(sym_col_P.begin(), sym_col_P.end());
    val_P.assign(sym_val_P.begin(), sym_val_P.end());
}

static void computeSquaredEuclideanDistance(std::vector<double>& X, int N, int D, vector<double>& DD_vector)
{
    const double* XnD = X.data();
    double* DD = DD_vector.data();
    for (int n = 0; n < N; ++n, XnD += D) {
        const double* XmD = XnD + D;
        double* curr_elem = &DD[n * N + n];
        *curr_elem = 0.0;
        double* curr_elem_sym = curr_elem + N;
        for (int m = n + 1; m < N; ++m, XmD += D, curr_elem_sym += N) {
            *(++curr_elem) = 0.0;
            for (int d = 0; d < D; ++d) {
                *curr_elem += (XnD[d] - XmD[d]) * (XnD[d] - XmD[d]);
            }
            *curr_elem_sym = *curr_elem;
        }
    }
}

// Makes data zero-mean
static void zeroMean(vector<double>& X, int N, int D)
{
    // Compute data mean
    vector<double> mean(D);
    int nD = 0;

    for (int n = 0; n < N; n++) {
        for (int d = 0; d < D; d++) {
            mean[d] += X[nD + d];
        }
        nD += D;
    }
    for (int d = 0; d < D; d++) {
        mean[d] /= (double)N;
    }

    // Subtract data mean
    nD = 0;
    for (int n = 0; n < N; n++) {
        for (int d = 0; d < D; d++) {
            X[nD + d] -= mean[d];
        }
        nD += D;
    }
}

// Generates a Gaussian random number
static double randn()
{
    double x, y, radius;
    do {
        x = 2 * (rand() / ((double)RAND_MAX + 1)) - 1;
        y = 2 * (rand() / ((double)RAND_MAX + 1)) - 1;
        radius = (x * x) + (y * y);
    } while ((radius >= 1.0) || (radius == 0.0));
    radius = sqrt(-2 * log(radius) / radius);
    x *= radius;
    y *= radius;
    return x;
}
