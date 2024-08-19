% The file mna_4.mat belongs to the SLICOT Benchmark Examples for Model Reduction
% ( Y. Chahlaoui, P. Van Dooren,
%   Benchmark Examples for Model Reduction of Linear Time-Invariant Dynamical Systems, Dimension Reduction of Large-Scale Systems,
%   Lecture Notes in Computational Science and Engineering, vol 45: 379--392, 2005.)
clear; close all; clc
load("mna_4.mat", "E", "A", "B")
problem_E = E; problem_A = A; problem_B = full(B);
sampler = @(z) problem_B' * ((z * problem_E - problem_A) \ problem_B);
is_system_selfadjoint = true;

z_min = 3e4; z_max = 3e9;
Smax = 1000; N_test = 10000; N_memory = 1;
tol = 1e-3; delta = 1e-8;

% train surrogate model
estimator_kind = "lookahead";
[supp, coeffs, vals, z_test, estimate] = trainSurrogate(sampler, z_min, z_max, estimator_kind, Smax, N_test, N_memory, is_system_selfadjoint, tol, delta);
postprocess(sampler, z_test, estimate, supp, coeffs, vals, estimator_kind, z_min, z_max, tol, delta);

function [] = postprocess(sampler, z_test, estimate, supp, coeffs, vals, estimator_kind, z_min, z_max, tol, delta)
    % predict and compute errors
    z_post = logspace(log10(z_min), log10(z_max), 101);
    H_exact = inf(numel(z_post), size(vals, 1));
    for j = 1:numel(z_post)
        sample = sampler(1j * z_post(j));
        H_exact(j, :) = sample(:);
    end
    H_approx = barycentricEvaluate(z_post, supp, coeffs, vals);
    H_err = computeError(H_approx, H_exact, delta) + 1e-16;

    % make plots
    figure()
    loglog(z_post, abs(H_exact))
    hold all
    loglog(z_post, abs(H_approx), '--')
    loglog(supp, abs(vals)', 'o')
    xlabel("Im(z)"), xlabel("|H|")
    title(estimator_kind)
    figure()
    loglog(z_post, H_err)
    hold all
    loglog(z_test, estimate, ":")
    loglog([z_min, z_max], [tol, tol], '--')
    legend("error", "estimator")
    xlabel("Im(z)"), xlabel("relative error")
    title(estimator_kind)
end
