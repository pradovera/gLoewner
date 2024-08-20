% The file iss.mat belongs to the SLICOT Benchmark Examples for Model Reduction
% ( Y. Chahlaoui, P. Van Dooren,
%   Benchmark Examples for Model Reduction of Linear Time-Invariant Dynamical Systems, Dimension Reduction of Large-Scale Systems,
%   Lecture Notes in Computational Science and Engineering, vol 45: 379--392, 2005.)
clear; close all; clc
load("iss.mat", "A", "B", "C")
problem_A = A; problem_E = speye(size(problem_A, 1));
problem_B = full(B); problem_C = full(C);
sampler = @(z) problem_C * ((z * problem_E - problem_A) \ problem_B);
is_system_selfadjoint = true;

z_min = 1e-1; z_max = 5e1;
Smax = 1000; N_test = 10000; N_memory = 3;
tol = 1e-3; delta = 1e-8;

% train surrogate model
estimator_kind = "lookahead";
z_test = logspace(log10(z_min), log10(z_max), N_test);
[supp, coeffs, vals, z_test, estimate] = trainSurrogate(sampler, z_test, [1 N_test], estimator_kind, Smax, N_memory, is_system_selfadjoint, tol, delta);
postprocess(sampler, z_test, estimate, supp, coeffs, vals, estimator_kind, z_min, z_max, tol, delta);

% train surrogate model
estimator_kind = "lookaheadbatch";
z_test = logspace(log10(z_min), log10(z_max), N_test);
[supp, coeffs, vals, z_test, estimate] = trainSurrogate(sampler, z_test, [1 N_test], estimator_kind, Smax, N_memory, is_system_selfadjoint, tol, delta, 5);
postprocess(sampler, z_test, estimate, supp, coeffs, vals, estimator_kind, z_min, z_max, tol, delta);

% train surrogate model
estimator_kind = "random";
z_test = logspace(log10(z_min), log10(z_max), N_test);
[supp, coeffs, vals, z_test, estimate] = trainSurrogate(sampler, z_test, [1 N_test], estimator_kind, Smax, N_memory, is_system_selfadjoint, tol, delta, 100, 42);
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
