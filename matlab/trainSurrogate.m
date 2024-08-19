function [supp, coeffs, vals, z_test, estimate] = trainSurrogate(sampler, z_min, z_max, estimator_kind, Smax, N_test, N_memory, is_system_selfadjoint, varargin)
%TRAINSURROGATE   Train surrogate model.
%   [SUPP, COEFFS, VALS, Z_TEST, ESTIMATE] = TRAINSURROGATE(SAMPLER, Z_MIN, Z_MAX, ESTIMATOR_KIND, SMAX, N_TEST, N_MEMORY, IS_SYSTEM_SELFADJOINT, VARARGIN)
%       trains a surrogate model for high-fidelity function SAMPLER for
%       Z_MIN <= z <= Z_MAX. An estimator of kind ESTIMATOR_KIND is used to
%       drive the adaptive sampling in the greedy Loewner framework, and
%       VARARGIN contains arguments needed for its initialization. At most
%       SMAX high-fidelity samples are taken. N_TEST candidate sample
%       points are taken. N_MEMORY memory terms are considered: the
%       algorithm terminates successfully only if the estimator yields a
%       "pass" at N_MEMORY successive iterations.
    [tol, delta] = varargin{1:2};
    if strcmp(estimator_kind, "lookahead")
        estimator_kind = "lookaheadbatch";
        batch_size = 1;
    elseif strcmp(estimator_kind, "lookaheadbatch")
        batch_size = varargin{3};
    elseif strcmp(estimator_kind, "random")
        [batch_size, seed] = varargin{3:4};
    end
    z_test = logspace(log10(z_min), log10(z_max), N_test).';

    if strcmp(estimator_kind, "random") % initialize estimator
        rng(seed)
        estimator_z = 10.^(log10(z_test(1)) + (log10(z_test(end)) - log10(z_test(1))) * rand(batch_size, 1));
        estimator_sample1 = samplerEff(sampler, estimator_z(1));
        estimator_samples = inf(batch_size, numel(estimator_sample1));
        estimator_samples(1, :) = estimator_sample1;
        for j = 2:batch_size
            estimator_samples(j, :) = samplerEff(sampler, estimator_z(j));
        end
        fprintf("computed %d extra test samples\n", batch_size);
        estimator_samples_norm = sum(abs(estimator_samples).^2, 2).^.5;
    end

    % first sample
    z_sample = z_test(1);
    z_test(1) = []; % remove initial sample point from test set
    y = samplerEff(sampler, z_sample);
    fprintf("1: sampled at z=%ej\n", z_sample);
    supp = z_sample; coeffs = 1; vals = y;
    if is_system_selfadjoint
        yC = conj(y);
    else
        yC = samplerEff(sampler, -z_sample);
        fprintf("1: sampled at z=%ej\n", -z_sample);
    end
    L = .5j * (yC - y) / z_sample; % Loewner matrix
    if ~is_system_selfadjoint; valsC = yC; end

    % adaptivity loop
    n_memory = 0;
    for i = 1:Smax % max number of samples
        if strcmp(estimator_kind, "random") % evaluate error at random points
            estimator_approx = barycentricEvaluate(estimator_z, supp, coeffs, vals);
            error = computeError(estimator_approx, estimator_samples, delta, estimator_samples_norm);
            [estimator_error_inf, idx] = max(error);
            estimator_error_z = estimator_z(idx);
            fprintf("error at z=%ej is %e\n", estimator_error_z, estimator_error_inf);
            if estimator_error_inf >= tol
                n_memory = 0; % error is too large
            else
                fprintf("termination check passed\n")
                n_memory = n_memory + 1; % error is below tolerance
            end
        end

        % termination check
        if n_memory >= N_memory; break; end % enough small errors in a row

        % find next sample point
        indicator = computeIndicator(z_test, supp, coeffs, vals);
        [~, idx_sample] = max(indicator);

        if strcmp(estimator_kind, "lookaheadbatch") % get batch of test points
            estimator_z_idx = idx_sample * ones(batch_size, 1);
            estimator_z = z_test(idx_sample) * ones(batch_size, 1);
            for j = 2:batch_size
                indicator = indicator .* abs(z_test - estimator_z(j - 1));
                [~, estimator_z_idx(j)] = max(indicator);
                estimator_z(j) = z_test(estimator_z_idx(j));
            end
        end

        % get next sample
        z_sample = z_test(idx_sample);
        z_test(idx_sample) = []; % remove sample point from test set
        y = samplerEff(sampler, z_sample);
        fprintf("%d: sampled at z=%ej\n", numel(supp) + 1, z_sample);
    
        if strcmp(estimator_kind, "lookaheadbatch") % get batch of test samples
            estimator_approx = barycentricEvaluate(estimator_z, supp, coeffs, vals);
            estimator_samples = inf(batch_size, numel(y));
            estimator_samples(1, :) = y;
            for j = 2:batch_size
                estimator_samples(j, :) = samplerEff(sampler, estimator_z(j));
            end
            if batch_size > 1
                fprintf("computed %d extra test samples\n", batch_size - 1);
            end
            error = computeError(estimator_approx, estimator_samples, delta);
            [estimator_error_inf, idx] = max(error);
            estimator_error_z = estimator_z(idx);
            fprintf("error at z=%ej is %e\n", estimator_error_z, estimator_error_inf);
            if estimator_error_inf >= tol
                n_memory = 0; % error is too large
            else
                fprintf("termination check passed\n")
                n_memory = n_memory + 1; % error is below tolerance
            end
        end
        % update surrogate with new support points and values
        supp = [supp; z_sample]; vals = [vals, y];

        % update Loewner matrix
        if is_system_selfadjoint
            yC = conj(y);
        else
            yC = samplerEff(sampler, -z_sample);
            fprintf("%d: sampled at z=%ej\n", numel(supp), -z_sample);
        end
        L1 = 1j * bsxfun(@rdivide, (yC - vals).', z_sample + supp).';
        if is_system_selfadjoint
            l1 = conj(L1(:, 1 : end-1));
        else
            l1 = 1j * bsxfun(@rdivide, (valsC - y).', z_sample + supp(1 : end - 1)).';
            valsC = [valsC, yC];
        end
        L = [L l1(:); L1];

        % update surrogate with new barycentric coefficients
        [~, R] = qr(L, 0);
        [~, ~, V] = svd(R);
        coeffs = V(:, end);
    end
    fprintf("greedy loop terminated at %d samples\n", numel(supp));

    if nargout >= 5 % compute error estimate
        if strcmp(estimator_kind, "lookaheadbatch")
            indicator = computeIndicator(z_test, supp, coeffs, vals);
            [~, idx_sample] = max(indicator);
            % get batch of test points
            estimator_z_idx = idx_sample * ones(batch_size, 1);
            estimator_z = z_test(idx_sample) * ones(batch_size, 1);
            for j = 2:batch_size
                indicator = indicator .* abs(z_test - estimator_z(j - 1));
                [~, estimator_z_idx(j)] = max(indicator);
                estimator_z(j) = z_test(estimator_z_idx(j));
            end
            % get test samples
            estimator_approx = barycentricEvaluate(estimator_z, supp, coeffs, vals);
            estimator_samples = inf(batch_size, numel(y));
            for j = 1:batch_size
                estimator_samples(j, :) = samplerEff(sampler, estimator_z(j));
            end
            fprintf("computed %d extra test samples for final estimate\n", batch_size);
            % get max test error
            error = computeError(estimator_approx, estimator_samples, delta);
            [estimator_error_inf, idx] = max(error);
            estimator_error_z = estimator_z(idx);
            fprintf("error at z=%ej is %e\n", estimator_error_z, estimator_error_inf);
        elseif strcmp(estimator_kind, "random")
            if n_memory < N_memory % abnormal termination (else estimate is already available)
                % evaluate error at random points
                estimator_approx = barycentricEvaluate(estimator_z, supp, coeffs, vals);
                error = computeError(estimator_approx, estimator_samples, delta, estimator_samples_norm);
                [estimator_error_inf, idx] = max(error);
                estimator_error_z = estimator_z(idx);
                fprintf("error at z=%ej is %e\n", estimator_error_z, estimator_error_inf);
            end
        end
        estimate = estimator_error_inf * computeIndicator(z_test, supp, coeffs, vals) / computeIndicator(estimator_error_z, supp, coeffs, vals);
    end
end

function [y] = samplerEff(sampler, z)
    y = sampler(1j * z);
    y = y(:);
end