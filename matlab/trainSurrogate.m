function [supp, coeffs, vals, z_test, estimate] = trainSurrogate(sampler, z_test, start_z_idx, estimator_kind, Smax, N_memory, is_system_selfadjoint, varargin)
%TRAINSURROGATE   Train surrogate model.
%   [SUPP, COEFFS, VALS, Z_TEST, ESTIMATE] = TRAINSURROGATE(SAMPLER, Z_TEST, START_Z_IDX, ESTIMATOR_KIND, SMAX, N_MEMORY, IS_SYSTEM_SELFADJOINT, VARARGIN)
%       trains a surrogate model for high-fidelity function SAMPLER(z).
%       Note: SAMPLER is evaluated at 1j*z. An estimator of kind
%       ESTIMATOR_KIND is used to drive the adaptive sampling in the greedy
%       Loewner framework, and VARARGIN contains arguments needed for its
%       initialization. At most SMAX high-fidelity samples are taken.
%       Sample points are taken from the list of candidates Z_TEST.
%       N_MEMORY memory terms are considered: the algorithm terminates
%       successfully only if the estimator yields a "pass" at N_MEMORY
%       successive iterations.
    [tol, delta] = varargin{1:2};
    if strcmp(estimator_kind, "lookahead")
        estimator_kind = "lookaheadbatch";
        batch_size = 1;
    elseif strcmp(estimator_kind, "lookaheadbatch")
        batch_size = varargin{3};
    elseif strcmp(estimator_kind, "random")
        [batch_size, seed] = varargin{3:4};
    end
    z_test = z_test(:);
    start_z_idx = start_z_idx(:);
    if numel(start_z_idx) == 0; start_z_idx = 1; end
    start_z_idx = unique(start_z_idx, "sorted");
    start_z_idx = start_z_idx(end:-1:1);
    
    is_system_selfadjoint = (is_system_selfadjoint && all(imag(z_test) == 0,"all"));

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

    % get initial samples
    for j = 1:numel(start_z_idx)
        i = start_z_idx(j);
        [z_sample, z_test, y, yC] = getSample(sampler, z_test, i, j, is_system_selfadjoint);

        if j == 1 % initial sample
            size_y = numel(y);
            if ~is_system_selfadjoint
                valsC = inf(size_y, numel(start_z_idx));
            end
        end
        if ~is_system_selfadjoint; valsC(:, j) = yC; end

        if j == 1 % initialize rational function
            supp = inf(numel(start_z_idx), 1);
            vals = inf(size_y, numel(start_z_idx));
        end
        supp(j) = z_sample; vals(:, j) = y; % update rational function

        if j == 1 % initialize Loewner matrix
            L = inf(size_y * numel(start_z_idx), numel(start_z_idx));
        end
        % update Loewner matrix
        L_new_oldandnew = 1j * bsxfun(@rdivide, (yC - vals(:, 1 : j)).', z_sample + supp(1 : j)).';
        if is_system_selfadjoint
            L_old_new = conj(L_new_oldandnew(:, 1 : end - 1));
        else
            L_old_new = 1j * bsxfun(@rdivide, (valsC - y).', z_sample + supp(1 : end - 1)).';
        end
        L(1 : (j - 1) * size_y, j) = L_old_new(:);
        L((j - 1) * size_y + 1 : j * size_y, 1 : j) = L_new_oldandnew;
    end
    coeffs = getBarycentricCoeffs(L);

    % adaptivity loop
    n_memory = 0;
    for i = numel(supp):Smax % max number of samples
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
        [z_sample, z_test, y, yC] = getSample(sampler, z_test, idx_sample, i, is_system_selfadjoint);
    
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
        L_new_oldandnew = 1j * bsxfun(@rdivide, (yC - vals).', z_sample + supp).';
        if is_system_selfadjoint
            L_old_new = conj(L_new_oldandnew(:, 1 : end - 1));
        else
            L_old_new = 1j * bsxfun(@rdivide, (valsC - y).', z_sample + supp(1 : end - 1)).';
        end
        L = [L L_old_new(:); L_new_oldandnew];

        coeffs = getBarycentricCoeffs(L);
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

function y = samplerEff(sampler, z)
    y = sampler(1j * z);
    y = y(:);
end

function [z_sample, z_test, y, yC] = getSample(sampler, z_test, idx, j, is_system_selfadjoint)
    z_sample = z_test(idx);
    z_test(idx) = []; % remove sample point from test set
    y = samplerEff(sampler, z_sample);
    fprintf("%d: sampled at z=%ej\n", j, z_sample);
    if is_system_selfadjoint
        yC = conj(y);
    else
        yC = samplerEff(sampler, -z_sample);
        fprintf("%d: sampled at z=%ej\n", j, -z_sample);
    end
end

function coeffs = getBarycentricCoeffs(L)
    [~, R] = qr(L, 0);
    [~, ~, V] = svd(R);
    coeffs = V(:, end);
end