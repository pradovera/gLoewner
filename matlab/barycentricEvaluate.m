function y = barycentricEvaluate(x, supp, coeffs, vals, only_den, tol)
%BARYCENTRICEVALUATE   Evaluate barycentric function.
%   Y = BARYCENTRICEVALUATE(X, SUPP, COEFFS, VALS, ONLY_DEN, TOL) evaluates
%     the rational function with coefficients COEFFS at support points
%     SUPPS and support values VALS. The tolerance TOL is used for robust
%     management of almost-zero values. If ONLY_DEN is true, only the
%     denominator is evaluated.
    if nargin<4 || any(isnan(vals),"all")
        no_vals = true;
    else
        no_vals = false;
    end
    if nargin<5, only_den = false; end
    if nargin<6, tol = 1e-10; end
    x = reshape(x, 1, []);
    dx = x - reshape(supp, [], 1);
    [N, M] = size(dx);
    % check if some x is too close to a support point
    % (we compute the samples at such bad x by hand)
    [~, idxs_dx_min] = min(abs(dx), [], 1);
    bad_idxs = []; bad_idx_vals = [];
    for j = 1:M
        idx_dx_min = idxs_dx_min(j);
        if abs(dx(idx_dx_min, j)) < tol
            bad_idxs = [bad_idxs; j];
            if ~only_den
                if no_vals
                    bad_idx_vals = [bad_idx_vals; canonical_j(idx_dx_min, N)];
                else
                    bad_idx_vals = [bad_idx_vals; vals(:, idx_dx_min).'];
                end
            end
        end
    end
    good_idxs = 1:M;
    for k = bad_idxs(end:-1:1)
        good_idxs(k) = [];
    end
    % numerator coefficients (to be multiplied by vals)
    num = bsxfun(@times, dx(:, good_idxs).^-1, coeffs);
    den = sum(num, 1);
    if only_den
        y = inf(M, 1);
        y(good_idxs) = den;
        return
    end
    if no_vals
        y = inf(M, N);
    else
        y = inf(M, size(vals, 1));
    end
    % compute numerator values
    for j = 1:numel(bad_idxs)
        y(bad_idxs(j), :) = bad_idx_vals(j, :);
    end
    if no_vals
        y(good_idxs, :) = bsxfun(@rdivide, num, den).';
    else
        y(good_idxs, :) = bsxfun(@rdivide, vals * num, den).';
    end
end

function ej = canonical_j(j, n)
    ej = [zeros(1, j - 1) 1 zeros(1, n - j)];
end