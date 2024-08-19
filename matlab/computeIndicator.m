function [i] = computeIndicator(z, supp, coeffs, vals)
    i = abs(barycentricEvaluate(z, supp, coeffs, vals, true)).^-1;
end