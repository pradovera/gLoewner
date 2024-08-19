function [e] = computeError(ap, ex, delta, ex_norm)
    if nargin<4
        ex_norm = sum(abs(ex).^2, 2).^.5;
    end
    e = sum(abs(ap - ex).^2, 2).^.5 ./ (ex_norm + delta);
end
