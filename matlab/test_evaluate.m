clear; close all; clc
rng(42)

x = linspace(0, 1, 25).';
supp = linspace(0, 1, 4).';
supp = rand(4, 1);

coeffs = randn(4, 1);
vals = randn(10, 4);
vals = NaN;

y = barycentricEvaluate(x, supp, coeffs, vals, true);
figure()
plot(x, y)

z = barycentricEvaluate(x, supp, coeffs, vals);
figure()
plot(x, z)