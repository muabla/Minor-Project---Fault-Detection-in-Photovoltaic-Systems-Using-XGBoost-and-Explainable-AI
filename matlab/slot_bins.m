function [counts, values] = slot_bins(light, K)
%SLOT_BINS  Puts the cells of one substring into K "slots" (Solar Cell blocks).
%   Cells in series can be reordered without changing the circuit, so cells are sorted by light and split
%   into K groups of similar light (optimal 1-D split, least squared error). Each slot = one Solar Cell block
%   with counts(k) cells, all receiving values(k) (fraction of the irradiance G).
%   Tested in Python: with K = 4 the model's prediction is identical to the per-cell simulation.
v = sort(light(:)); n = numel(v); K = min(K, n);
c1 = [0; cumsum(v)]; c2 = [0; cumsum(v .^ 2)];
D = inf(K + 1, n + 1); B = zeros(K + 1, n + 1); D(1, 1) = 0;
for k = 1:K
    for j = k:n
        for i = k - 1:j - 1
            cnt = j - i;
            cost = (c2(j + 1) - c2(i + 1)) - (c1(j + 1) - c1(i + 1)) ^ 2 / cnt;
            c = D(k, i + 1) + cost;
            if c < D(k + 1, j + 1)
                D(k + 1, j + 1) = c; B(k + 1, j + 1) = i;
            end
        end
    end
end
edges = zeros(1, K + 1); edges(K + 1) = n; j = n;
for k = K:-1:1
    j = B(k + 1, j + 1); edges(k) = j;
end
counts = diff(edges);
values = zeros(1, K);
for k = 1:K
    values(k) = mean(v(edges(k) + 1:edges(k + 1)));
end
end
