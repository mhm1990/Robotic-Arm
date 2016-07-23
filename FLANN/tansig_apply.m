function [a] = tansig_apply(n)
    a = 2 ./ (1 + exp(-2*n)) - 1;
end


