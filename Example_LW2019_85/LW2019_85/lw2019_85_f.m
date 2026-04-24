% Function (8.5) from [Lewis, Wylie (2019), "A simple Newton method for
% local nonsmooth optimization"] 

function [val,signs] = lw2019_85_f(x,g_arr,H_cell,c_arr)

    k = size(g_arr,2);

    tmp = zeros(k,1);
    for i = 1:k
        tmp(i) = g_arr(:,i)'*x + 1/2 * x'*H_cell{i}*x + c_arr(i)/24 * norm(x,2)^4;
    end
    
    val = sum(abs(tmp));
    signs = sign(tmp);
    signs(signs == 0) = 1;
end

