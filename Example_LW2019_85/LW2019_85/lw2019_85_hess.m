function hess = lw2019_85_hess(x,g_arr,H_cell,c_arr)
    
    [~,signs] = lw2019_85_f(x,g_arr,H_cell,c_arr);

    [n,k] = size(g_arr);
    hess = zeros(n);
    for i = 1:k
        hess = hess + signs(i) * (H_cell{i} + c_arr(i)/24 * (8*(x*x') + diag(4*norm(x,2)^2 * ones(1,n))));
    end

end

