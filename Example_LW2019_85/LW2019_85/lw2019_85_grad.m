function grad = lw2019_85_grad(x,g_arr,H_cell,c_arr)
    
    [~,signs] = lw2019_85_f(x,g_arr,H_cell,c_arr);

    [n,k] = size(g_arr);
    grad = zeros(n,1);
    for i = 1:k
        grad = grad + signs(i) * (g_arr(:,i) + H_cell{i}*x + c_arr(i)/24 * 4*norm(x,2)^2 * x);
    end

end

