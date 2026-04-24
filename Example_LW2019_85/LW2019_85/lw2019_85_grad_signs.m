function grad = lw2019_85_grad_signs(x,g_arr,H_cell,c_arr,signs)

    [n,k] = size(g_arr);
    grad = zeros(n,1);
    for i = 1:k
        grad = grad + signs(i) * (g_arr(:,i) + H_cell{i}*x + c_arr(i)/24 * 4*norm(x,2)^2 * x);
    end

end

