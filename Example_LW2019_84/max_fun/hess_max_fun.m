function grad = hess_max_fun(x,f_cell,hess_f_cell)

    [~,I] = max_fun(x,f_cell);

    grad = hess_f_cell{I}(x);

end

