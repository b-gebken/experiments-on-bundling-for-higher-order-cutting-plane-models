function grad = grad_max_fun(x,f_cell,grad_f_cell)

    [~,I] = max_fun(x,f_cell);

    grad = grad_f_cell{I}(x);

end

