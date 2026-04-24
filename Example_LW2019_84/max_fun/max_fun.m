function [y,I] = max_fun(x,f_cell)

    k = numel(f_cell);
    f_x_arr = zeros(1,k);
    for i = 1:k
        f_x_arr(i) = f_cell{i}(x);
    end

    [y,I] = max(f_x_arr);

end

