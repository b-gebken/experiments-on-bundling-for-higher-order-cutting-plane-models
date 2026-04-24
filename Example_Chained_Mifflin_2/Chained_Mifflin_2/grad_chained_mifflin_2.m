function grad = grad_chained_mifflin_2(x)
    
    n = size(x,1);
    grad = zeros(n,1);
    [~,I] = chained_mifflin_2(x);

    for i = 1:n-1
        grad(i) = grad(i) + (-1 + 4*x(i) + I(i)*1.75*2*x(i));
        grad(i+1) = grad(i+1) + (4*x(i+1) + I(i)*1.75*2*x(i+1));
    end
    
end

