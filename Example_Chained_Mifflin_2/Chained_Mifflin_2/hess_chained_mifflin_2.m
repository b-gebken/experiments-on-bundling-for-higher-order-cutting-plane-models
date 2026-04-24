function hess = hess_chained_mifflin_2(x)
    
    [~,I] = chained_mifflin_2(x);
    
    n = size(x,1);
    hess = zeros(n);

    for i = 1:n-1
        hess(i,i) = hess(i,i) + (4 + I(i)*1.75*2);
        hess(i+1,i+1) = hess(i+1,i+1) + (4 + I(i)*1.75*2); 
    end
end

