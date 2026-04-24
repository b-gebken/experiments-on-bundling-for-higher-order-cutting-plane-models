function [y,I] = chained_mifflin_2(x)

    n = size(x,1);
    I = NaN(n-1,1);
    
    y = 0;
    for i = 1:n-1
        y = y + -x(i) + 2*(x(i)^2 + x(i+1)^2 - 1) + 1.75*abs(x(i)^2 + x(i+1)^2 - 1);
        I(i) = sign(x(i)^2 + x(i+1)^2 - 1);
        if(I(i) == 0)
            I(i) = 1;
        end
    end   
end

