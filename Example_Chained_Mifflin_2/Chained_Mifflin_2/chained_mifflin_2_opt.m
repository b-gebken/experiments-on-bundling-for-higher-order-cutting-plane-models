function x_opt = chained_mifflin_2_opt(n)

if(mod(n,2) == 0)
    input('Only works if n if odd.')
end

x_opt = [0.815219744122870;
    0.579151766624787;
    1/sqrt(2)*ones(n-3,1);
    0];

end

