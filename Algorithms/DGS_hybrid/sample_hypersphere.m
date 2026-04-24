function pts = sample_hypersphere(n,N)
%   Returns the zero-vector in pts(:,1) and N-1 randomly (uniformly) 
%   sampled points from the n-dimensional unit sphere (with respect to the 
%   2-norm) in pts(:,2:N). 

    pts(:,1) = zeros(n,1);

    tmp = randn(n,N-1);
    tmp = tmp./vecnorm(tmp,2,1);
    pts(:,2:N) = (rand(1,N-1).^(1/n)).*tmp;
    
end

