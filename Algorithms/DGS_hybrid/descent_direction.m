% Algorithm 2 in [GP2021] (or Algorithm 4.2 in [G2022]) combined with
% Algorithm 2 in [G2024b]. (See eps_descent_method.m for the input and the
% reference list.)

function [v,f_eps_v,memory,eval_counter,bundle] = descent_direction(x,f_x,f,subgrad_f,eps,delta,c,rand_sample_N,memory,qp_optns,eval_counter)

W = [];
bundle.sample_pts = [];

% Add subgradients at sample points in B_eps(x) from memory
if(memory.max_size > 0 && ~isempty(memory.sample_pts))
    inside_Beps = vecnorm(memory.sample_pts - x,2,1) <= eps + 10^-14; % Tolerance for machine precision
    W = [W,memory.subgrads(:,inside_Beps)];
    bundle.sample_pts = [bundle.sample_pts, memory.sample_pts(:,inside_Beps)];
end

if(rand_sample_N == 0)
    % Step 1 in [GP2021] (Deterministic initial approximation)
    subgrad_f_x = subgrad_f(x);
    W = [W,subgrad_f_x]; eval_counter.subgrad_eval = eval_counter.subgrad_eval + 1;
    if(memory.max_size > 0)
        memory = add_to_memory(x,subgrad_f_x,memory);
    end

    bundle.sample_pts = [bundle.sample_pts,x];
else
    % Random initial approximation
    n = size(x,1);
    W_rnd = zeros(n,rand_sample_N);
    rand_init_pts = x + eps*sample_hypersphere(n,rand_sample_N);
    for i = 1:rand_sample_N
        W_rnd(:,i) = subgrad_f(rand_init_pts(:,i)); eval_counter.subgrad_eval = eval_counter.subgrad_eval + 1;
    end
    W = [W,W_rnd];
    if(memory.max_size > 0)
        memory = add_to_memory(rand_init_pts,W_rnd,memory);
    end

    bundle.sample_pts = [bundle.sample_pts,rand_init_pts];
end

while(1)
    % Step 2 in [GP2021] (Compute direction based on W)
    num_subgrad = size(W,2);
    Q = W'*W;
    alpha = quadprog(Q,[],[],[],ones(1,num_subgrad),1,zeros(1,num_subgrad),ones(1,num_subgrad),[],qp_optns);
    v = -W*alpha;
    
    % Step 3 in [GP2021] (Stopping criterion)
    if(norm(v,2) <= delta)
        f_eps_v = NaN;
        break
    end
    
    % Step 4 in [GP2021] (Check for sufficient decrease)
    f_eps_v = f(x + eps/norm(v,2)*v); eval_counter.f_eval = eval_counter.f_eval + 1;
    if(f_eps_v - f_x > -c*eps*norm(v,2))
        % Apply Algorithm 2 from [G2024b]
        c_min = -(f_eps_v - f_x)/(eps*norm(v,2));
        c_tilde = (c + c_min)/2;
        
        % Step 1 in [G2024b] (Initialization)
        a = 0; 
        b = eps/norm(v,2);
        t = (a+b)/2;
        bis_flag = 0; % 0 - Start; 1 - Right; 2 - Left
        
        while(1)
            % Step 2 in [G2024b] (Evaluate subgradient)
            xi = subgrad_f(x + t*v); eval_counter.subgrad_eval = eval_counter.subgrad_eval + 1;
            
            % Step 3 in [G2024b] (Check if xi is in conv(W) and if not, add it to W and stop)
            if(xi'*v > -c*norm(v,2)^2)
                W = [W,xi];
                bundle.sample_pts = [bundle.sample_pts,x + t*v];
                
                if(memory.max_size > 0)
                    memory = add_to_memory(x + t*v,xi,memory);
                end

                break
            end
            
            % Step 4 in [G2024b] (Perform bisection of [a,b])
            if(bis_flag == 2)
                h_b = h_t;
            elseif(bis_flag == 0)
                h_b = f_eps_v - f_x + c_tilde*b*norm(v,2)^2;
            end
            h_t = f(x + t*v) - f_x + c_tilde*t*norm(v,2)^2; eval_counter.f_eval = eval_counter.f_eval + 1;
            
            if(h_b > h_t)
                a = t;
                bis_flag = 1;
            else
                b = t;
                bis_flag = 2;
            end
            
            % Step 5 in [G2024b] (Update t)
            t = (a+b)/2;
        end
    else
        break
    end
    
end

bundle.subgrads = W;

end

