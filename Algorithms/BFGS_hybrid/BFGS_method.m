% A simple implementation of the BFGS method (see, e.g., Alg. 6.1 in
% [Nocedal, Wright (2006)]). For the BFGS update, the formula (2.2) from
% [Lewis, Overton (2013)] is used. The stopping criteria are the step
% length being too small or the descent being too shallow.
%
% When the inputs are symbolic (e.g., in vpa) then the method performs
% symbolic computation. (This is very expensive and only useful for
% research.) 

function output = BFGS_method(problem_data,algo_options)

% Read inputs
f = problem_data.f;
grad_f = problem_data.grad_f;
n = problem_data.n;
x0 = problem_data.x0;

N_iter = algo_options.N_iter;
H0 = algo_options.H0;
reset_period = algo_options.reset_period;
step_threshold = algo_options.step_threshold;
descent_threshold = algo_options.descent_threshold;
disp_flag = algo_options.disp_flag;

% Initialization
x_arr = zeros(n,N_iter+1);
f_arr = zeros(1,N_iter+1);
grad_arr = zeros(n,N_iter+1);

t_arr = zeros(1,N_iter);
y_arr = zeros(n,N_iter);
s_arr = zeros(n,N_iter);
p_arr = zeros(n,N_iter);

% Convert arrays to symbolic arrays if input is symbolic
if(isa(x0,"sym"))
    x_arr = sym(x_arr);
    f_arr = sym(f_arr);
    grad_arr = sym(grad_arr);

    t_arr = sym(t_arr);
    y_arr = sym(y_arr);
    s_arr = sym(s_arr);
    p_arr = sym(p_arr);
end

x_arr(:,1) = x0;
f_arr(1) = f(x0);
grad_arr(:,1) = grad_f(x0);

H_cell = cell(1,N_iter+1);
H_cell{1} = H0;

if(disp_flag > 0)
    disp('Running BFGS method...')
    fprintf('    Iter%% = ');
    start_tic = tic;
    print_iter_arr = [1,ceil((0.01:0.01:1) .* N_iter)];
end

% Loop over k
for k = 1:N_iter

    if(disp_flag > 0 && ismember(k,print_iter_arr))
        fprintf([repmat('\b',1,6*(k > 1)),'%.4f',repmat('\n',1,k == N_iter)], k/N_iter);
    end

    % Compute search direction
    p_arr(:,k) = -H_cell{k} * grad_arr(:,k);

    % Compute Wolfe step length
    [t_arr(k),f_arr(k+1),grad_arr(:,k+1)] = linesearch_armijo_wolfe(x_arr(:,k),p_arr(:,k),f_arr(k),grad_arr(:,k),problem_data,algo_options);

    % Update iterate
    x_arr(:,k+1) = x_arr(:,k) + t_arr(k)*p_arr(:,k);

    % Update H via inverse BFGS update
    y_arr(:,k) = grad_arr(:,k+1) - grad_arr(:,k);
    s_arr(:,k) = x_arr(:,k+1) - x_arr(:,k);

    V = eye(n) - (p_arr(:,k)'*y_arr(:,k))^(-1) * p_arr(:,k)*y_arr(:,k)';
    H_cell{k+1} = V*H_cell{k}*V' + t_arr(k)*(p_arr(:,k)'*y_arr(:,k))^(-1) * (p_arr(:,k)*p_arr(:,k)');

    % Ensure symmetry
    H_cell{k+1} = (H_cell{k+1} + H_cell{k+1}')/2;

    % Restarts, if reset_period < Inf. (The reset index is shifted by 1
    % compared to the pseudo code due to Matlab arrays starting at 1
    % instead of 0.)
    if(mod(k,reset_period) == 0)
        H_cell{k+1} = H0;
    end

    % Stopping criterion (step length)
    if(t_arr(k) < step_threshold)
        if(disp_flag > 0)
            fprintf('\n\n');
            disp(['    Stopped because step length too small. (t_k = ',num2str(double(t_arr(k))),')'])
        end
        break
    end

    % Stopping criterion (descent)
    if(-grad_arr(:,k)' * H_cell{k} * grad_arr(:,k) > -descent_threshold)
        if(disp_flag > 0)
            fprintf('\n\n');
            disp(['    Stopped because descent too shallow. (∇f(x^k)^T * p^k = ',num2str(double(grad_arr(:,k)'*p_arr(:,k))),')'])
        end
        break
    end

    if(disp_flag > 0 && k == N_iter)
        fprintf('\n');
        disp(' ')
    end

end

% Prepare output
output.x_arr = x_arr(:,1:k+1);
output.f_arr = f_arr(1:k+1);
output.grad_arr = grad_arr(:,1:k+1);
output.H_cell = H_cell(1:k+1);
output.t_arr = t_arr(1:k);
output.y_arr = y_arr(:,1:k);
output.s_arr = s_arr(:,1:k);
output.p_arr = p_arr(:,1:k);

if(disp_flag > 0)
    runtime = toc(start_tic);
    disp(['    f value = ',num2str(double(f_arr(k+1)))]);
    disp(['    iters   = ',num2str(k)]);
    disp(['    time    = ',num2str(runtime)]);
end

end
