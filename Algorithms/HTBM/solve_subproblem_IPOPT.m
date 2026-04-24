% Function for solving the subproblem (3.5) in [GU2026a] via IPOPT. For
% details, see  
%   [Wächter, Biegler (2006), "On the Implementation of a Primal-Dual Interior Point Filter Line Search Algorithm for Large-Scale Nonlinear Programming"]
%   https://github.com/ebertolazzi/mexIPOPT
%   https://ethz.ch/content/dam/ethz/special-interest/mavt/dynamic-systems-n-control/idsc-dam/Research_Onder/Downloads/IPOPT/IPOPT_MatlabInterface_V0p1.pdf

% ------------------------------- IMPORTANT -------------------------------
% The results shown in our articles use a modified version of mexIPOPT,
% where the check whether Matlab or Octave is used is disabled. More
% precisely, line 251 in "mexIPOPT/toolbox/lib/ipopt.m" in the github (or
% "MATLAB Add-Ons\Toolboxes\IPOPT-toolbox\lib\ipopt.m" on the hard drive),
% which originally read
%   isOctave = isempty(ver('matlab'));
% was replaced by
%   isOctave = false;
%
% This significantly improved the runtime of mexIPOPT in our experiments.
% -------------------------------------------------------------------------

function [z_bar,theta,mu] = solve_subproblem_IPOPT(W,W_f,W_grad,W_hess,x,eps,sp_solver_options)

n = size(W_grad,1);
k = numel(W_hess);

% Define funcs struct for IPOPT
funcs.objective = @objective;
funcs.constraints = @constraints;
funcs.gradient = @gradient;
funcs.jacobian = @jacobian;
funcs.jacobianstructure = @jacobianstructure;
funcs.hessian = @hessian;
funcs.hessianstructure = @hessianstructure;

% Define auxdata struct containing the data for the subproblem
auxdata.n = n;
auxdata.k = k;
auxdata.eps = eps;
auxdata.x = x;
auxdata.W = W; 
auxdata.W_f = W_f;
auxdata.W_grad = W_grad;
auxdata.W_hess = W_hess;

% Set IPOPT options
IPOPT_opts.lb = [(x - eps);-Inf]';
IPOPT_opts.ub = [(x + eps);Inf]';
IPOPT_opts.cl = -Inf(1,k+1);
IPOPT_opts.cu = zeros(1,k+1);

IPOPT_opts.auxdata = auxdata;
IPOPT_opts.ipopt.max_iter = 10000;
IPOPT_opts.ipopt.tol = sp_solver_options.tol;
IPOPT_opts.ipopt.constr_viol_tol = sp_solver_options.constr_viol_tol;
IPOPT_opts.ipopt.print_level = 0;  

IPOPT_opts.ipopt.line_search_method = 'cg-penalty';

% Define initial point for IPOPT. (x is modified to avoid an error in the
% restoration phase of IPOPT, which was likely caused by the gradient of
% the eps-ball constraint being zero in x.)  
x_mod = x + 1/2*eps*[1;zeros(n-1,1)];
init_pt = [x_mod',max(funcs.constraints([x_mod',0],auxdata)) + 1];

% Run the solver
[sol_IPOPT, info_IPOPT] = ipopt_auxdata(init_pt, funcs, IPOPT_opts);

% Check IPOPT status
if(info_IPOPT.status ~= 0)
    fprintf(2,['Warning: IPOPT status error (',num2str(info_IPOPT.status),'). (max hess eigval = ',num2str(max(cellfun(@(x) eigs(x,1),W_hess))),')\n'])
end

% Prepare the output
z_bar = sol_IPOPT(1:n)';
nonlcon_sol = funcs.constraints(sol_IPOPT,auxdata);
theta = max(nonlcon_sol(1:k) + sol_IPOPT(n+1));
mu = info_IPOPT.lambda(end);

end

% Auxiliary functions
function f = objective(in,auxdata)
    f = in(auxdata.n + 1);
end

function g = gradient(in,auxdata)
    g = [zeros(auxdata.n,1);1]';
end

function c = constraints(in,auxdata)
	k = auxdata.k;
    n = auxdata.n;
    
    tmp1 = in(1:n)' - auxdata.W;
    c = [(auxdata.W_f + sum(auxdata.W_grad .* tmp1,1) - in(n+1))'; ...
        0];

    tmp2 = zeros(k,1);
    for i = 1:k
        tmp2(i) = tmp1(:,i)'*auxdata.W_hess{i}*tmp1(:,i);
    end
    c(1:k) = c(1:k) + 1/2*tmp2;
    c(k+1) = norm(in(1:n)' - auxdata.x,2)^2 - auxdata.eps^2;

    %%% Slower, but more readable version
    % k = auxdata.k;
    % c = zeros(k+1,1);
    % n = auxdata.n;
    % 
    % for i = 1:k
    %     c(i) = auxdata.W_f(i) + auxdata.W_grad(:,i)'*(in(1:n)' - auxdata.W(:,i)) + 1/2*(in(1:n)' - auxdata.W(:,i))'*auxdata.W_hess{i}*(in(1:n)' - auxdata.W(:,i)) - in(n+1);
    % end
    % c(k+1) = norm(in(1:n)' - auxdata.x,2)^2 - auxdata.eps^2;
end

function J = jacobian(in,auxdata)
    k = auxdata.k;
    n = auxdata.n;
    
    J = [auxdata.W_grad',-ones(k,1);
        2*(in(1:n) - auxdata.x'),0];
    tmp1 = in(1:n)' - auxdata.W;

    tmp2 = zeros(k,n);
    for i = 1:k
        tmp2(i,:) = tmp1(:,i)'*auxdata.W_hess{i};
    end
    J(1:k,1:n) = J(1:k,1:n) + tmp2; 
    
    J = sparse(J);

    %%% Slower, but more readable version
    % k = auxdata.k;
    % n = auxdata.n;
    % J = zeros(k+1,n+1);
    % for i = 1:k
    %     J(i,:) = [auxdata.W_grad(:,i) + auxdata.W_hess{i}*(in(1:n)' - auxdata.W(:,i)); -1]';
    % end
    % J(k+1,:) = [2*(in(1:n)' - auxdata.x);0]';
    % 
    % J = sparse(J);
end

function J_str = jacobianstructure(auxdata)
    k = numel(auxdata.W_hess);
    n = auxdata.n;
    J_str = sparse(ones(k+1,n+1));
end

function H = hessian (in, sigma, lambda, auxdata)
    k = numel(auxdata.W_hess);
    n = auxdata.n;
    H = zeros(n+1);

    tmp = zeros(n);
    for i = 1:k
        tmp = tmp + lambda(i)*auxdata.W_hess{i};
    end
    H = [tmp,zeros(n,1);zeros(1,n+1)];
    
    H(1:n,1:n) = H(1:n,1:n) + lambda(k+1)*2*eye(n);

    H = sparse(tril(H));
end

function H_str = hessianstructure(auxdata)
    n = auxdata.n;
    H_str = sparse([ones(n,n),zeros(n,1); zeros(1,n), 1]);
    H_str = sparse(tril(H_str));
end