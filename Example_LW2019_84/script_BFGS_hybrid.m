save_figure_flag = 0;

%% LW2019, (8.4)
addpath('max_fun');
n = 50;
k = 40;

rng("default");

lambda = rand(k,1); lambda = lambda/sum(lambda);
tmp = 2*rand(n,k)-1; g_arr = tmp - tmp*lambda;

H_cell = cell(1,k);
for i = 1:k
    tmp = 2*rand(n)-1;
    H_cell{i} = tmp * tmp';
end

c_arr = rand(1,k);

f_cell = cell(1,k);
grad_f_cell = cell(1,k);
hess_f_cell = cell(1,k);
for i = 1:k
    f_cell{i} = @(x) g_arr(:,i)'*x + 1/2 * x'*H_cell{i}*x + c_arr(i)/24 * norm(x,2)^4;
    grad_f_cell{i} = @(x) g_arr(:,i) + H_cell{i}*x + c_arr(i)/24 * 4*norm(x,2)^2 * x;
    hess_f_cell{i} = @(x) H_cell{i} + c_arr(i)/24 * (8*(x*x') + diag(4*norm(x,2)^2 * ones(1,n)));
end

f = @(x) max_fun(x,f_cell);
grad_f = @(x) grad_max_fun(x,f_cell,grad_f_cell);
hess_f = @(x) hess_max_fun(x,f_cell,hess_f_cell);

%% Set problem data
problem_data.n = n;
problem_data.f = f;
problem_data.grad_f = grad_f;
problem_data.hess_f = hess_f;

problem_data.x0 = ones(n,1);

%% Set options for BFGS method

algo_options.N_iter = 200;
algo_options.H0 = eye(n);
algo_options.c1 = 0.0001;
algo_options.c2 = 0.5;
algo_options.reset_period = Inf;
algo_options.step_threshold = -1; % Disabling stopping condition
algo_options.descent_threshold = -1; % Disabling stopping condition
algo_options.disp_flag = 1;

N_inner_iters = 40;
N_outer_iters = 15;

%% Set parameters for HTBM
htbm_options.htbm_flag = true;
htbm_options.eps = 10^2;
htbm_options.sp_solver_options.tol = 10^-15;
htbm_options.sp_solver_options.constr_viol_tol = 10^-15;

%% Run the algorithm
addpath('../Algorithms/BFGS_hybrid/')
addpath('../Algorithms/HTBM/')
[x_arr,bundle_cell,htbm_success_flags] = BFGS_method_hybrid(problem_data,algo_options,N_inner_iters,N_outer_iters,htbm_options);

j_max = N_outer_iters;

%% Visualization

x_min = zeros(n,1);

lw = 1.25;
ms = 13;

fig = figure;

h = plot(1:j_max,log10(vecnorm(x_arr - x_min,2,1)),'k.-','MarkerSize',ms,'LineWidth',lw);
hold on
plot(find(htbm_success_flags),log10(vecnorm(x_arr(:,htbm_success_flags) - x_min,2,1)),'ko','MarkerSize',8,'LineWidth',1.1);

legend(h,{'$\| x^j - x^* \|$'},'Interpreter','latex','FontSize',18,'Location','ne');

xlim([1,j_max])
ylim([-14.25,0.75])

xticks(1:2:j_max);
yticks(-14:2:2);

xlabel('$j$','Interpreter','latex');

% Log tick labeling (y)
old_ticks = yticks;
new_ticks_cell = cell(numel(old_ticks),1);
for i = 1:numel(old_ticks)
    new_ticks_cell{i} = ['10^{',num2str(old_ticks(i)),'}'];
end
yticklabels(new_ticks_cell)

set(gca,'linewidth',1.1)
set(gca,'fontsize',15)
axis square
grid on

if(save_figure_flag)
    export_fig 'plot_BFGS_hybrid' '-png' '-r500' %'-transparent'
    close(fig)
end