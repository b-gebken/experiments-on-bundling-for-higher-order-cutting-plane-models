save_figure_flag = 0;

%% Chained Mifflin 2
addpath('Chained_Mifflin_2');
n = 51;

f = @(x) chained_mifflin_2(x);
grad_f = @(x) grad_chained_mifflin_2(x);
hess_f = @(x) hess_chained_mifflin_2(x);

%% Set problem data
problem_data.n = n;
problem_data.f = f;
problem_data.subgrad_f = grad_f;
problem_data.hess_f = hess_f;

problem_data.x0 = ones(n,1);

%% Set parameters for DGS 
j_max = 7;
kappa_eps = 0.1;
eps0 = 10;
eps_fun = @(j) eps0*kappa_eps.^(j-1);
del_fun = eps_fun;
dgs_options.eps_arr = eps_fun(1:j_max);
dgs_options.delta_arr = 10^-5*ones(1,j_max);

dgs_options.rand_sample_N = 0;
dgs_options.memory_size = 100;
dgs_options.c = 0.5;
dgs_options.ls_flag = 'armijo';
dgs_options.max_iter = 10000;
dgs_options.disp_flag = 2;

%% Set parameters for HTBM
htbm_options.htbm_flag = true;
htbm_options.eps = 10^2;
htbm_options.sp_solver_options.tol = 10^-15;
htbm_options.sp_solver_options.constr_viol_tol = 10^-15;

%% Run the algorithm
addpath('../Algorithms/DGS_hybrid/')
addpath('../Algorithms/HTBM/')
[x_opt,f_opt,x_cell,eval_counter,bundle_cell,htbm_success_flags] = eps_descent_method_hybrid(problem_data,dgs_options,htbm_options);

x_arr = zeros(n,j_max);
for j = 1:j_max
    x_arr(:,j) = x_cell{j}(:,end);
end

%% Visualization

x_min = chained_mifflin_2_opt(n);

lw = 1.25;
ms = 13;

fig = figure;

h = plot(1:j_max,log10(vecnorm(x_arr - x_min,2,1)),'k.-','MarkerSize',ms,'LineWidth',lw);
hold on
plot(find(htbm_success_flags),log10(vecnorm(x_arr(:,htbm_success_flags) - x_min,2,1)),'ko','MarkerSize',8,'LineWidth',1.1);

legend(h,{'$\| \tilde{x}^j - x^* \|$'},'Interpreter','latex','FontSize',18,'Location','ne');

xlim([1,j_max])
ylim([-7.5,0.25])

xticks(1:1:j_max);
yticks(-14:1:2);

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
    export_fig 'plot_DGS_hybrid' '-png' '-r500' %'-transparent'
    close(fig)
end