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
htbm_options.htbm_flag = false;

%% Run the algorithm
addpath('../Algorithms/BFGS_hybrid/')
[x_arr,bundle_cell] = BFGS_method_hybrid(problem_data,algo_options,N_inner_iters,N_outer_iters,htbm_options);

j_max = N_outer_iters;

%% Visualization

x_min = zeros(n,1);

lw = 1.25;
ms = 13;
figure_flag = 1;

if(figure_flag) % ----------------------------------------------------------
    fig1 = figure;
else
    subplot(1,3,1)
end
 
h = plot(1:j_max,log10(vecnorm(x_arr - x_min,2,1)),'k.-','MarkerSize',ms,'LineWidth',lw);

legend(h,{'$\| x^j - x^* \|$'},'Interpreter','latex','FontSize',18,'Location','ne');

xlim([1,j_max])
ylim([-4.5,0.25])

xticks(1:2:j_max);
yticks(-8:1:2);

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
    export_fig 'plot_BFGS_bundling_1' '-png' '-r500' %'-transparent'
    close(fig1)
end

if(figure_flag) % ----------------------------------------------------------
    fig2 = figure;
else
    subplot(1,3,2)
end

unique_inds_arr = zeros(1,j_max);
crit_arr = zeros(1,j_max);
num_sample_pts = zeros(1,j_max);
qp_optns = optimoptions('quadprog','Display','none');
lp_optns = optimoptions('linprog','Display','none');
for j = 1:j_max
    num_sample_pts(j) = size(bundle_cell{j}.sample_pts,2);
    act_arr = zeros(1,num_sample_pts(j));

    W_grad_min = zeros(n,num_sample_pts(j));
    for i = 1:num_sample_pts(j)
        [~,act_arr(i)] = f(bundle_cell{j}.sample_pts(:,i));
        W_grad_min(:,i) = grad_f_cell{act_arr(i)}(x_min);
    end

    unique_inds_arr(j) = numel(unique(act_arr));

    Q = W_grad_min'*W_grad_min;
    alpha = quadprog(Q,[],[],[],ones(1,num_sample_pts(j)),1,zeros(1,num_sample_pts(j)),ones(1,num_sample_pts(j)),[],qp_optns);
    crit_arr(j) = norm(W_grad_min*alpha,2);

    % Use inf-norm version for increased accuracy when zero lies in conv. hull
    lp_f = [zeros(n,1);1];
    lp_A = [-W_grad_min',-ones(num_sample_pts(j),1)];
    lp_b = zeros(num_sample_pts(j),1);
    lp_lb = [-ones(1,n),-Inf];
    lp_ub = [ones(1,n),Inf];

    [lp_sol,~,~,~,lp_lambda] = linprog(lp_f,lp_A,lp_b,[],[],lp_lb,lp_ub,lp_optns);

    crit_arr(j) = min(norm(W_grad_min*alpha,2),norm(W_grad_min*lp_lambda.ineqlin,2));
end

h = plot(1:j_max,log10(crit_arr),'k.-','MarkerSize',ms,'LineWidth',lw);

legend(h,{'$\theta^*(W_j)$'},'Interpreter','latex','FontSize',18,'Location','ne');

xlim([1,j_max])
ylim([-15,1])

xticks(1:2:j_max);
yticks(-15:2:1)

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
    export_fig 'plot_BFGS_bundling_2' '-png' '-r500' %'-transparent'
    close(fig2)
end

if(figure_flag) % ----------------------------------------------------------
    fig3 = figure;
else
    subplot(1,3,3)
end

h1 = plot(1:j_max,unique_inds_arr,'k.-','MarkerSize',ms,'LineWidth',lw);
hold on
h2 = plot(1:j_max,num_sample_pts,'k.--','MarkerSize',ms,'LineWidth',lw);
yline(min([n+1,k]),'r--','LineWidth',lw);

legend([h1,h2],{'$|s(W_j)|$','$|W_j|$'},'Interpreter','latex','FontSize',18,'Location','se');

xlim([1,j_max])
ylim([14,42]);

xticks(1:2:j_max);

xlabel('$j$','Interpreter','latex');

set(gca,'linewidth',1.1)
set(gca,'fontsize',15)
axis square
grid on

if(save_figure_flag)
    export_fig 'plot_BFGS_bundling_3' '-png' '-r500' %'-transparent'
    close(fig3)
end