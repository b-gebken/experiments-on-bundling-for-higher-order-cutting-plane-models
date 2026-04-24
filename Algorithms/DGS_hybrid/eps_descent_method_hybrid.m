		
function [x_opt,f_opt,x_cell,eval_counter,bundle_cell,htbm_success_flags] = eps_descent_method_hybrid(problem_data,dgs_options,htbm_options)

% Read input
n = problem_data.n;
f = problem_data.f;
subgrad_f = problem_data.subgrad_f;
x0 = problem_data.x0;

eps_arr = dgs_options.eps_arr;
delta_arr = dgs_options.delta_arr;
rand_sample_N = dgs_options.rand_sample_N;
memory_max_size = dgs_options.memory_size;
c = dgs_options.c;
ls_flag = dgs_options.ls_flag;
max_iter = dgs_options.max_iter;
disp_flag = dgs_options.disp_flag;

% Initialization
j_max = numel(eps_arr);
x_cell = cell(j_max,1);
bundle_cell = cell(1,j_max);
qp_optns = optimoptions('quadprog','Display','off','OptimalityTolerance',10^-16);

htbm_success_flags = false(1,j_max);

memory.sample_pts = [];
memory.subgrads = [];
memory.max_size = memory_max_size;

eval_counter.f_eval = 0;
eval_counter.subgrad_eval = 0;

if(disp_flag >= 1)
    start_tic = tic;
end

if(disp_flag >= 1)
    disp('Deterministic gradient sampling...')
end

% Loop over all (eps,delta) pairs
for j = 1:j_max
    
    if(disp_flag >= 2)
        disp(['    Iteration j = ',num2str(j),'/',num2str(j_max),'...']);
        disp(['        eps_j   = ',num2str(eps_arr(j))]);
        disp(['        delta_j = ',num2str(delta_arr(j))]);
        disp('        Running descent iterations...');
    end
    
    % Starting point is either x0 (if j = 1) or the final iterate of the
    % previous descent sequence.
    if(j == 1)
        x_arr = [x0,zeros(n,max_iter)];
        f_xi = f(x0); eval_counter.f_eval = eval_counter.f_eval + 1;
    else
        x_arr = [x_cell{j-1}(:,end),zeros(n,max_iter)];
        f_xi = f_x_new;
    end
    
    if(disp_flag >= 2)
        desc_tic = tic;
    end

    % Descent loop for fixed eps and delta
    for i = 1:max_iter
        
        if(disp_flag >= 3)
            tmp_counter = eval_counter.subgrad_eval;
            disp(['            Iteration i = ',num2str(i),' (j = ',num2str(j),')...']);
            disp(['                Computing descent direction...']);
        end
        
        % Step 2 in [G2024a]
        [v,f_eps_v,memory,eval_counter,bundle] = descent_direction(x_arr(:,i),f_xi,f,subgrad_f,eps_arr(j),delta_arr(j),c,rand_sample_N,memory,qp_optns,eval_counter);

        if(disp_flag >= 3)
            disp(['                    ...done!']);
            disp(['                Req. subgrad. eval.: ',num2str(eval_counter.subgrad_eval - tmp_counter)]);
        end
        
        % Step 3 in [G2024a]
        if(norm(v,2) <= delta_arr(j))
            % Step 4 in [G2024a]
            x_arr = x_arr(:,1:i);
            f_x_new = f_xi;
            break
        % Step 5 in [G2024a]
        else
            % Step 6 in [G2024a]
            if(strcmp(ls_flag,'eps'))
                t = eps_arr(j)/norm(v,2);
                f_x_new = f_eps_v;
            elseif(strcmp(ls_flag,'armijo') || strcmp(ls_flag,'armijo_normal'))
                if(strcmp(ls_flag,'armijo'))
                    t = 1;
                elseif(strcmp(ls_flag,'armijo_normal'))
                    t = 1/norm(v,2);
                end

                if(t <= eps_arr(j)/norm(v,2))
                    t = eps_arr(j)/norm(v,2);
                    f_x_new = f_eps_v;
                else
                    f_x_new = f(x_arr(:,i) + t*v); eval_counter.f_eval = eval_counter.f_eval + 1;
                    while(f_x_new - f_xi > -c*t*norm(v,2)^2)
                        t = t/2;
                        
                        if(t <= eps_arr(j)/norm(v,2))
                            t = eps_arr(j)/norm(v,2);
                            f_x_new = f_eps_v;
                            break;
                        end

                        f_x_new = f(x_arr(:,i) + t*v); eval_counter.f_eval = eval_counter.f_eval + 1;
                    end
                end
            end
 
            % Step 7 in [G2024a]
            x_arr(:,i+1) = x_arr(:,i) + t*v;
            
            if(disp_flag >= 3)
                disp(['                norm(v,2)  = ',num2str(norm(v,2)),' (delta_j = ',num2str(delta_arr(j)),', eps_j = ',num2str(eps_arr(j)),')']);
                disp(['                New f val. = ',num2str(f_x_new)]);
                disp(['                f decr.    = ',num2str(f_xi - f_x_new)]);
            end
            
            f_xi = f_x_new;
        end
    end
    
    if(disp_flag >= 2)
        desc_time = toc(desc_tic);
        disp(['        ...done in N_j = ',num2str(size(x_arr,2)-1),' iterations (in ',num2str(desc_time),'s).']);
    end
    
    if(i == max_iter)
        disp(['Warning: Maximum number of iterations reached for j = ',num2str(j),' before eps-delta-critical point was found.'])
    end

    % Apply HTBM step %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if(htbm_options.htbm_flag)
        hess_f = problem_data.hess_f;
    
        num_sample_pts = size(bundle.sample_pts,2);
        bundle.f = zeros(1,num_sample_pts);
        bundle.hess = cell(1,num_sample_pts);
        for i = 1:num_sample_pts
            bundle.f(i) = f(bundle.sample_pts(:,i));
            bundle.hess{i} = hess_f(bundle.sample_pts(:,i));
        end
    
        [z_bar,theta,mu] = solve_subproblem_IPOPT(bundle.sample_pts,bundle.f,bundle.subgrads,bundle.hess,x_arr(:,end),htbm_options.eps,htbm_options.sp_solver_options);
    
        f_z_bar = f(z_bar);
        if(f_z_bar < f_x_new)
            disp(['HTBM step successful! f_x = ',num2str(f_x_new),', f_z_bar = ',num2str(f_z_bar)])
            htbm_success_flags(j) = true;
            x_arr = [x_arr,z_bar];
            f_x_new = f_z_bar; 
        else
            disp(['HTBM step not successful. f_x = ',num2str(f_x_new),', f_z_bar = ',num2str(f_z_bar)])
        end
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    x_cell{j} = x_arr;
    bundle_cell{j} = bundle;

end

x_opt = x_cell{end}(:,end);
f_opt = f_x_new;

if(disp_flag >= 1)
    total_time = toc(start_tic);
    disp('    ...done!')
    disp(['    Final obj. value:    ',num2str(f_x_new)])
    disp(['    Total iterations:    ',num2str(sum(cellfun(@(in) size(in,2),x_cell))),' = ',num2str(sum(cellfun(@(in) size(in,2)-1,x_cell))),' + ',num2str(j_max)])
    disp(['    Total f eval.:       ',num2str(eval_counter.f_eval)])
    disp(['    Total subgrad eval.: ',num2str(eval_counter.subgrad_eval)])
    disp(['    Total time:          ',num2str(total_time)])
end

end

