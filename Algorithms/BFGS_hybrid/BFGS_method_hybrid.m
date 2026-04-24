function [x_arr,bundle_cell,htbm_success_flags] = BFGS_method_hybrid(problem_data,algo_options,N_inner_iters,N_outer_iters,htbm_options)

    n = problem_data.n;
    x_arr = zeros(n,N_outer_iters);
    bundle_cell = cell(1,N_outer_iters);
    htbm_success_flags = false(1,N_outer_iters+1);

    algo_options.N_iter = N_inner_iters;

    for j = 1:N_outer_iters
        disp(['Outer iteration j = ',num2str(j)])
        if(j == 1)
            problem_data.x0 = problem_data.x0;
        else
            problem_data.x0 = x_arr(:,j-1);
        end
        
        output = BFGS_method(problem_data,algo_options);

        x_arr(:,j) = output.x_arr(:,end);

        bundle.sample_pts = output.x_arr;
        bundle.f = output.f_arr;
        bundle.grads = output.grad_arr;

        bundle_cell{j} = bundle;

        % Apply HTBM step %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if(htbm_options.htbm_flag)
            f = problem_data.f;
            hess_f = problem_data.hess_f;

            num_sample_pts = size(bundle.sample_pts,2);
            bundle.hess = cell(1,num_sample_pts);
            for i = 1:num_sample_pts
                bundle.hess{i} = hess_f(bundle.sample_pts(:,i));
            end

            [z_bar,theta,mu] = solve_subproblem_IPOPT(bundle.sample_pts,bundle.f,bundle.grads,bundle.hess,bundle.sample_pts(:,1),htbm_options.eps,htbm_options.sp_solver_options);

            f_z_bar = f(z_bar);
            if(f_z_bar < bundle.f(end))
                disp(['HTBM step successful! f_x = ',num2str(bundle.f(end)),', f_z_bar = ',num2str(f_z_bar)])
                htbm_success_flags(j) = true;
                x_arr(:,j) = z_bar;
            else
                disp(['HTBM step not successful. f_x = ',num2str(bundle.f(end)),', f_z_bar = ',num2str(f_z_bar)])
            end
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    end

end

