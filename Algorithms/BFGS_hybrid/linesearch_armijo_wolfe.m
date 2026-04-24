% The inexact Wolfe line search from [Lewis, Overton (2013)], Alg. 4.6,
% without differentiability check. A stopping criterion is added for
% numerical reasons, which stops when the step length becomes too small (in
% accordance with BFGS_method.m).

function [t,f_xtp,grad_f_xtp] = linesearch_armijo_wolfe(x,p,f_x,grad_f_x,problem_data,algo_options)

f = problem_data.f;
grad_f = problem_data.grad_f;

c1 = algo_options.c1;
c2 = algo_options.c2;
step_threshold = algo_options.step_threshold;

s = grad_f_x'*p;

A = @(t,f_xtp) f_xtp - f_x < c1*s*t;
W = @(grad_f_xtp) grad_f_xtp'*p > c2*s;

alpha = 0;
beta = Inf;
if(isa(x,"sym"))
    t = vpa(1);
else
    t = 1;
end

while(1)

    f_xtp = f(x + t*p);
    if(~A(t,f_xtp))
        beta = t;
    else
        grad_f_xtp = grad_f(x + t*p);
        if(~W(grad_f_xtp))
            alpha = t;
        else
            break
        end
    end

    if(~isinf(beta))
        t = (alpha+beta)/2;
    else
        t = 2*alpha;
    end

    if(t <= step_threshold || beta - alpha <= step_threshold)
        t = 0;
        f_xtp = f_x;
        grad_f_xtp = grad_f_x;
        
        break
    end
end

end

