function [out]=optU(pr)
% Function optU finds a local minimizer of an unconstrained function on R^n
% using user-selected a gradient based method.
% Author: Tom Asaki
% Version: December 17, 2023
%
% Call:
%
%   [out]=optU(pr)
%
% Inputs:
%
%   pr      a structure variable containing all necessary problem 
%           information. The various fields and [default values] are as 
%           follows. 
%       .progress   A positive intgeger.  Progress will be displayed
%                   every (pr.progress) iterations.
%       .obj        function handle to the objective/gradient computation
%       .x0         vector of intial decision variable values.
%                   (or) n by p array of p initial points for NM or GA.
%       .par        variable to pass to objective function (for example
%                   containing parameters in a structure variable).
%       .method     string indicating optimization method.  Options:
%                       'GD'  (GradientDescent)
%                       'CG'  (ConjugateGradient)
%                       'BFGS' (quasi-Newton)
%                       'LBFGS' (Limited-Memory BFGS)
%                       'TR' (SR1 Trust Region and Steihaug-Toint steps)
%       .maxiter [inf]  maximum number of decision variable updates 
%       .ngtol [1E-8]  stop tolerance on gradient norm
%       .dftol [1E-8]  stop tolerance on change in objective
%       .dxtol [1E-8]  stop tolerance on change in decision varable norm
%       .LS.method  string indicating the type of linesearch to perform
%                       'Armijo'       (appropriate for GD)
%                       'StrongWolfe'  (appropriate for CG and BFGS)
%       .LS.lambda [1]  line search initial step size multiplier
%       .LS.lambdamax [100] maximum line search step length
%       .LS.c1 [0.0001]  Armijo sufficient decrease parameter ( 0 < c1 < 1/2 )
%       .LS.c2 [0.9] Curvature condition parameter ( 0 < c1 < c2 < 1 )
%                 If using Conjugate Gradient method ( 0 < c1 < c2 < 1/2 )
%                 with default value [0.4]
%       .LBFGS.m [7] number of L-BFGS iterations to save
%       .CG.reset [0.1] orthogonality reset value for Conjugate Gradient
%       .TR.delta [1] initial trust region size
%       .TR.deltamax [100] maximum trust region size
%       .TR.deltatol [1E-8] stop tolerance on trust region size.
%       .TR.eta [0.01 0.25 0.75]  trust region parameters 
%                  [ sufficient decrease , shrink , expand ]
%                  ( 0 <= eta1 < eta2 < eta3 < 1 )
%       .TR.maxcond [1000]  maximum condition number on approximate model
%                   hessian for trust region method.
%   
% Outputs:
%
%   out     a structure variable containing all intial input values
%           and additional results of the optimization procedure.
%       .pr     copy of the input structure variable with added
%               default values and possibly some other algorithmic
%               necessary changes.
%       .x      (n by iter) array whose columns are the decision
%               variable iterates at each iteration
%       .f      (1 by iter) array whose entries are the corresponding
%               objective function values
%       .g      (n by iter) array whose columns are the gradient
%               vectors at each iteration.
%       .feval  total number of function evaluations
%       .geval  total number of gradient evaluations 
%       .teval  total evaluation clock time
%       .msg    output message - reason for algorithm termination
%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Intialization

out=struct();
pr=setdefaults(pr);

% The output structure variable is initialized here
out.pr=pr;

% start execution timer
beg=datetime('now');

% Make the initial call to the objective function
[out.f,out.g]=ev(pr.x0,pr,2);
out.x=pr.x0;
out.feval=1;
out.geval=1;

% Set iteration information
out.msg='';         % terminate when the output message is not empty
iter=1;             % counter
n=length(pr.x0);    % dimension of decision variable space

% Initialize terminal output
if pr.progress
    fprintf('\n');
    fprintf('    date      time    iter       f          |g|        |ap|        |df|   \n');
    fprintf('---------------------------------------------------------------------------\n');
    fprintf([char(datetime),' %5d  %+7.4e  %6.4e  \n'],iter-1,out.f(end),norm(out.g(:,end)))
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Main Routines
    
% This section is the main iterative solver.  In each iteration, 
% a descent direction is chosen (for linesearch) or the model is 
% formed (for Trust Region).  Then the step is computed as a 
% subroutine.  Then updates and termination checks are performed.

while isempty(out.msg)

    switch pr.method

        case 'GD' %%%%% GRADIENT DESCENT %%%%%

            % use negative gradient direction
            p=-out.g(:,iter);

            % call the linesearch with direction p
            ox=out.x(:,iter);
            of=out.f(iter);
            og=out.g(:,iter);
            [xnew,flag,nf,ng]=linesearch(ox,of,og,p,pr);
            out.feval=out.feval+nf;
            out.geval=out.geval+ng;

        case 'CG' %%%%% CONJUGATE GRADIENT %%%%%

            % Compute the conjugate gradient direction.  If the iteration 
            % is the first, or if descent is stalling, then use the
            % negative gradient direction (reset)
            g1=out.g(:,iter);
            if iter==1
                p=-g1;
            else
                g0=out.g(:,iter-1);
                rfactor=abs(g1'*g0)/(g1'*g1);
                if rfactor>=pr.CG.reset
                    p=-g1;
                else
                    beta=(g1'*(g1-g0))/(g0'*g0);
                    beta=max(beta,0);
                    p=-g1+beta*p;
                end
            end

            % call the linesearch in direction p
            ox=out.x(:,iter);
            of=out.f(iter);
            og=out.g(:,iter);
            [xnew,flag,nf,ng]=linesearch(ox,of,og,p,pr);
            out.feval=out.feval+nf;
            out.geval=out.geval+ng;

        case 'BFGS'  %%%%% QUASI-NEWTON (GFGS) %%%%%

            % Compute the BFGS direction.  If iter=1 then use the negative
            % gradient direction.  Also use the negative gradient direction
            % if the update will yield a nearly non-positive definite H.
            if iter==1
                H=max(abs(out.f(1)),sqrt(pr.dftol))*speye(n);
                p=-H*out.g;
            else
                s=out.x(:,end)-out.x(:,end-1);
                y=out.g(:,end)-out.g(:,end-1);
                if y'*s>0.0001*norm(s)*norm(y)
                    r=1/(y'*s);
                    I=eye(n);
                    H=(I-(r*s)*y')*H*(I-(r*y)*s')+(r*s)*s';
                else
                    H=max(abs(out.f(end)),sqrt(pr.dftol))*speye(n);
                end
                p=-H*out.g(:,end);
            end

            % call the linesearch in direction p, set alpha=1
            pr.LS.alpha=1;
            ox=out.x(:,iter);
            of=out.f(iter);
            og=out.g(:,iter);
            [xnew,flag,nf,ng]=linesearch(ox,of,og,p,pr);
            out.feval=out.feval+nf;
            out.geval=out.geval+ng;

        case 'LBFGS' %%%%% LIMITED MEMORY BFGS %%%%%

            % compute the L-BFGS newton step.
            % if the first iteration, this is scaled gradient descent
            if iter==1
                p=-max(abs(out.f(1)),sqrt(pr.dftol))*out.g;
                s=[];
                y=[];
            % If not the first iteration, update H using the previous
            % m steps
            else
                s(:,end+1)=out.x(:,end)-out.x(:,end-1);   %#ok
                y(:,end+1)=out.g(:,end)-out.g(:,end-1);   %#ok
                if iter>pr.LBFGS.m+1
                    s(:,1)=[];
                    y(:,1)=[];
                end
                p=-out.g(:,end);
                for i=min(pr.LBFGS.m,iter-1):-1:1
                    alph(i)=(s(:,i)'*p)/(s(:,i)'*y(:,i));
                    p=p-alph(i)*y(:,i);
                end
                p=(s(:,end)'*y(:,end))/(y(:,end)'*y(:,end))*p;
                for i=1:min(pr.LBFGS.m,iter-1)
                    beta=(y(:,i)'*p)/(s(:,i)'*y(:,i));
                    p=p+(alph(i)-beta)*s(:,i);
                end
            end

            % Call the linesearch in direction p,  set alpha=1
            pr.LS.alpha=1;
            ox=out.x(:,iter);
            of=out.f(iter);
            og=out.g(:,iter);
            [xnew,flag,nf,ng]=linesearch(ox,of,og,p,pr);
            out.feval=out.feval+nf;
            out.geval=out.geval+ng;

        case 'TR'  %%%%% SR1 TRUST REGION %%%%%

            % Update model hessian using SR1
            if iter==1
                B=abs(out.f)*eye(n);
            else
                s=out.x(:,end)-out.x(:,end-1);
                y=out.g(:,end)-out.g(:,end-1);
                w=y-B*s;
                B=B+(w*w')/(w'*s);
            end
            
            % Call the Trust Region step algorithm (Steihaug-Toint)
            ox=out.x(:,end);
            of=out.f(end);
            og=out.g(:,end);
            [xnew,flag,nf,delta]=TrustRegionStep(ox,of,og,B,pr);
            out.feval=out.feval+nf;
    
            % Save the trust region size for the next iteration
            pr.TR.delta=delta;

    end   % end of method switch 
              
    % update the iteration counter
    iter=iter+1;

    % update x,f,g in the output structure
    out.x(:,iter)=xnew;
    [out.f(iter),out.g(:,iter)]=ev(xnew,pr,2);
    out.geval=out.geval+1;

    % check termination criteria and set output message
    if iter>pr.maxiter
        out.msg='Maximum number of iterations reached.';
    end
    if norm(out.g(:,iter))<pr.ngtol
        out.msg='Minimum gradient norm reached.';
    end
    if (iter>1 && norm(diff(out.x(:,iter-1:iter),[],2))<pr.dxtol)
        out.msg='Minimum step size reached.';
    end
    if (iter>1 && abs(diff(out.f(iter-1:iter)))<pr.dftol)
        out.msg='Minimum change in objective reached.';
    end
    if flag % step determination "failed" for some reason
        switch pr.method
            case 'TrustRegion'
                out.msg='Minimum trust region size reached.';
            otherwise
                out.msg='Linesearch failed to find an acceptable iterate.';
        end
    end

    % Print iteration status/result to terminal
    if pr.progress 
        if ~mod(iter-1,pr.progress) || ~isempty(out.msg)
            ff=out.f(end);
            gg=norm(out.g(:,end));
            pp=norm(out.x(:,end-1)-out.x(:,end));
            df=out.f(end-1)-out.f(end);
            fprintf([char(datetime), ...
                ' %5d  %+7.4e  %6.4e  %6.4e  %6.4e  \n'], ...
                iter-1,ff,gg,pp,df)
        end
    end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Wrap Up

% save execution run time
fin=datetime('now');
out.teval=seconds(duration(fin-beg));

% closing terminal messages (key results)
if pr.progress
    fprintf('\n');
    fprintf('Objective Value : %+8.5e\n',out.f(end));
    fprintf('      Algorithm : %s\n',pr.method);
    fprintf('        Message : %s\n',out.msg);
    fprintf(' Execution Time : %g seconds\n',out.teval);
    fprintf(' Function Evals : %d\n',out.feval);
    fprintf(' Gradient Evals : %d\n',out.geval);
    fprintf('Effective Evals : %d\n',out.feval+n*out.geval);
    fprintf('\n');
end

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [pr]=setdefaults(pr)

df.par              =   [];
df.method           =   'BFGS';
df.linesearch       =   'StrongWolfe';
df.maxiter          =   inf;
df.ngtol            =   1E-8;
df.dftol            =   1E-8;
df.dxtol            =   1E-8;
df.progress         =   1;
fn=fieldnames(df);
for k=1:length(fn)
    if ~isfield(pr,fn{k}) || isempty(pr.(fn{k}))
        pr.(fn{k})=df.(fn{k});
    end
end

df.LS.lambda        =   1;
df.LS.lambdamax     =   100;
df.LS.c1            =   0.0001;
df.LS.c2            =   0.9;
fn=fieldnames(df.LS);
if ~isfield(pr,'LS')
    gn=fn';gn{2,1}=[];
    pr.LS=struct(gn{:});
end
for k=1:length(fn)
    if ~isfield(pr.LS,fn{k}) || isempty(pr.LS.(fn{k})) 
        pr.LS.(fn{k})=df.LS.(fn{k});
    end
end

df.LBFGS.m          =   7;
fn=fieldnames(df.LBFGS);
if ~isfield(pr,'LBFGS')
    gn=fn';gn{2,1}=[];
    pr.LBFGS=struct(gn{:});
end
for k=1:length(fn)
    if ~isfield(pr.LBFGS,fn{k}) || isempty(pr.LBFGS.(fn{k}))
        pr.LBFGS.(fn{k})=df.LBFGS.(fn{k});
    end
end

df.CG.reset         =   0.2;
fn=fieldnames(df.CG);
if ~isfield(pr,'CG')
    gn=fn';gn{2,1}=[];
    pr.CG=struct(gn{:});
end
for k=1:length(fn)
    if ~isfield(pr.CG,fn{k}) || isempty(pr.CG.(fn{k}))
        pr.CG.(fn{k})=df.CG.(fn{k});
    end
end

df.TR.delta         =   1;
df.TR.deltamax      =   100;
df.TR.deltatol      =   1E-8;
df.TR.eta           =   [0.01 0.25 0.75];
df.TR.maxcond       =   1000;
fn=fieldnames(df.TR);
if ~isfield(pr,'TR')
    gn=fn';gn{2,1}=[];
    pr.TR=struct(gn{:});
end
for k=1:length(fn)
    if ~isfield(pr.TR,fn{k}) || isempty(pr.TR.(fn{k}))
        pr.TR.(fn{k})=df.TR.(fn{k});
    end
end

if strcmp(pr.method,'CG') && pr.LS.c2>=0.5
    pr.LS.c2=0.45;
end

if pr.LS.c1>=pr.LS.c2
    pr.LS.c1=0.0001; 
end

pr.LBFGS.m=min(pr.LBFGS.m,length(pr.x0));

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [f,g]=ev(x,pr,m)
% this function governs all calls to an objective function.
%   x   is the point to evaluate
%   pr  is the problem structure variable
%   m   is an evaluation indicator.  If m=1 then compute f only
%           otherwise compute both f and g (gradient).
if m>1
    if isempty(pr.par)
        [f,g]=pr.obj(x);
    else
        [f,g]=pr.obj(x,pr.par);
    end
else
    if isempty(pr.par)
        [f]=pr.obj(x);
    else
        [f]=pr.obj(x,pr.par);
    end
end
return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [xnew,flag,nf,ng]=linesearch(x,f,g,p,pr)

% when a linesearch is started, we are considering a current iterate x
% and we already have the objective f and gradient g at x.  We also have
% a descent direction vector p.  The other variable pr is the entire
% problem structure variable that contains linesearch hyperparameters.

nf=0;
ng=0;

switch pr.linesearch

    case 'Armijo'
        goflag=true;
        flag=false;
        L=pr.LS.lambda;
        c1=pr.LS.c1;
        dx=pr.dxtol;
        d0=p'*g;
        while goflag
            xnew=x+L*p;
            fnew=ev(xnew,pr,1);
            nf=nf+1;
            if fnew>f+c1*L*d0
                L=L/2;
                if norm(L*p)<dx,goflag=false;flag=true;end
            else
                goflag=false;
            end
        end

    case 'StrongWolfe'
        goflag=true;
        flag=false;
        k=1;
        L(k)=pr.LS.lambda;
        c1=pr.LS.c1;
        c2=pr.LS.c2;
        d0=p'*g;
        while goflag
            F(k)=ev(x+L(k)*p,pr,1);                        %#ok
            nf=nf+1;
            if (F(k)>f+c1*L(k)*d0) || (k>1 && F(k)>=F(k-1)) 
                if k==1
                    [lambdastar,mf,mg]=zoom(0,L(k),x,f,p,d0,f,pr);
                else
                    [lambdastar,mf,mg]=zoom(L(k-1),L(k),x,f,p,d0,F(k-1),pr);
                end
                nf=nf+mf;
                ng=ng+mg;
                goflag=false;
            end
            if goflag
                [dummy,g]=ev(x+L(k)*p,pr,2);                        %#ok
                nf=nf+1;
                ng=ng+1;
                dk=p'*g;
                if abs(dk)<=-c2*d0
                    lambdastar=L(k);
                    goflag=false;
                end
            end
            if goflag && dk>=0
                if k==1
                    [lambdastar,mf,mg]=zoom(L(k),0,x,f,p,d0,F(k),pr);
                else
                    [lambdastar,mf,mg]=zoom(L(k),L(k-1),x,f,p,d0,F(k),pr);
                end
                nf=nf+mf;
                ng=ng+mg;
                goflag=false;
            end
            if goflag
                k=k+1;
                L(k)=2*L(k-1);
                if L(k)>pr.LS.lambdamax
                    flag=true;
                    goflag=false;
                    lambdastar=0;
                end
            end
        end
        xnew=x+lambdastar*p;

    otherwise
               
end    

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [lambdastar,nf,ng]=zoom(L,H,x,f,p,d0,fL,pr)
nf=0;
ng=0;
goflag=true;
while goflag
    M=(L+H)/2;
    fM=ev(x+M*p,pr,1);
    nf=nf+1;
    if fM>f+pr.LS.c1*M*d0 || fM>=fL
        H=M;
    else
        [~,gM]=ev(x+M*p,pr,2);
        nf=nf+1;
        ng=ng+1;
        dk=p'*gM;
        if abs(dk)<=-pr.LS.c2*d0
            lambdastar=M;
            goflag=false;
        end
        if goflag
            if dk*(H-L)>=0
                H=L;
            end
            L=M;
        end
        
    end
    if M*norm(p)<2*pr.dxtol
        lambdastar=M;
        goflag=false;
    end
end
return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [xnew,flag,nf,delta]=TrustRegionStep(x,f,g,B,pr)

% when calling for a TR step we have a current iterate x and we have
% already evaluated the objective f and gradient g at x.  We also have
% a model defined by a local approximate hessian B.  The only other
% input is the entire problem structure variable pr containing 
% necessary TR hyperparameters.

flag=false;
delta=pr.TR.delta;
rho=-1;
nf=0;
g0=g;
n=length(x);

% loop until an acceptable point is found
while rho<=pr.TR.eta(1)

    % Compute the Steihaug-Toint step.
    z=zeros(size(g));
    r=g;
    d=-g;
    ng=norm(g);
    e=min(0.5,sqrt(ng))*ng;
    StopCondition=false;
    inneriter=0;
    while StopCondition==false 
        t=d'*(B*d);
        if t<=0
            tau=posroot(z,d,delta);
            p=z+tau*d;
            StopCondition=true;
        else
            alpha=(r'*r)/t;
            z=z+alpha*d;
            if norm(z)>=delta
                tau=posroot(z,d,delta);
                p=z+tau*d;
                StopCondition=true;
            else
                rold=r;
                r=r+alpha*(B*d);
                if norm(r)<e
                    p=z;
                    StopCondition=true;
                end
                beta=(r'*r)/(rold'*rold);
                d=-r+beta*d;
            end
        end
        inneriter=inneriter+1;
        if inneriter==n 
            p=z;
            StopCondition=true;
        end
    end

    % evaluate rho
    [fnew]=ev(x+p,pr,1);
    nf=nf+1;
    modelchange=-g0'*p-(p'*(B*p))/2;
    rho=(f-fnew)/modelchange;

    % updates: shrink or grow delta, keep an improved point
    if rho<pr.TR.eta(2)
        delta=delta/4;
    elseif rho>pr.TR.eta(3) && norm(p)>0.999*delta
        delta=min(2*delta,pr.TR.deltamax);
    end
    if rho>pr.TR.eta(1)
        xnew=x+p;
    end

    % if trust region gets too small, then stop
    if delta<pr.TR.deltatol
        flag=true;
        xnew=x+p;
        rho=inf;
    end

end

% if the last iteration did not find an improved point because
% it stopped on deltatol, then do not update x.
if fnew>=f
    xnew=x;
end

return
   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function tau=posroot(z,d,delta)
a=z'*d;
b=d'*d;
tau=-(a/b)+sqrt((a/b)^2+(delta^2-z'*z)/b);
return

