%% Double-Well Potential
% cosnider the following density to sample from:
% pi(x) = exp(-U(x));
% where
% U(x) = (1/4) * ||x||^4 - (1/2) ||x||^2, x \in \mathbb{R}^d, d = 100
% Should choose $\kappa$ and $\sigma$ such that 
% 2*kappa = sigma^2
% b(x) = - grad U(x) =  -(||X||^2 - 1) * x
% Goal: Given phi(x) = x. Estimate $\pi(\phi)$. The truth is $\mu = 0_d$,
%                       a vector of zeros of length d.
%%
close all; 
clear; 
clc;
format long

rng(9) %set seed to reproduce the results
save_file = 0; %o you want to save the plots? 1: save, 0:no save
date = floor(clock); %get today's date and time of simulations
%% export_fig to save plots in high resolution
% Check if export_fig-master folder is in this current folder.
% Otherwise download it.
if ~exist('export_fig-master', 'dir')
  url = 'https://github.com/altmany/export_fig/archive/refs/heads/master.zip';
    outfilename = websave([pwd,'/export_fig-master'],url);
    unzip('export_fig-master.zip')
end
addpath([pwd,'/export_fig-master'])

%% Simulation settings
ncores = feature('numcores'); %get number of physical cores on this machine
run_parallel = 1; % If this is true, it will loop in parallel over "M" in 
                  % unbiased and single-level estimators. It will run in 
                  % parallel over "nsimul" in random-walk MCMC and tamed
                  % ULA
nsimul = 50; %number of iindependent repeats
k = 100; %burn-in period for the Markov chains
m = 2*k; % (m-k+1) is the number of samples used to approximate the single
         % level estimate of E[phi]
Lmax = 12; % Maximum level for the unbiased estimator (can set it even 
           % larger than this as the probability of picking it is very
           % small = 2^(-3/2 * Lmax)/sum_{l=Lmin}^Lmax 2^(-3/2*l)
Lmin =  5; % Starting level. Need this to be large enough so that the chains
           % are samples from $\pi_l --> \pi$
Ls = 5; % Setting $\sigma_l = 2^(-(L+Ls))$ smaller than \Delta_L
alpha = 0.9; % need this to be closer to 1 so that we give more chance for
             % for the dynamics to evolve before the chains meet.
sig = 3; % I found out that this choice of $\sigma$ is good for the simuls
         % in all models
kap = sig^2/2; % Need to choose $\kappa = \sigma^2/2$ so that you will be 
               % sampling from $\pi(x) \propto exp(-U(x))$
mcmc_maxiter = 2e3; % mcmc can run up to this number of iterations. In all 
                    % the chains meet much earlier than this.
% For printing: Only print the first 5 elements of the estimate
% $\widehat{\pi(\phi)}
outputstr = [repmat('%.5f  ', 1, 5) ' --- \n'];
d = 10^2; %pi is d-dimensional
mu = zeros(1,d); % Truth

%% 1) For comparison run "nsimul" of MCMC with Random-Walk proposal
t1 = tic;
mu0 = zeros(1,d);
mu0(1) = 10; %mu0 is the initial sample in the Markov chain
RWMH_iter = 5e6; %number of iterations of RWMH MCMC
RWMH_bp = 1e3; %burn-in period for RWMH MCMC
RWMH_proposal_sig = 6e-3;
mu_RWMHh = zeros(nsimul,d);
arh = zeros(nsimul,1);
if run_parallel
    parfor h = 1:nsimul
        [mu_RWMHh(h,:),arh(h)] = RWMH(mu0,RWMH_bp,RWMH_iter,RWMH_proposal_sig);
    end
else
    for h = 1:nsimul
        [mu_RWMHh(h,:),arh(h)] = RWMH(mu0,RWMH_bp,RWMH_iter,RWMH_proposal_sig);
    end
end
mu_RWMH = mean(mu_RWMHh);
MSE_RWMH = mean(sum((mu_RWMHh - mu).^2,2),1);
fprintf('============== RWMH ================ \n')
fprintf(['Avg Accep Rate = %.4f ,','mu_RWMH = ',outputstr, ], ...
                                mean(arh), mu_RWMH(1:5)) 
fprintf('Simul time = %.3f sec, MSE = %.5E\n', toc(t1), MSE_RWMH)
fprintf('--------------------------------------------------------------\n')
%% 2) For comparison run "nsimul" of Tamed ULA MCMC  
t2 = tic;
mu0 = zeros(1,d);
mu0(1) = 10; % mu0 is the initial sample in the Markov chain
TULA_iter = 1e6; % number of iterations of RWMH MCMC
TULA_bp = 1e3; % burn-in period for RWMH MCMC
TULA_proposal_sig = 1e-2;
mu_TULAh = zeros(nsimul,d);
if run_parallel
    parfor h = 1:nsimul
        mu_TULAh(h,:) = TULA(mu0, TULA_bp, TULA_iter, TULA_proposal_sig);
    end
else
    for h = 1:nsimul
        mu_TULAh(h,:) = TULA(mu0, TULA_bp, TULA_iter, TULA_proposal_sig);
    end
end
mu_TULA = mean(mu_TULAh);
MSE_TULA = mean(sum((mu_TULAh - mu).^2,2),1);
fprintf('============== TULA ================ \n')
fprintf(['mu_TULA = ',outputstr, ], mu_TULA(1:5)) 
fprintf('Simul time = %.3f sec, MSE = %.5E\n',toc(t2), MSE_TULA)
fprintf('--------------------------------------------------------------\n')

%% Run the time-averaged estimator for single level L = O(-log2(\epsilon))
% we need MSE = O(\epsilon^2), for given $\epsilon$
% The optimal cost for the single level is O(\epsilon^(-3)) with the choice
% M = O(\epsilon^(-2)) and L = O(-log2(\epsilon))
% In practice this might be higher depending on the gradient of U.

epsil = sqrt([0.05, 0.01, 0.001, 0.0005, 0.0001]); %\epsilon 
M_samples_s = floor(4 * epsil.^(-2)); %M = O(\epsilon^(-2))
MSE_len_s = length(epsil);
Time_s = zeros(MSE_len_s,1);
Cost_s = zeros(MSE_len_s,1);
MSE_s = zeros(MSE_len_s,1);
mu_s = zeros(MSE_len_s,d);

for j = 1 : MSE_len_s
    s_pi_phi = zeros(nsimul,d);
    time = zeros(nsimul,1);
    CostForm = zeros(nsimul,1);
    L = ceil(-1.7* log2(epsil(j)));  %L = O(-log2(\epsilon))
    M = M_samples_s(j);
    params = {d,mcmc_maxiter,M,L,sig,k,m,kap,alpha,Ls,run_parallel};

    for h = 1 : nsimul
        [s_pi_phi(h,:),time(h),CostForm(h)] = single_Expec_doubWell(params);
        if mod(h,10) == 0
            fprintf(['h = %d, s_pi_phi = ',outputstr], h, s_pi_phi(h,1:5))
        end
    end
    mu_s(j,:) = mean(s_pi_phi);
    MSE_s(j) = mean(sum((s_pi_phi - mu).^2,2));
    Time_s(j) = mean(time);
    Cost_s(j) = mean(CostForm);
    fprintf('========= Single level: L = %d ,  M = %d ===============\n',L,M)
    fprintf(['TRUTH  = ',outputstr,'APPROX = ',outputstr],mu(1:5), mu_s(j,1:5))
    fprintf('M = %d, MSE = %.5E\n', M,MSE_s(j))
    fprintf('AvgTime = %.3f, AvgCost = %.3f\n',Time_s(j),Cost_s(j))
    fprintf('------------------------------------------------------------\n')
end

%% Unbiased Estimator with levels are cut and the range is Lmin : Lmax 
% Set the following to be multiples of the number of cores in case parallel
% computation is enabled.

M_samples_ub = [1,10,100,1000,10000]*ncores * 2;

% If you have small number of cores, increase the factor from 2 to whatever
% We need to get MSE values close to the values we got from the single
% level for comparison purposes

MSE_len_ub = length(M_samples_ub);
Time_ub = zeros(MSE_len_ub,1);
Cost_ub = zeros(MSE_len_ub,1);
MSE_ub = zeros(MSE_len_ub,1);
mu_ub = zeros(MSE_len_ub,d);

for j = 1 : MSE_len_ub

    ub_pi_phi = zeros(nsimul,d);
    time = zeros(nsimul,1);
    CostForm = zeros(nsimul,1);
    M = M_samples_ub(j);  
    params = {d,mcmc_maxiter,M,Lmin,Lmax,sig,k,m,kap,alpha,Ls,run_parallel};
        
    for h = 1 : nsimul
        [ub_pi_phi(h,:),time(h),CostForm(h)]= ub_Expec_doubWell(params);
        if mod(h,10) == 0
            fprintf(['h = %d, ub_pi_phi = ',outputstr], h, ub_pi_phi(h,1:5)) 
        end
    end
    
    mu_ub(j,:) = mean(ub_pi_phi);
    MSE_ub(j) = mean(sum((ub_pi_phi - mu).^2,2));
    Time_ub(j) = mean(time);
    Cost_ub(j) = mean(CostForm);
    fprintf('========= Unbiased: M = %d ===============\n',M)
    fprintf(['TRUTH = ',outputstr,'APPROX = ',outputstr],mu(1:5), mu_ub(j,1:5))
    fprintf('nsimul = %d, M = %d, MSE = %.5E\n',nsimul, M, MSE_ub(j))
    fprintf('AvgTime = %.3f, AvgCost = %.3f\n', Time_ub(j), Cost_ub(j))
    fprintf('------------------------------------------------------------\n')
end


%%
if save_file 
    %create a folder with time and date
    subfolder = sprintf('Doub_Well_Date%d_%d_%d_%d_%d', date(1:5));
    mkdir(subfolder)
    % write the true \pi(\phi)
    writematrix(mu, [subfolder,'/true_mu.dat'])
    
    %write the results of running the single level estimator 
    writematrix(mu_s, [subfolder,'/mu_s.dat'])
    writematrix(MSE_s, [subfolder,'/MSE_s.dat'])
    writematrix(Cost_s, [subfolder,'/Cost_s.dat'])
    writematrix(Time_s, [subfolder,'/Time_s.dat'])
    
    %write the results of running the unbiased estimator 
    writematrix(mu_ub, [subfolder,'/mu_ub.dat'])
    writematrix(MSE_ub, [subfolder,'/MSE_ub.dat'])
    writematrix(Cost_ub, [subfolder,'/Cost_ub.dat'])
    writematrix(Time_ub, [subfolder,'/Time_ub.dat'])
end
%% Fitting

%fit the single level MSE VS COST
log2_MSE_s = log2(MSE_s);
log2_Cost_s = log2(Cost_s);
log2_Time_s = log2(Time_s);
Pol_Cost_s = polyfit(log2_MSE_s,log2_Cost_s, 1);
Pol_Time_s = polyfit(log2_MSE_s,log2_Time_s, 1);
Cost_fit_s = 2.^(Pol_Cost_s(1) * log2_MSE_s + Pol_Cost_s(2));
Time_fit_s = 2.^(Pol_Time_s(1) * log2_MSE_s + Pol_Time_s(2));

%fit the ub MSE VS COST
log2_MSE_ub = log2(MSE_ub);
log2_Cost_ub  = log2(Cost_ub);
log2_Time_ub = log2(Time_ub);
Pol_Cost_ub = polyfit(log2_MSE_ub,log2_Cost_ub, 1);
Pol_Time_ub = polyfit(log2_MSE_ub,log2_Time_ub, 1);
Cost_fit_ub = 2.^(Pol_Cost_ub(1) * log2_MSE_ub + Pol_Cost_ub(2));
Time_fit_ub = 2.^(Pol_Time_ub(1) * log2_MSE_ub + Pol_Time_ub(2));

%% Plotting

set(0, 'defaultLegendInterpreter','latex');
set(0, 'defaultTextInterpreter','latex');
%set the background of the figure to be white
set(0,'defaultfigurecolor',[1 1 1])

% Plot Cost VS MSE
txt1 = sprintf('Unbiased Fit $O(\\epsilon^{%.3f})$',2*Pol_Cost_ub(1));
txt2 = sprintf('Single Fit $O(\\epsilon^{%.3f})$',2*Pol_Cost_s(1));

figure('Position', [800 800 1000 600])
loglog(MSE_ub, Cost_ub,'rs','DisplayName','Unbiased','MarkerSize',30,...
    'MarkerEdgeColor','red','MarkerFaceColor',[1 .6 .6])
hold on
loglog(MSE_ub, Cost_fit_ub,'-b','DisplayName',txt1, 'LineWidth',5)
hold on
loglog(MSE_s,Cost_s,'mv','DisplayName','Single','MarkerSize',30,...
    'MarkerEdgeColor','m','MarkerFaceColor',[1 0 .6])
hold on
loglog(MSE_s,Cost_fit_s,'-k','DisplayName',txt2,'LineWidth',5)
hold off
grid on
ax = gca;
ax.GridAlpha = 0.5;
ax.FontSize = 38;
legend('Location','southwest')
set(legend, 'FontSize', 45)
legend('show');
xlabel('MSE', 'FontSize', 45)
ylabel('Cost','FontSize', 45);
hs = get(gca, 'YLabel');
pos = get(hs, 'Position');
pos(1) = pos(1)+0.00000008;
set(hs, 'Position', pos)

title('Double-Well Model: Cost VS MSE','FontSize', 45)
if save_file 
    file_name = [subfolder,'/DoubWell_mse_cost.pdf'];
    export_fig(file_name, '-q101')
end


% Plot time VS MSE
figure('Position', [200 800 1000 600])
txt1 = sprintf('Unbiased Fit $O(\\epsilon^{%.3f})$',2*Pol_Time_ub(1));
txt2 = sprintf('Single Fit $O(\\epsilon^{%.3f})$',2*Pol_Time_s(1));
loglog(MSE_ub, Time_ub,'rs','DisplayName','Unbiased','MarkerSize',30,...
    'MarkerEdgeColor','red','MarkerFaceColor',[1 .6 .6])
hold on
loglog(MSE_ub, Time_fit_ub,'-b','DisplayName',txt1, 'LineWidth',5)
hold on
loglog(MSE_s,Time_s,'mv','DisplayName','Single','MarkerSize',30,...
    'MarkerEdgeColor','m','MarkerFaceColor',[1 0 .6])
hold on
loglog(MSE_s,Time_fit_s,'-k','DisplayName',txt2,'LineWidth',5)
hold off
grid on
ax = gca;
ax.GridAlpha = 0.5;
ax.FontSize = 38;
legend('Location','southwest')
set(legend, 'FontSize', 45)
legend('show');
xlabel('MSE', 'FontSize', 45)
ylabel('Time (sec)','FontSize', 45);
hs = get(gca, 'YLabel');
pos = get(hs, 'Position');
pos(1) = pos(1)+0.00000008;
set(hs, 'Position', pos)

title('Double-Well Model: Time VS MSE','FontSize', 45)
if save_file 
    file_name = [subfolder,'/DoubWell_mse_time.pdf'];
    export_fig(file_name, '-q101')
end

%% Functions
function [ub_pi_phi,time,CostForm] = ub_Expec_doubWell(params)
    
    d = params{1}; mcmc_maxiter = params{2}; M = params{3}; 
    Lmin = params{4}; Lmax = params{5}; sig = params{6};
    k = params{7}; m = params{8}; kap = params{9}; alpha = params{10};
    Ls = params{11}; run_parallel = params{12};

    u0 = [zeros(1,d),mvnrnd(zeros(1,d),sig^2/kap * eye(d))]; % x0, v0
    u0_tilde = u0; % x0_tild, v0_tild
    uf0 = u0; % x0f, v0f
    uf0_tilde = u0; % x0f_tild, v0f_tild 
    uc0 = u0; % x0c, v0c
    uc0_tilde = u0; % x0c_tild, v0c_tild
        
    t1 = tic;
    %\phi(x,v) = x, pi(x,v) \propto exp(-U(x,v))

    P_L = 2.^(-(3/2)*(Lmin:Lmax));
    P_L = P_L/sum(P_L);

    pi_phi_k_m = zeros(M,d);
    cost  = zeros(M,1);
    
    if run_parallel
        parfor i = 1 : M
            L =  randsample(Lmax-Lmin+1,1,true,P_L) + Lmin - 1;
            if L == Lmin
                %Initialization of the 2 coupled chains at level Lmin
                [u0s,~] = sample_from_Q(u0,u0_tilde,d,Lmin,sig,kap,Ls);
                %There is a lag of 1. u0s is ahead of u0s_tilde by one lag
                % Start the chains with (u0s, u0_tilde)
                [pi_phi_k_m(i,:),cost(i)] = pi_phi_single(d,mcmc_maxiter, u0s,...
                                        u0_tilde,L,sig,kap,alpha,k,m,Ls);
            else
                %Initialization of the4 coupled chains at levels > Lmin        
                [uf0s,uc0s] = sample_from_Kfc(uf0,uc0,d,L,sig,kap,Ls);
                % There is a lag of 1: uf0s is ahead of uf0s_tilde by one lag
                % and uc0s is ahead of uc0s_tilde by one lag

                % Start the chains with (uf0s,uc0s,uf0s_tilde,uc0s_tilde)
                [pi_phi_k_m(i,:),cost(i)] = pi_phi_coupled(d,mcmc_maxiter,uf0s,...
                                            uf0_tilde,uc0s,uc0_tilde,L,sig,...
                                            kap,alpha,k,m,Ls);
            end
            pi_phi_k_m(i,:) = pi_phi_k_m(i,:)/P_L(L-Lmin + 1);
        end
    else
        for i = 1 : M

            L =  randsample(Lmax-Lmin+1,1,true,P_L) + Lmin - 1;
            if L == Lmin
                %Initialization of the 2 coupled chains at level Lmin
                [u0s,~] = sample_from_Q(u0,u0_tilde,d,Lmin,sig,kap,Ls);
                %There is a lag of 1. u0s is ahead of u0s_tilde by one lag
                % Start the chains with (u0s, u0_tilde)
                [pi_phi_k_m(i,:),cost(i)] = pi_phi_single(d,mcmc_maxiter, u0s,...
                                        u0_tilde,L,sig,kap,alpha,k,m,Ls);
            else
                %Initialization of the4 coupled chains at levels > Lmin        
                [uf0s,uc0s] = sample_from_Kfc(uf0,uc0,d,L,sig,kap,Ls);
                % There is a lag of 1: uf0s is ahead of uf0s_tilde by one lag
                % and uc0s is ahead of uc0s_tilde by one lag

                % Start the chains with (uf0s,uc0s,uf0s_tilde,uc0s_tilde)
                [pi_phi_k_m(i,:),cost(i)] = pi_phi_coupled(d,mcmc_maxiter,uf0s,...
                                            uf0_tilde,uc0s,uc0_tilde,L,sig,...
                                            kap,alpha,k,m,Ls);
            end
            pi_phi_k_m(i,:) = pi_phi_k_m(i,:)/P_L(L-Lmin + 1);
        end
    end
    
    time = toc(t1);

    ub_pi_phi = mean(pi_phi_k_m);
    CostForm = sum(cost);
end

function [sing_pi_phi,time,CostForm] = single_Expec_doubWell(params)
    
    t1 = tic;
    d = params{1}; mcmc_maxiter = params{2}; M = params{3}; 
    L = params{4}; sig = params{5};
    k = params{6}; m = params{7}; kap = params{8}; alpha = params{9};
    Ls = params{10}; run_parallel = params{11};
 
    u0    = [zeros(1,d),mvnrnd(zeros(1,d),sig^2/kap * eye(d))]; % x0, v0
    u0_tilde  = u0; %x0_tild, v0_tild

    pi_phi_k_m = zeros(M,d);
    cost  = zeros(M,1);
    if run_parallel
        parfor i = 1 : M
            %Initialization of the 2 coupled chains at level L
            [u0s,~] = sample_from_Q(u0,u0_tilde,d,L,sig,kap,Ls);
            %There is a lag of 1. u0s is ahead of u0s_tilde by one lag
            % Start the chains with (u0s, u0_tilde) and run with level L
            [pi_phi_k_m(i,:),cost(i)] = pi_phi_single(d,mcmc_maxiter, u0s,...
                                        u0_tilde,L,sig,kap,alpha,k,m,Ls);
        end
    else
        for i = 1 : M
            %Initialization of the 2 coupled chains at level L
            [u0s,~] = sample_from_Q(u0,u0_tilde,d,L,sig,kap,Ls);
            %There is a lag of 1. u0s is ahead of u0s_tilde by one lag
            % Start the chains with (u0s, u0_tilde) and run with level L
            [pi_phi_k_m(i,:),cost(i)] = pi_phi_single(d,mcmc_maxiter, u0s,...
                                        u0_tilde,L,sig,kap,alpha,k,m,Ls);
        end
    end
    time = toc(t1);
    sing_pi_phi = mean(pi_phi_k_m);
    CostForm = sum(cost);
end

function [pi_phi_k_m, cost] = pi_phi_single(d,mcmc_maxiter,u0,u0_tilde,L,...
                                    sig,kap,alpha,k,m,Ls)

    U = zeros(ceil(mcmc_maxiter/2),2*d);
    U_tilde = zeros(ceil(mcmc_maxiter/2),2*d);
    U(1,:) = u0;
    U_tilde(1,:) = u0_tilde;
    stop_time = mcmc_maxiter;
    cost = 0;
    identical = false;
    for i =  2 : mcmc_maxiter
        if rand < alpha
            [uQ,uQ_tilde] = sample_from_Q(u0,u0_tilde,d,L,sig,kap,Ls);
            U(i,:) = uQ;
            U_tilde(i,:) = uQ_tilde;
            
            u0 = uQ;
            u0_tilde = uQ_tilde;
            cost = cost + 2*2^L; %cost of sampling x and v (from SDEs)
        else
            [uP,uP_tilde,identical]=sample_from_P(u0,u0_tilde,d,L,sig,kap,Ls);
            U(i,:) = uP;
            U_tilde(i,:) = uP_tilde;
            
            u0 = uP;
            u0_tilde = uP_tilde;
            %cost of sampling x and v (from SDEs up to time 1-\Delta_l)
            cost = cost + 2*2^L-2; 
        end
      
        if identical 
            stop_time = i;
            break
        end
    end
    U = U(1:stop_time,:);
    U_tilde = U_tilde(1:stop_time,:);

    if k <= stop_time - 2
        k0 = min(m,stop_time-1);
        temp = min(1, ((k+1:stop_time-1)-k)/(k0-k+1))';
        pi_phi_k_m = 1/(k0-k+1) * sum(U(k:k0,1:d)) ...
                    + sum(temp.* (U(k+1:end-1,1:d)-U_tilde(k+1:end-1,1:d)));
    else 
        %start with k = 0.5 * stop_time;
        k0 = min(m,stop_time-1);
        k = min(ceil(stop_time/2),ceil(m/2));
        temp = min(1, ((k+1:stop_time-1)-1)/(k0-k+1))';
        pi_phi_k_m = 1/(k0-k+1) * sum(U(k:k0,1:d)) ...
                    + sum(temp.*(U(k+1:end-1,1:d)-U_tilde(k+1:end-1,1:d)));
    end

end


function [pi_phi_k0_m, cost] = pi_phi_coupled(d,mcmc_maxiter,uf0,uf0_tilde,...
                                 uc0,uc0_tilde,L,sig,kap,alpha,k,m,Ls)
    
Uf = zeros(ceil(mcmc_maxiter/2),2*d);
Uf_tilde = zeros(ceil(mcmc_maxiter/2),2*d);
Uc = zeros(ceil(mcmc_maxiter/2),2*d);
Uc_tilde = zeros(ceil(mcmc_maxiter/2),2*d);


%initialize the chains
Uf(1,:) = uf0;
Uf_tilde(1,:) = uf0_tilde;
Uc(1,:) = uc0;
Uc_tilde(1,:) = uc0_tilde;

identical_f = false;
identical_c = false;

cost = 0;

for i =  2 : mcmc_maxiter
    
    %(uf,uf_tilde,uc,uc_tilde) in D^2
    if identical_f && identical_c 

        [ufQ,ufQ_tilde,ucQ,ucQ_tilde]=sample_from_Q_fc(uf0,uf0_tilde,...
                               uc0,uc0_tilde,d,L,sig,kap,Ls);
        Uf(i,:) = ufQ;
        Uf_tilde(i,:) = ufQ_tilde;
        Uc(i,:) = ucQ;
        Uc_tilde(i,:) = ucQ_tilde;
        
        uf0 = ufQ;
        uf0_tilde = ufQ_tilde;
        uc0 = ucQ;
        uc0_tilde = ucQ_tilde;
        %cost of sampling x and v (from coupled SDEs)
        cost = cost + 2*(2^L+2^(L-1)); 
    else
        if rand < alpha
            [ufQ,ufQ_tilde,ucQ,ucQ_tilde] = sample_from_Q_fc(uf0,uf0_tilde,...
                          uc0,uc0_tilde,d,L,sig,kap,Ls);
            Uf(i,:) = ufQ;
            Uf_tilde(i,:) = ufQ_tilde;
            Uc(i,:) = ucQ;
            Uc_tilde(i,:) = ucQ_tilde;
            
            uf0 = ufQ;
            uf0_tilde = ufQ_tilde;
            uc0 = ucQ;
            uc0_tilde = ucQ_tilde;
            %cost of sampling x and v (from coupled SDEs)
            cost = cost + 2*(2^L+2^(L-1));
        else
            [ufP,ufP_tilde,ucP,ucP_tilde, identical_f, identical_c]=...
                    sample_from_P_fc(uf0,uf0_tilde,uc0,uc0_tilde,d,L,sig,...
                    kap,Ls,identical_f, identical_c);
            Uf(i,:) = ufP;
            Uf_tilde(i,:) = ufP_tilde;
            Uc(i,:) = ucP;
            Uc_tilde(i,:) = ucP_tilde;
            
            uf0 = ufP;
            uf0_tilde = ufP_tilde;
            uc0 = ucP;
            uc0_tilde = ucP_tilde; 
            %cost of sampling x and v (from coupled SDEs up to time 1-Delta_l)
            cost = cost + 2*(2^L+2^(L-1))-4;
        end
    end
    if  identical_f && identical_c
          stop_time = i;
        break
    end
end

Uf = Uf(1:stop_time,:);
Uf_tilde = Uf_tilde(1:stop_time,:);
Uc = Uc(1:stop_time,:);
Uc_tilde = Uc_tilde(1:stop_time,:);

if k <= stop_time - 2
    k0 = min(m,stop_time-1);
    temp = min(1, ((k+1:stop_time-1)-k)/(k0-k+1))';
    pi_phi_k0_m_f = 1/(k0-k+1) * sum(Uf(k:k0,1:d)) ...
                + sum(temp.*(Uf(k+1:end-1,1:d)-Uf_tilde(k+1:end-1,1:d)));
    pi_phi_k0_m_c = 1/(k0-k+1) * sum(Uc(k:k0,1:d)) ...
                + sum(temp.* (Uc(k+1:end-1,1:d)-Uc_tilde(k+1:end-1,1:d)));
else 
    %start with k = 1;
    k0 = min(m,stop_time-1);    
    k = min(ceil((stop_time-1)/2), ceil(m/2));
    temp = min(1, ((k+1:stop_time-1)-k)/(k0-k+1))';
    pi_phi_k0_m_f = 1/(k0-k+1) * sum(Uf(k:k0,1:d)) ...
                + sum(temp.* ( Uf(k+1:end-1,1:d) - Uf_tilde(k+1:end-1,1:d) ) );
    pi_phi_k0_m_c = 1/(k0-k+1) * sum(Uc(k:k0,1:d)) ...
                + sum(temp.* ( Uc(k+1:end-1,1:d) - Uc_tilde(k+1:end-1,1:d) ) );
end

pi_phi_k0_m = pi_phi_k0_m_f - pi_phi_k0_m_c;
end


function [u,u_tilde] = sample_from_Q(u0,u0_tilde,d,L,sig,kap,Ls)
    
    n = 2^L;
    h = 1/n;
    sig_L = 2^(-(L+Ls));
    
    %copy the first d columns of dB to d+1:end.
    %Similar with Gamma
    Gamma = repmat(sqrt(h) * randn(n,d),1,2);
    dB = repmat(sqrt(h) * randn(n,d), 1,2);

    X = [u0(1:d), u0_tilde(1:d)];
    V = [u0(d+1:end),u0_tilde(d+1:end)];
    X_old = X;
    V_old = V;
    
    for k = 1: n      
        b = [-gradU(X_old(1:d)),-gradU(X_old(d+1:end))];
        X = X_old + V_old * h + sig_L * Gamma(k,:);
        V = V_old + (b - kap * V_old) * h + sig * dB(k,:);
        X_old = X;
        V_old = V;
    end
    
    u = [X(1:d),V(1:d)];
    u_tilde = [X(d+1:end),V(d+1:end)];
end

function [U,U_tilde, identical] = sample_from_P(u0,u0_tilde,d,L,sig,kap,Ls)

    n = 2^L;
    h = 1/n;
    sig_L = 2^(-(L+Ls));
      
    %copy the first d columns of dB to d+1:end.
    %Similar with Gamma
    Gamma = repmat(sqrt(h) * randn(n-1,d),1,2);
    dB = repmat(sqrt(h) * randn(n-1,d), 1,2);
    
    X = [u0(1:d), u0_tilde(1:d)];
    V = [u0(d+1:end),u0_tilde(d+1:end)];
    X_old = X;
    V_old = V;
    
    for k = 1: n - 1    
        b = [-gradU(X_old(1:d)),...
             -gradU(X_old(d+1:end))];
        X = X_old + V_old * h + sig_L * Gamma(k,:);
        V = V_old + (b - kap * V_old) * h + sig * dB(k,:);
        X_old = X;
        V_old = V;
    end
        
    b = -gradU(X(1:d));
    p_mu = [X(1:d),V(1:d)] + [V(1:d),b-kap*V(1:d)] * h;
    b = -gradU(X(d+1:end));
    q_mu = [X(d+1:end),V(d+1:end)]+[V(d+1:end),b-kap*V(d+1:end)]* h;
    
    [U,U_tilde, identical] = refl_max_coupl(d, p_mu,q_mu, h, sig_L,sig);
end


function [ufQ,ufQ_tilde,ucQ,ucQ_tilde] = sample_from_Q_fc(uf0,uf0_tilde,...
                                            uc0,uc0_tilde,d, L,sig,...
                                            kap,Ls)
    nf = 2^L;
    hf = 1/nf;
    nc = 2^(L-1);
    hc = 1/nc;
    sig_Lf = 2^(-(L+Ls));
    sig_Lc = 2^(-(L-1+Ls));
    
    %copy the first d columns of dB to d+1:end.
    %Similar with Gamma
    Gammaf = repmat(sqrt(hf) * randn(nf,d),1,2);
    dBf = repmat(sqrt(hf) * randn(nf,d),1,2);
    
    Xf = [uf0(1:d),uf0_tilde(1:d)];
    Vf = [uf0(d+1:end),uf0_tilde(d+1:end)];
    Xc = [uc0(1:d),uc0_tilde(1:d)];
    Vc = [uc0(d+1:end),uc0_tilde(d+1:end)];
    
    Xf_old = Xf;
    Vf_old = Vf;
    Xc_old = Xc;
    Vc_old = Vc;
    
    for k = 1: nc  
        Gammac = zeros(1,2*d);
        dBc = zeros(1,2*d);
        for j = 1 : 2
            bf = [-gradU(Xf_old(1:d)),...
                    -gradU(Xf_old(d+1:end))];
            Xf = Xf_old + Vf_old * hf + sig_Lf * Gammaf(2*(k-1)+j,:);
            Vf = Vf_old + (bf -kap*Vf_old)*hf+sig*dBf(2*(k-1)+j,:);
            Xf_old = Xf;
            Vf_old = Vf;
            
            Gammac = Gammac + Gammaf(2*(k-1)+j,:);
            dBc = dBc + dBf(2*(k-1)+j,:);
        end
        
        bc = [-gradU(Xc_old(1:d)),...
                -gradU(Xc_old(d+1:end))];
        Xc = Xc_old + Vc_old * hc + sig_Lc * Gammac;
        Vc = Vc_old + (bc - kap * Vc_old) * hc + sig * dBc;
        Xc_old = Xc;
        Vc_old = Vc;
    end
    
    ufQ = [Xf(1:d),Vf(1:d)];
    ufQ_tilde = [Xf(d+1:end),Vf(d+1:end)];
    
    ucQ = [Xc(1:d),Vc(1:d)];
    ucQ_tilde = [Xc(d+1:end),Vc(d+1:end)];   
end

function [Uf,Uf_tilde,Uc,Uc_tilde, identical_f, identical_c] = ...
                sample_from_P_fc(uf0,uf0_tilde,uc0,uc0_tilde,d,L,sig,...
                kap,Ls,identical_f, identical_c)
	nf = 2^L;
    hf = 1/nf;
    nc = 2^(L-1);
    hc = 1/nc;
    sig_Lf = 2^(-(L+Ls));
    sig_Lc = 2^(-(L-1+Ls));
    
    if ~identical_f && ~identical_c
        %copy the first d columns of dB to d+1:end.
        %Similar with Gamma
        Gammaf = repmat(sqrt(hf) * randn(nf-1,d),1,2);
        dBf = repmat(sqrt(hf) * randn(nf-1,d),1,2);

        Xf = [uf0(1:d),uf0_tilde(1:d)];
        Vf = [uf0(d+1:end),uf0_tilde(d+1:end)];
        Xc = [uc0(1:d),uc0_tilde(1:d)];
        Vc = [uc0(d+1:end),uc0_tilde(d+1:end)];

        Xf_old = Xf;
        Vf_old = Vf;
        Xc_old = Xc;
        Vc_old = Vc;

        for k = 1: nc - 1 
            Gammac = zeros(1,2*d);
            dBc = zeros(1,2*d);
            for j = 1 : 2
                bf = [-gradU(Xf_old(1:d)),...
                        -gradU(Xf_old(d+1:end))];
                Xf = Xf_old + Vf_old * hf + sig_Lf * Gammaf(2*(k-1)+j,:);
                Vf = Vf_old + (bf -kap*Vf_old)*hf+sig*dBf(2*(k-1)+j,:);
                Xf_old = Xf;
                Vf_old = Vf;

                Gammac = Gammac + Gammaf(2*(k-1)+j,:);
                dBc = dBc + dBf(2*(k-1)+j,:);
            end

            bc = [-gradU(Xc_old(1:d)),...
                    -gradU(Xc_old(d+1:end))];
            Xc = Xc_old + Vc_old * hc + sig_Lc * Gammac;
            Vc = Vc_old + (bc - kap * Vc_old) * hc + sig * dBc;
            Xc_old = Xc;
            Vc_old = Vc;
        end

        bf = -gradU(Xf(1:d));
        muf = [Xf(1:d),Vf(1:d)] + [Vf(1:d), bf-kap*Vf(1:d)] * hf;
        bf_tilde = -gradU(Xf(d+1:end));
        muf_tilde = [Xf(d+1:end),Vf(d+1:end)]+...
                            [Vf(d+1:end),bf_tilde-kap*Vf(d+1:end)]*hf;

        bc = -gradU(Xc(1:d));
        muc = [Xc(1:d),Vc(1:d)] + [Vc(1:d), bc-kap*Vc(1:d)] * hc;
        bc_tilde = -gradU(Xc(d+1:end));
        muc_tilde = [Xc(d+1:end),Vc(d+1:end)] + ...
                            [Vc(d+1:end),bc_tilde-kap*Vc(d+1:end)]*hc;
    elseif identical_f && ~identical_c
        %copy the first d columns of dB to d+1:end.
        %Similar with Gamma
        Gammaf = repmat(sqrt(hf) * randn(nf-1,d),1,2);
        dBf = repmat(sqrt(hf) * randn(nf-1,d),1,2);

        Xf = uf0(1:d);
        Vf = uf0(d+1:end);
        Xc = [uc0(1:d),uc0_tilde(1:d)];
        Vc = [uc0(d+1:end),uc0_tilde(d+1:end)];

        Xf_old = Xf;
        Vf_old = Vf;
        Xc_old = Xc;
        Vc_old = Vc;

        for k = 1: nc - 1 
            Gammac = zeros(1,2*d);
            dBc = zeros(1,2*d);
            for j = 1 : 2
                bf = -gradU(Xf_old);
                Xf = Xf_old + Vf_old * hf + sig_Lf * Gammaf(2*(k-1)+j,1:d);
                Vf = Vf_old + (bf -kap*Vf_old)*hf+sig*dBf(2*(k-1)+j,1:d);
                Xf_old = Xf;
                Vf_old = Vf;

                Gammac = Gammac + Gammaf(2*(k-1)+j,:);
                dBc = dBc + dBf(2*(k-1)+j,:);
            end

            bc = [-gradU(Xc_old(1:d)),...
                    -gradU(Xc_old(d+1:end))];
            Xc = Xc_old + Vc_old * hc + sig_Lc * Gammac;
            Vc = Vc_old + (bc - kap * Vc_old) * hc + sig * dBc;
            Xc_old = Xc;
            Vc_old = Vc;
        end

        bf = -gradU(Xf);
        muf = [Xf,Vf] + [Vf, bf-kap*Vf] * hf;
        muf_tilde = muf;

        bc = -gradU(Xc(1:d));
        muc = [Xc(1:d),Vc(1:d)] + [Vc(1:d), bc-kap*Vc(1:d)] * hc;
        bc_tilde = -gradU(Xc(d+1:end));
        muc_tilde = [Xc(d+1:end),Vc(d+1:end)] +...
                        [Vc(d+1:end),bc_tilde-kap*Vc(d+1:end)]*hc;
    elseif ~identical_f && identical_c
        %copy the first d columns of dB to d+1:end.
        %Similar with Gamma
        Gammaf = repmat(sqrt(hf) * randn(nf-1,d),1,2);
        dBf = repmat(sqrt(hf) * randn(nf-1,d),1,2);

        Xf = [uf0(1:d),uf0_tilde(1:d)];
        Vf = [uf0(d+1:end),uf0_tilde(d+1:end)];
        Xc = uc0(1:d);
        Vc = uc0(d+1:end);

        Xf_old = Xf;
        Vf_old = Vf;
        Xc_old = Xc;
        Vc_old = Vc;

        for k = 1: nc - 1 
            Gammac = zeros(1,d);
            dBc = zeros(1,d);
            for j = 1 : 2
                bf = [-gradU(Xf_old(1:d)),...
                        -gradU(Xf_old(d+1:end))];
                Xf = Xf_old + Vf_old * hf + sig_Lf * Gammaf(2*(k-1)+j,:);
                Vf = Vf_old + (bf -kap*Vf_old)*hf+sig*dBf(2*(k-1)+j,:);
                Xf_old = Xf;
                Vf_old = Vf;

                Gammac = Gammac + Gammaf(2*(k-1)+j,1:d);
                dBc = dBc + dBf(2*(k-1)+j,1:d);
            end

            bc = -gradU(Xc_old);
            Xc = Xc_old + Vc_old * hc + sig_Lc * Gammac;
            Vc = Vc_old + (bc - kap * Vc_old) * hc + sig * dBc;
            Xc_old = Xc;
            Vc_old = Vc;
        end

        bf = -gradU(Xf(1:d));
        muf = [Xf(1:d),Vf(1:d)] + [Vf(1:d), bf-kap*Vf(1:d)] * hf;
        bf_tilde = -gradU(Xf(d+1:end));
        muf_tilde = [Xf(d+1:end),Vf(d+1:end)]+...
                            [Vf(d+1:end),bf_tilde-kap*Vf(d+1:end)]*hf;

        bc = -gradU(Xc(1:d));
        muc = [Xc,Vc] + [Vc, bc-kap*Vc] * hc;
        muc_tilde = muc;
    elseif identical_f && identical_c
        %copy the first d columns of dB to d+1:end.
        %Similar with Gamma
        Gammaf = sqrt(hf) * randn(nf-1,d);
        dBf = sqrt(hf) * randn(nf-1,d);

        Xf = uf0(1:d);
        Vf = uf0(d+1:end);
        Xc = uc0(1:d);
        Vc = uc0(d+1:end);

        Xf_old = Xf;
        Vf_old = Vf;
        Xc_old = Xc;
        Vc_old = Vc;

        for k = 1: nc - 1 
            Gammac = zeros(1,d);
            dBc = zeros(1,d);
            for j = 1 : 2
                bf = -gradU(Xf_old);
                Xf = Xf_old + Vf_old * hf + sig_Lf * Gammaf(2*(k-1)+j,:);
                Vf = Vf_old + (bf -kap*Vf_old)*hf+sig*dBf(2*(k-1)+j,:);
                Xf_old = Xf;
                Vf_old = Vf;

                Gammac = Gammac + Gammaf(2*(k-1)+j,:);
                dBc = dBc + dBf(2*(k-1)+j,:);
            end

            bc = -gradU(Xc_old(1:d));
            Xc = Xc_old + Vc_old * hc + sig_Lc * Gammac;
            Vc = Vc_old + (bc - kap * Vc_old) * hc + sig * dBc;
            Xc_old = Xc;
            Vc_old = Vc;
        end

        bf = -gradU(Xf);
        muf = [Xf,Vf] + [Vf, bf-kap*Vf] * hf;
        muf_tilde = muf;

        bc = -gradU(Xc);
        muc = [Xc,Vc] + [Vc, bc-kap*Vc] * hc;
        muc_tilde = muc;
    end

    [Uf,Uf_tilde,Uc,Uc_tilde, identical_f, identical_c] = ...
        synch_refl_max_coupl(d, muf, muf_tilde,muc, muc_tilde, sig_Lf, sig_Lc,...
        sig, hf, hc);
end

function [uf,uc] = sample_from_Kfc(uf0,uc0,d,L,sig,kap,Ls)
    nf = 2^L;
    hf = 1/nf;
    nc = 2^(L-1);
    hc = 1/nc;
    sig_Lf = 2^(-(L+Ls));
    sig_Lc = 2^(-(L-1+Ls));

    %copy the first d columns of dB to d+1:end.
    %Similar with Gamma
    Gammaf = sqrt(hf) * randn(nf,d);
    dBf = sqrt(hf) * randn(nf,d);
    
    Xf = uf0(1:d);
    Vf = uf0(d+1:end);
    Xc = uc0(1:d);
    Vc = uc0(d+1:end);
    
    Xf_old = Xf;
    Vf_old = Vf;
    Xc_old = Xc;
    Vc_old = Vc;
    
    for k = 1: nc  
        Gammac = zeros(1,d);
        dBc = zeros(1,d);
        for j = 1 : 2
            bf = -gradU(Xf_old);
            Xf = Xf_old + Vf_old * hf + sig_Lf * Gammaf(2*(k-1)+j,:);
            Vf = Vf_old + (bf -kap*Vf_old)*hf+sig*dBf(2*(k-1)+j,:);
            Xf_old = Xf;
            Vf_old = Vf;
            
            Gammac = Gammac + Gammaf(2*(k-1)+j,:);
            dBc = dBc + dBf(2*(k-1)+j,:);
        end

        bc = -gradU(Xf_old);
        Xc = Xc_old + Vc_old * hc + sig_Lc * Gammac;
        Vc = Vc_old + (bc - kap * Vc_old) * hc + sig * dBc;
        Xc_old = Xc;
        Vc_old = Vc;
    end
    
    uf = [Xf,Vf];
    uc = [Xc,Vc];
end


function [X, Y, identical] = refl_max_coupl(d, p_mu,q_mu, h, sig_L,sig)
    %The reflection maximal coupling is only for two Gaussians with the
    %same covariance
    % cov = [sig_L^2 * h * I_{dxd},  0_{dxd}                ;
    %               0_{dxd}       ,  sig^2 * h * I_{dxd}  ]
    % cov is a diagonal matrix
    % A below is the diagonal of chol(cov) = sqrt(diag(cov))
    d1 = 2*d;
    A = [sig_L * sqrt(h) * ones(1,d), sig * sqrt(h) * ones(1,d)];
    
    z = (p_mu - q_mu) ./ A; %Sig^{-1/2} (mu_1 - mu_2)
    e = z/norm(z);
    
    %draw Xdot from s = N(0,I_{2*d}) (which is spherically symmetrical)
    Xdot = randn(1,d1);
    ratio = log_stand_mvnpdf(d1, Xdot + z) - log_stand_mvnpdf(d1, Xdot);
    
    if log(rand) <= ratio
        X = p_mu + A .* Xdot;
        Y = X;  
        identical = true;
    else
        Ydot = Xdot - 2*(e*Xdot')*e;
        X = p_mu + A .* Xdot;
        Y = q_mu + A .* Ydot;
        identical = false;
    end
    
end

function [Uf,Uf_tilde, Uc, Uc_tilde, identical_f, identical_c] = ...
                synch_refl_max_coupl(d,muf, muf_tilde, muc, muc_tilde,...
                sig_Lf, sig_Lc, sig,hf, hc)

    %Synchronous pairwise reflection maximal couplings%
    
    d1 = 2*d;
    Af = [sig_Lf * sqrt(hf) * ones(1,d), sig * sqrt(hf) * ones(1,d)];
    Ac = [sig_Lc * sqrt(hc) * ones(1,d), sig * sqrt(hc) * ones(1,d)];
    
    %draw v from s = N(0,I_{2*d}) (which is spherically symmetrical)
    v = randn(1,d1);
    
    %1)
    vf = v; 
    Uf = muf + Af .* vf;
    vc = v; 
    Uc = muc + Ac .* vc;
    
    %2)
    uf = (muf - muf_tilde) ./ Af;
    ef = uf / norm(uf);
    uc = (muc - muc_tilde) ./ Ac;
    ec = uc / norm(uc);
    
    %3)
    if log(rand) < log_stand_mvnpdf(d1, vf+uf) - log_stand_mvnpdf(d1, vf)
        Uf_tilde = Uf;
        identical_f = true;
    else
        vfdot = vf - 2*(ef * vf')*ef;
        Uf_tilde = muf_tilde + Af .* vfdot;
        identical_f = false;
    end
    
    if log(rand) < log_stand_mvnpdf(d1, vc+uc) - log_stand_mvnpdf(d1, vc)
        Uc_tilde = Uc;
        identical_c = true;
    else
        vcdot = vc - 2*(ec * vc')*ec;
        Uc_tilde = muc_tilde + Ac .* vcdot;
        identical_c = false;
    end
           
end

function [X,Y,identical] = max_coupl(d, p_mu,q_mu, h, sig_L,sig)
    % Note that this very specific to the case
    % where p & q are 2d - dimensional Gaussian with the same covariance 
    % cov = [sig_L^2 * h * I_{dxd},  0_{dxd}                ;
    %               0_{dxd}       ,  sig^2 * h * I_{dxd}  ]
    % A below is the diagonal of chol(cov) = sqrt(diag(cov))
    d1 = 2*d;
    A = [sig_L * sqrt(h) * ones(1,d), sig * sqrt(h) * ones(1,d)];
    X = randn(1,d1) .* A + p_mu; % mvnrnd(p_mu, cov); 
    W = log(rand) + log_mvnpdf1(d1, X, p_mu, A);
    %W = rand * mvnpdf(X,p_mu,p_sig);
    %if W < mvnpdf(X,q_mu,q_sig)
    if W < log_mvnpdf1(d1,X,q_mu,A)
        Y = X;
        identical = true;
        return 
    else
        while 1
            %Y = mvnrnd(q_mu,q_sig);
            Y = randn(1,d1) .* A + q_mu;
            %W = rand * mvnpdf(Y,q_mu,q_sig);
            W = log(rand) + log_mvnpdf1(d1,Y,q_mu,A);
            %if W > mvnpdf(Y,p_mu,p_sig)
            if W > log_mvnpdf1(d1,Y,p_mu,A)
                identical = false;
                return
            end
        end
    end
end

function [Uf1,Uf2,Uc1,Uc2,identical_f, identical_c] = ...
           fourWay_max_coupl(d, muf1, muf2, muc1, muc2, sig_Lf, sig_Lc,...
             sig, hf, hc,identical_f, identical_c)
      
    % Note that this very specific to the case
    % where pf & qf are 2d - dimensional Gaussians with the same covariance 
    % covf = [sig_Lf^2 * hf * I_{dxd},  0_{dxd}                ;
    %               0_{dxd}       ,  sig^2 * hf * I_{dxd}  ]
    % Af below is the diagonal of chol(cov) = sqrt(diag(covf))
    % Similar with pc & qc

    
    d1 = 2*d;
    Af = [sig_Lf * sqrt(hf) * ones(1,d), sig * sqrt(hf) * ones(1,d)];
    Ac = [sig_Lc * sqrt(hc) * ones(1,d), sig * sqrt(hc) * ones(1,d)];
  
    Uc1 = randn(1,d1) .* Ac + muc1; %mvnrnd(pc_mu, pc_sig);
    
    la1 = log_mvnpdf1(d1,Uc1, muc1, Ac);
    la2 = log_mvnpdf1(d1,Uc1, muc2, Ac);
    la3 = log_mvnpdf1(d1,Uc1, muf1, Af); 
    la4 = log_mvnpdf1(d1,Uc1, muf2, Af); 
    
    la = min([la1,la2,la3,la4],[],'all') - la1;
    fprintf('outside if la = %.3e\n',la)
    if log(rand) <= la
        Uf1 = Uc1;
        Uf2 = Uc1;
        Uc2 = Uc1;
        identical_f = true;
        identical_c = true;
    else
        if identical_c
            % if coarse chains have met: sample common proposal\
            reject = true;
            while reject
                Uc1 = randn(1,d1) .* Ac + muc1;
                la1 = log_mvnpdf1(d1,Uc1, muc1, Ac);
                la2 = log_mvnpdf1(d1,Uc1, muc2, Ac);
                la3 = log_mvnpdf1(d1,Uc1, muf1, Af);
                la4 = log_mvnpdf1(d1,Uc1, muf2, Af);
    
                la = min([la1,la2,la3,la4],[],'all') - la1;
                fprintf('inside if identc inside while la = %.3e\n',la)
                reject = (log(rand) < la);
            end
            
            Uc2 = Uc1;
            identical_c = true;
        else
            % if coarse chains have not met: sample proposals independently
            identical_c = false;
            reject = true;
            while reject
                % sample first proposal of coarse chain
                Uc1 = randn(1,d1) .* Ac + muc1; 
                la1 = log_mvnpdf1(d1,Uc1, muc1, Ac); 
                la2 = log_mvnpdf1(d1,Uc1, muc2, Ac); 
                la3 = log_mvnpdf1(d1,Uc1, muf1, Af); 
                la4 = log_mvnpdf1(d1,Uc1, muf2, Af); 
                        
                la = min([la1,la2,la3,la4],[],'all') - la1;
                fprintf('inside if not identc inside while1 la = %.3e\n',la)
                reject = (log(rand) < la);
            end
            
            % sample proposal for second coarse chain
            reject = true;
            while reject
                % sample first proposal of coarse chain
                Uc2 = randn(1,d1) .* Ac + muc2;
                la1 = log_mvnpdf1(d1,Uc2, muc1, Ac); 
                la2 = log_mvnpdf1(d1,Uc2, muc2, Ac); 
                la3 = log_mvnpdf1(d1,Uc2, muf1, Af);
                la4 = log_mvnpdf1(d1,Uc2, muf2, Af);
    
                la = min([la1,la2,la3,la4],[],'all') - la2;
                fprintf('inside if not identc inside while2 la = %.3e\n',la)
                reject = (log(rand) < la);
            end
        end
        
        if identical_f
            % if fine chains have met: sample common proposal
            reject = true;
            while reject
                Uf1 = randn(1,d1) .* Af + muf1; 
                la1 = log_mvnpdf1(d1,Uf1, muc1, Ac);
                la2 = log_mvnpdf1(d1,Uf1, muc2, Ac); 
                la3 = log_mvnpdf1(d1,Uf1, muf1, Af); 
                la4 = log_mvnpdf1(d1,Uf1, muf2, Af);
    
                la = min([la1,la2,la3,la4],[],'all') - la3;
                fprintf('inside if identf inside while la = %.3e\n',la)
                reject = (log(rand) < la);
            end
            
            Uf2 = Uf1;
            identical_f = true;
        else
            % if fine chains have not met: sample proposals independently
            identical_f = false;
            
            reject = true;
            while reject
                % sample first proposal of fine chain
                Uf1 = randn(1,d1) .* Af + muf1;
                la1 = log_mvnpdf1(d1,Uf1, muc1, Ac);
                la2 = log_mvnpdf1(d1,Uf1, muc2, Ac);
                la3 = log_mvnpdf1(d1,Uf1, muf1, Af);
                la4 = log_mvnpdf1(d1,Uf1, muf2, Af); 
    
                la = min([la1,la2,la3,la4],[],'all') - la3;
                fprintf('inside if not identf inside while1 la = %.3e\n',la)
                reject = (log(rand) < la);
            end
            
            reject = true;
            while reject
                % sample second proposal of fine chain
                Uf2 = randn(1,d1) .* Af + muf2;
                la1 = log_mvnpdf1(d1,Uf2, muc1, Ac);
                la2 = log_mvnpdf1(d1,Uf2, muc2, Ac);
                la3 = log_mvnpdf1(d1,Uf2, muf1, Af);
                la4 = log_mvnpdf1(d1,Uf2, muf2, Af);
    
                la = min([la1,la2,la3,la4],[],'all') - la4;
                fprintf('inside if not identf inside while2 la = %.3e\n',la)
                reject = (log(rand) < la);
            end
        end
    end
end

function ly = log_mvnpdf1(d,X,mu,sig_sqrt) 
% evaluate the Multivariate Normal Distribution at X with mean mu and a
% a cov that is a diagonal matrix. 
% Inputs: d    - dimension
%         X    - 1xd to evaluate the PDF at
%         mu   - 1xd vector (mean)
%         sigd - 1xd vector (sqrt of diag of Sigma)
    X0 = X - mu;
    xSigSqrtinv = X0./sig_sqrt;
    logSqrtDetSigma = sum(log(sig_sqrt));
    quadform = sum(xSigSqrtinv.^2, 2);
    ly = -0.5*quadform - logSqrtDetSigma - d*log(2*pi)/2; 
end

function ly = log_stand_mvnpdf(d,X) 
% evaluate the log Multivariate Standard Normal Distribution at X 
    ly = -0.5*sum(X.^2, 2) - d*log(2*pi)/2; 
end

function U = compute_U(x)
    temp1 = norm(x)^2;
    U = (1/4) * temp1^2 - (1/2) * temp1;
end

function g = gradU(x)
    g = (norm(x)^2 -1) * x;
end

function [m,ar] = RWMH(x0, burn_in, N, step)
    d = length(x0);
    x = x0;
    sqrts = sqrt(step);
    m = zeros(size(x0)); %mean of samples 
    
    lb = -compute_U(x);
    count = 0;

    for i = 2 : burn_in + N
        xp = x + sqrts * randn(1,d);
        la = -compute_U(xp);
        if log(rand) <= (la-lb)
            x = xp;
            count = count + 1; %accept
            lb = la;
        end
        
        if i > burn_in
            m = m + x;
        end

    end
    m = m / (N+burn_in);
    ar = count/(N+burn_in);
end

function m = TULA(x0, burn_in, N, step)

    d = length(x0);
    x = x0;
    sqrtstep = sqrt(2*step);
    m = zeros(size(x0));
    
    for i = 2 : burn_in + N
        b = gradU(x);
        %bTamed = b / (1 + step*norm(b));
        bTamed = b ./ (1 + step*abs(b));
        x = x - step * bTamed + sqrtstep * randn(1,d); 
        if i > burn_in
            m = m + x;
        end
    end
    m = m /(burn_in + N);
end