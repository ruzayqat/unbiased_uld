
close all; 
clear; 
clc;
format long
rng(6)
save_file = 1;
date = floor(clock);
%% Check if export_fig-master folder is in this current folder,
% otherwise download it
% export_fig function is used to generate high quality plots in pdf or
% other formats
if ~exist('export_fig-master', 'dir')
  url = 'https://github.com/altmany/export_fig/archive/refs/heads/master.zip';
    outfilename = websave([pwd,'/export_fig-master'],url);
    unzip('export_fig-master.zip')
end
addpath([pwd,'/export_fig-master'])

%%

nsimul = 50;
k = 100; %starting point -- to avoid burn-in period
m = 2*k; 
Lmax = 12;
Lmin =  5;
Ls = 9;
alpha = 0.9;
sig = 3;
kap = sig^2/2;
mcmc_maxiter = 1e4;

%% pi is 1-dimensional-Gaussian
dim = 10;
d = dim^3;
tau = 2;
beta = 0.5;
gamma = 0.1;

params1 = {dim,tau*gamma,1-tau,tau*beta};

% for printing: Only print first 5
outputstr = [repmat('%.5f  ', 1, 5) '\n'];

%% Truth
mu = zeros(1,d);

%% 1) MCMC to get the reference
% tic;
% mu0 = zeros(1,d);
% mu0(1) = 10;
% mu_RWMHh = zeros(nsimul,d);
% arh = zeros(nsimul,1);
% parfor h = 1:nsimul
%     [mu_RWMHh(h,:),arh(h)] = RWMH(mu0, 1e3, 1e6, 0.0003,params1);
% end
% mu_RWMH = mean(mu_RWMHh);
% fprintf(['AR= %.4f ','mu_RWMH = ',outputstr, ],mean(arh), mu_RWMH(1:5)) 
% fprintf('-----------------------------------------------------------------\n')
% toc;
% 
% %% %2) Tamed ULA
% tic;
% mu_TULAh = zeros(nsimul,d);
% parfor h = 1:nsimul
%     mu_TULAh(h,:) = TULA(mu0, 1e3, 1e6, 0.01, params1);
% end
% mu_TULA = mean(mu_TULAh);
% fprintf(['mu_TULA = ',outputstr], mu_TULA(1:5)) 
% fprintf('---------------------------------------------------------------\n')
% toc;


%% Unbiased Estimator 
% \varphi(x) = x;

M_samples_ub = [52,520,5200,52000]*2;

MSE_len_ub = length(M_samples_ub);
Time_ub = zeros(MSE_len_ub,1);
Cost_ub = zeros(MSE_len_ub,1);
MSE_ub = zeros(MSE_len_ub,1);
muApprox_ub = zeros(MSE_len_ub,d);

for j = 1 : MSE_len_ub

    ub_pi_phi = zeros(nsimul,d);
    time = zeros(nsimul,1);
    CostForm = zeros(nsimul,1);

    M = M_samples_ub(j);
    params = {d,mcmc_maxiter,M,Lmin, Lmax, sig,k,m, kap, alpha,Ls,params1};
        
    for h = 1 : nsimul
        [ub_pi_phi(h,:),time(h),CostForm(h)] = ub_Expec_GinzLand(params);
        fprintf(['h = %d, ub_pi_phi = ',outputstr], h, ub_pi_phi(h,1:5))
        if j > 3 && mod(h,5)== 0
            poolobj = gcp('nocreate');
            delete(poolobj);
            pause(1)
        end
    end
    
    muApprox_ub(j,:) = mean(ub_pi_phi);
    MSE_ub(j) = mean(sum((ub_pi_phi - mu).^2,2));
    Time_ub(j) = mean(time);
    Cost_ub(j) = mean(CostForm);
    fprintf('-------------------------%d-------------------------------\n',j)
    fprintf(['TRUTH = ',outputstr,'APPROX = ',outputstr],mu(1:5),muApprox_ub(j,1:5))
    fprintf('nsimul = %d, M = %d, MSE = %.5E\n',nsimul, M, MSE_ub(j))
    fprintf('AvgTime = %.3f, AvgCost = %.3f\n', Time_ub(j), Cost_ub(j))
    fprintf('------------------------------------------------------------\n')

end


%% Single Level Estimator
% \varphi(x) = x;
epsil = sqrt([0.05, 0.01, 0.005, 0.001, 0.0005]); %MSE
M_samples_single = floor(3 * epsil.^(-2));

MSE_len_s= length(epsil);
Time_single = zeros(MSE_len_s,1);
Cost_single = zeros(MSE_len_s,1);
MSE_single = zeros(MSE_len_s,1);
muApprox_s = zeros(MSE_len_s,d);



for j = 1 : MSE_len_s
    
    s_pi_phi = zeros(nsimul,d);
    time = zeros(nsimul,1);
    CostForm = zeros(nsimul,1);

    L = ceil(-1.7* log2(epsil(j)));
    M = M_samples_single(j);
       
    params = {d,mcmc_maxiter,M,L,sig,k,m, kap, alpha,Ls,params1};

    for h = 1 : nsimul
        [s_pi_phi(h,:),time(h),CostForm(h)] = single_Expec_GinzLand(params);
        fprintf(['h = %d, s_pi_phi = ',outputstr], h, s_pi_phi(h,1:5))
        if j > 3 && mod(h,5)== 0
            poolobj = gcp('nocreate');
            delete(poolobj);
            pause(1)
        end
    end
    muApprox_s(j,:) = mean(s_pi_phi);
    MSE_single(j) = mean(sum((s_pi_phi - mu).^2,2));
    Time_single(j) = mean(time);
    Cost_single(j) = mean(CostForm);
    fprintf('-------------------------%d-------------------------------\n',j)
    fprintf(['TRUTH = ',outputstr,'APPROX = ',outputstr],mu(1:5),muApprox_s(j,1:5))
    fprintf('nsimul = %d, L = %d, M = %d, MSE = %.5E\n',nsimul,L,M,MSE_single(j))
    fprintf('AvgTime = %.3f, AvgCost = %.3f\n', Time_single(j), Cost_single(j))
    fprintf('------------------------------------------------------------\n')
end
%%
if save_file 
    subfolder = sprintf('Gizb_Land_Date%d_%d_%d_%d_%d', date(1:5));
    mkdir(subfolder)
    writematrix(mu, [subfolder,'/true_mu.dat'])
end

%%
if save_file 
    writematrix(muApprox_ub, [subfolder,'/muApprox_ub.dat'])
    writematrix(MSE_ub, [subfolder,'/MSE_ub.dat'])
    writematrix(Cost_ub, [subfolder,'/Cost_ub.dat'])
    writematrix(Time_ub, [subfolder,'/Time_ub.dat'])
end
%%
if save_file 
    writematrix(muApprox_s, [subfolder,'/muApprox_s.dat'])
    writematrix(MSE_single, [subfolder,'/MSE_single.dat'])
    writematrix(Cost_single, [subfolder,'/Cost_single.dat'])
    writematrix(Time_single, [subfolder,'/Time_single.dat'])
end

    
%% Plotting

log2_MSE_ub = log2(MSE_ub);
log2_Cost_ub  = log2(Cost_ub);
Pol_ub = polyfit(log2_MSE_ub,log2_Cost_ub, 1);
Cost_fit_ub = 2.^(Pol_ub(1) * log2_MSE_ub + Pol_ub(2));

log2_MSE_single = log2(MSE_single);
log2_Cost_single  = log2(Cost_single);
Pol_single = polyfit(log2_MSE_single,log2_Cost_single, 1);
Cost_fit_single = 2.^(Pol_single(1) * log2_MSE_single + Pol_single(2));


set(0, 'defaultLegendInterpreter','latex');
set(0, 'defaultTextInterpreter','latex');
%set the background of the figure to be white
set(0,'defaultfigurecolor',[1 1 1])

txt1 = sprintf('Unbiased Fit $O(\\epsilon^{%.3f})$',2*Pol_ub(1));
txt2 = sprintf('Single Fit $O(\\epsilon^{%.3f})$',2*Pol_single(1));
figure('Position', [800 800 1000 600])
loglog(MSE_ub, Cost_ub,'rs','DisplayName','Unbiased','MarkerSize',30,...
    'MarkerEdgeColor','red','MarkerFaceColor',[1 .6 .6])
hold on
loglog(MSE_ub, Cost_fit_ub,'-b','DisplayName',txt1, 'LineWidth',5)
hold on
loglog(MSE_single,Cost_single,'mv','DisplayName','Single','MarkerSize',30,...
    'MarkerEdgeColor','m','MarkerFaceColor',[1 0 .6])
hold on
loglog(MSE_single,Cost_fit_single,'-k','DisplayName',txt2,'LineWidth',5)
hold off
grid on
ax = gca;
ax.GridAlpha = 0.5;
ax.FontSize = 40;

legend('Location','southwest')
set(legend, 'FontSize', 45)
legend('show');
xlabel('MSE', 'FontSize', 45)
ylabel('Cost','FontSize', 45)
title('Ginzburg Landau Model','FontSize', 45)


if save_file 
    file_name = [subfolder,'/GinzLand_mse_cost.pdf'];
    export_fig(file_name, '-q101')
end
%%
figure
set(gcf, 'Position', [700 600 1000 1000]);
loglog(MSE_ub, Cost_ub, '-ro', 'LineWidth', 3)
xlabel('MSE')
ylabel('Cost')
ax = gca;
ax.FontSize = 35;


%%
figure
set(gcf, 'Position', [700 600 1000 1000]);
loglog(M_samples_ub, MSE_ub, '-ro', 'LineWidth', 3)
xlabel('M')
ylabel('MSE')
ax = gca;
ax.FontSize = 35;


%%
figure
set(gcf, 'Position', [700 600 1000 1000]);
loglog(MSE_ub, Time_ub, '-ro', 'LineWidth', 3)
xlabel('MSE')
ylabel('Avg Time')
ax = gca;
ax.FontSize = 35;
%% Functions
function [ub_pi_phi,time,CostForm] = ub_Expec_GinzLand(params)
    
    d = params{1}; mcmc_maxiter = params{2}; M = params{3}; 
    Lmin = params{4}; Lmax = params{5}; sig = params{6};
    k = params{7}; m = params{8}; kap = params{9}; alpha = params{10};
    Ls = params{11}; params1 = params{12}; 

    u0 = [zeros(1,d),...
            mvnrnd(zeros(1,d),sig^2/kap * eye(d))]; % x0, v0
    u0_tilde = u0;
    uf0 = u0;
    uf0_tilde = u0;
    uc0 = u0;
    uc0_tilde = u0;
        
    t1 = tic;
    %\phi(x,v) = x;  where U =(x,v);

    P_L = 2.^(-(3/2)*(Lmin:Lmax));
    P_L = P_L/sum(P_L);

    pi_phi_k_m = zeros(M,d);
    cost  = zeros(M,1);
    parfor i = 1 : M
        L =  randsample(Lmax-Lmin+1,1,true,P_L) + Lmin - 1;
        if L == Lmin
            %Initialization of the 2 coupled chains at level Lmin
            [u0s,~] = sample_from_Q(u0,u0_tilde,d,Lmin,sig,kap,...
                                       params1,Ls);

            % start the chains
            [pi_phi_k_m(i,:),cost(i)] = pi_phi_single(d,mcmc_maxiter, u0s,...
                                    u0_tilde,L,sig,kap,alpha,k,m,...
                                    params1,Ls);
        else
            %Initialization of the4 coupled chains at levels > Lmin        
            [uf0s,uc0s] = sample_from_Kfc(uf0,uc0,d,L,sig,kap,...
                                            params1,Ls);
            
            % start the chains
            [pi_phi_k_m(i,:),cost(i)] = pi_phi_coupled(d,mcmc_maxiter,uf0s,...
                                        uf0_tilde,uc0s,uc0_tilde,L,sig,...
                                        kap,alpha,k,m,params1,Ls);
        end
        pi_phi_k_m(i,:) = pi_phi_k_m(i,:)/P_L(L-Lmin + 1);
    end
    time = toc(t1);

    ub_pi_phi = mean(pi_phi_k_m);
    CostForm = sum(cost);
end

function [sing_pi_phi,time,CostForm] = single_Expec_GinzLand(params)
    
    t1 = tic;
    d = params{1}; mcmc_maxiter = params{2}; M = params{3}; 
    L = params{4}; sig = params{5};
    k = params{6}; m = params{7}; kap = params{8}; alpha = params{9};
    Ls = params{10}; params1 = params{11}; 
    
    
    u0   = [zeros(1,d),mvnrnd(zeros(1,d),sig^2/kap * eye(d))]; % x0, v0
    u0_tilde = u0;     %x0_tild,v0_tild


    %\phi(x,v) = x;  where U =(x,v);
    pi_phi_k_m = zeros(M,d);
    cost  = zeros(M,1);
    parfor i = 1 : M
        %Initialization of the 2 coupled chains at level L
        [u0s,~] = sample_from_Q(u0,u0_tilde,d,L,sig,kap,params1,Ls);
        %run the algorithm at level L
        [pi_phi_k_m(i,:),cost(i)] = pi_phi_single(d,mcmc_maxiter, u0s,...
                                    u0_tilde,L,sig,kap,alpha,k,m,...
                                    params1,Ls);
    end
    time = toc(t1);

    sing_pi_phi = mean(pi_phi_k_m);
    CostForm = sum(cost);
end


function [pi_phi_k_m, cost] = pi_phi_single(d,mcmc_maxiter,u0,u0_tilde,L,...
                                    sig,kap,alpha,k,m,params1,Ls)

    U = zeros(ceil(mcmc_maxiter/2),2*d);
    U_tilde = zeros(ceil(mcmc_maxiter/2),2*d);
    U(1,:) = u0;
    U_tilde(1,:) = u0_tilde;
    stop_time = mcmc_maxiter;
    cost = 0;
    identical = false;
    for i =  2 : mcmc_maxiter
        
        if rand >= alpha 
            
            [uP,uP_tilde, identical] = sample_from_P(u0,u0_tilde,d,L,sig,kap,...
                                        params1,Ls);
            U(i,:) = uP;
            U_tilde(i,:) = uP_tilde;
            
            u0 = uP;
            u0_tilde = uP_tilde;
            cost = cost + 2*2^L-2;
            
        else
            
            [uQ,uQ_tilde]=sample_from_Q(u0,u0_tilde,d,L,sig,kap,...
                                            params1,Ls);
            U(i,:) = uQ;
            U_tilde(i,:) = uQ_tilde;
            
            u0 = uQ;
            u0_tilde = uQ_tilde;
            
            cost = cost + 2*2^L; %cost of sampling x and v (from SDEs)      
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
                    + sum(temp.*(U(k+1:end-1,1:d) - U_tilde(k+1:end-1,1:d)));
    else 
        %start with k = 0.5 * stop_time;
        k0 = min(m,stop_time-1);
        k = min(ceil(stop_time/2),ceil(m/2));
        temp = min(1, ((k+1:stop_time-1)-1)/(k0-k+1))';
        pi_phi_k_m = 1/(k0-k+1) * sum(U(k:k0,1:d)) ...
                    + sum(temp.* (U(k+1:end-1,1:d)-U_tilde(k+1:end-1,1:d)));
    end
end


function [pi_phi_k0_m, cost] = pi_phi_coupled(d,mcmc_maxiter,uf0,uf0_tilde,...
                        uc0, uc0_tilde,L,sig,kap,alpha,k,m,params1,Ls)
    
    Uf = zeros(ceil(mcmc_maxiter/2),2*d);
    Uf_tilde = zeros(ceil(mcmc_maxiter/2),2*d);
    Uc = zeros(ceil(mcmc_maxiter/2),2*d);
    Uc_tilde = zeros(ceil(mcmc_maxiter/2),2*d);


    %initialize the chains
    Uf(1,:) = uf0;
    Uf_tilde(1,:) = uf0_tilde;
    Uc(1,:) = uc0;
    Uc_tilde(1,:) = uc0_tilde;

    %
    cost = 0;

    identical_f = false;
    identical_c = false;


    for i =  2 : mcmc_maxiter
        
        %(uf,uf_tilde,uc,uc_tilde) in D^2
        if identical_f && identical_c  

            [ufQ,ufQ_tilde,ucQ,ucQ_tilde]=sample_from_Q_fc(uf0,uf0_tilde,...
                                   uc0,uc0_tilde,d,L,sig,kap,params1,Ls);
            Uf(i,:) = ufQ;
            Uf_tilde(i,:) = ufQ_tilde;
            Uc(i,:) = ucQ;
            Uc_tilde(i,:) = ucQ_tilde;

            uf0 = ufQ;
            uf0_tilde = ufQ_tilde;
            uc0 = ucQ;
            uc0_tilde = ucQ_tilde;
            cost = cost + 2*(2^L+2^(L-1));
        else
            if rand >= alpha %&& distf <1e-12 && distc <1e-12
                [ufP,ufP_tilde,ucP,ucP_tilde, identical_f, identical_c]=...
                        sample_from_P_fc(uf0,uf0_tilde,uc0,uc0_tilde,d,L,...
                        sig,kap,params1,Ls,identical_f, identical_c);
                %fprintf('f = %d, c =%d, i = %d \n', identical_f, identical_c, i)    
                Uf(i,:) = ufP;
                Uf_tilde(i,:) = ufP_tilde;
                Uc(i,:) = ucP;
                Uc_tilde(i,:) = ucP_tilde;

                uf0 = ufP;
                uf0_tilde = ufP_tilde;
                uc0 = ucP;
                uc0_tilde = ucP_tilde; 
                cost = cost + 2*(2^L+2^(L-1))-4;
            else
                [ufQ,ufQ_tilde,ucQ,ucQ_tilde] = sample_from_Q_fc(uf0,uf0_tilde,...
                                  uc0,uc0_tilde,d,L,sig,kap,params1,Ls);
                Uf(i,:) = ufQ;
                Uf_tilde(i,:) = ufQ_tilde;
                Uc(i,:) = ucQ;
                Uc_tilde(i,:) = ucQ_tilde;

                uf0 = ufQ;
                uf0_tilde = ufQ_tilde;
                uc0 = ucQ;
                uc0_tilde = ucQ_tilde;
                cost = cost + 2*(2^L+2^(L-1));
            
            end
        end
       
        if identical_f && identical_c
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


function [u,u_tilde] = sample_from_Q(u0,u0_tilde,d,L,sig,kap,params1,Ls)
       
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
        b = [-gradU(X_old(1:d),params1),...
             -gradU(X_old(d+1:end),params1)];
        X = X_old + V_old * h + sig_L * Gamma(k,:);
        V = V_old + (b - kap * V_old) * h + sig * dB(k,:);
        X_old = X;
        V_old = V;
    end
    
    u = [X(1:d),V(1:d)];
    u_tilde = [X(d+1:end),V(d+1:end)];
end


function [U,U_tilde, identical] = sample_from_P(u0,u0_tilde,d,L,sig,kap,params1,Ls)
    
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
        b = [-gradU(X_old(1:d),params1),...
             -gradU(X_old(d+1:end),params1)];
        X = X_old + V_old * h + sig_L * Gamma(k,:);
        V = V_old + (b - kap * V_old) * h + sig * dB(k,:);
        X_old = X;
        V_old = V;
    end
        
    b = -gradU(X(1:d),params1);
    p_mu = [X(1:d),V(1:d)] + [V(1:d),b-kap*V(1:d)] * h;
    b = -gradU(X(d+1:end),params1);
    q_mu = [X(d+1:end),V(d+1:end)]+[V(d+1:end),b-kap*V(d+1:end)]* h;

    [U,U_tilde, identical] = refl_max_coupl(d, p_mu,q_mu, h, sig_L,sig);
end


function [ufQ,ufQ_tilde,ucQ,ucQ_tilde] = sample_from_Q_fc(uf0,uf0_tilde,...
                                            uc0,uc0_tilde,d, L,sig,...
                                            kap,params1,Ls)
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
            bf = [-gradU(Xf_old(1:d),params1),...
                    -gradU(Xf_old(d+1:end),params1)];
            Xf = Xf_old + Vf_old * hf + sig_Lf * Gammaf(2*(k-1)+j,:);
            Vf = Vf_old + (bf -kap*Vf_old)*hf+sig*dBf(2*(k-1)+j,:);
            Xf_old = Xf;
            Vf_old = Vf;
            
            Gammac = Gammac + Gammaf(2*(k-1)+j,:);
            dBc = dBc + dBf(2*(k-1)+j,:);
        end
        
        bc = [-gradU(Xc_old(1:d),params1),...
                -gradU(Xc_old(d+1:end),params1)];
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
            sample_from_P_fc(uf0,uf0_tilde,uc0,uc0_tilde,d, L,sig,kap,...
            params1,Ls,identical_f,identical_c)
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
                bf = [-gradU(Xf_old(1:d),params1),...
                        -gradU(Xf_old(d+1:end),params1)];
                Xf = Xf_old + Vf_old * hf + sig_Lf * Gammaf(2*(k-1)+j,:);
                Vf = Vf_old + (bf -kap*Vf_old)*hf+sig*dBf(2*(k-1)+j,:);
                Xf_old = Xf;
                Vf_old = Vf;

                Gammac = Gammac + Gammaf(2*(k-1)+j,:);
                dBc = dBc + dBf(2*(k-1)+j,:);
            end

            bc = [-gradU(Xc_old(1:d),params1),...
                    -gradU(Xc_old(d+1:end),params1)];
            Xc = Xc_old + Vc_old * hc + sig_Lc * Gammac;
            Vc = Vc_old + (bc - kap * Vc_old) * hc + sig * dBc;
            Xc_old = Xc;
            Vc_old = Vc;
        end

        bf = -gradU(Xf(1:d),params1);
        muf = [Xf(1:d),Vf(1:d)] + [Vf(1:d), bf-kap*Vf(1:d)] * hf;
        bf = -gradU(Xf(d+1:end),params1);
        muf_tilde = [Xf(d+1:end),Vf(d+1:end)]+[Vf(d+1:end),bf-kap*Vf(d+1:end)]*hf;

        bc = -gradU(Xc(1:d),params1);
        muc = [Xc(1:d),Vc(1:d)] + [Vc(1:d), bc-kap*Vc(1:d)] * hc;
        bc = -gradU(Xc(d+1:end),params1);
        muc_tilde = [Xc(d+1:end),Vc(d+1:end)] + [Vc(d+1:end),bc-kap*Vc(d+1:end)]*hc;
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
                bf = -gradU(Xf_old,params1);
                Xf = Xf_old + Vf_old * hf + sig_Lf * Gammaf(2*(k-1)+j,1:d);
                Vf = Vf_old + (bf -kap*Vf_old)*hf+sig*dBf(2*(k-1)+j,1:d);
                Xf_old = Xf;
                Vf_old = Vf;

                Gammac = Gammac + Gammaf(2*(k-1)+j,:);
                dBc = dBc + dBf(2*(k-1)+j,:);
            end

            bc = [-gradU(Xc_old(1:d),params1),...
                    -gradU(Xc_old(d+1:end),params1)];
            Xc = Xc_old + Vc_old * hc + sig_Lc * Gammac;
            Vc = Vc_old + (bc - kap * Vc_old) * hc + sig * dBc;
            Xc_old = Xc;
            Vc_old = Vc;
        end

        bf = -gradU(Xf,params1);
        muf = [Xf,Vf] + [Vf, bf-kap*Vf] * hf;
        muf_tilde = muf;

        bc = -gradU(Xc(1:d),params1);
        muc = [Xc(1:d),Vc(1:d)] + [Vc(1:d), bc-kap*Vc(1:d)] * hc;
        bc = -gradU(Xc(d+1:end),params1);
        muc_tilde = [Xc(d+1:end),Vc(d+1:end)] + [Vc(d+1:end),bc-kap*Vc(d+1:end)]*hc;
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
                bf = [-gradU(Xf_old(1:d),params1),...
                        -gradU(Xf_old(d+1:end),params1)];
                Xf = Xf_old + Vf_old * hf + sig_Lf * Gammaf(2*(k-1)+j,:);
                Vf = Vf_old + (bf -kap*Vf_old)*hf+sig*dBf(2*(k-1)+j,:);
                Xf_old = Xf;
                Vf_old = Vf;

                Gammac = Gammac + Gammaf(2*(k-1)+j,1:d);
                dBc = dBc + dBf(2*(k-1)+j,1:d);
            end

            bc = -gradU(Xc_old,params1);
            Xc = Xc_old + Vc_old * hc + sig_Lc * Gammac;
            Vc = Vc_old + (bc - kap * Vc_old) * hc + sig * dBc;
            Xc_old = Xc;
            Vc_old = Vc;
        end

        bf = -gradU(Xf(1:d),params1);
        muf = [Xf(1:d),Vf(1:d)] + [Vf(1:d), bf-kap*Vf(1:d)] * hf;
        bf = -gradU(Xf(d+1:end),params1);
        muf_tilde = [Xf(d+1:end),Vf(d+1:end)]+[Vf(d+1:end),bf-kap*Vf(d+1:end)]*hf;

        bc = -gradU(Xc(1:d),params1);
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
                bf = -gradU(Xf_old,params1);
                Xf = Xf_old + Vf_old * hf + sig_Lf * Gammaf(2*(k-1)+j,:);
                Vf = Vf_old + (bf -kap*Vf_old)*hf+sig*dBf(2*(k-1)+j,:);
                Xf_old = Xf;
                Vf_old = Vf;

                Gammac = Gammac + Gammaf(2*(k-1)+j,:);
                dBc = dBc + dBf(2*(k-1)+j,:);
            end

            bc = -gradU(Xc_old(1:d),params1);
            Xc = Xc_old + Vc_old * hc + sig_Lc * Gammac;
            Vc = Vc_old + (bc - kap * Vc_old) * hc + sig * dBc;
            Xc_old = Xc;
            Vc_old = Vc;
        end

        bf = -gradU(Xf,params1);
        muf = [Xf,Vf] + [Vf, bf-kap*Vf] * hf;
        muf_tilde = muf;

        bc = -gradU(Xc,params1);
        muc = [Xc,Vc] + [Vc, bc-kap*Vc] * hc;
        muc_tilde = muc;
    end

    [Uf,Uf_tilde, Uc, Uc_tilde, identical_f, identical_c] = ...
                synch_refl_max_coupl(d,muf, muf_tilde, muc, muc_tilde,...
                sig_Lf, sig_Lc, sig,hf, hc);
end



function [uf,uc] = sample_from_Kfc(uf0,uc0,d,L,sig,kap,params1,Ls)
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
            bf = -gradU(Xf_old,params1);
            Xf = Xf_old + Vf_old * hf + sig_Lf * Gammaf(2*(k-1)+j,:);
            Vf = Vf_old + (bf -kap*Vf_old)*hf+sig*dBf(2*(k-1)+j,:);
            Xf_old = Xf;
            Vf_old = Vf;
            
            Gammac = Gammac + Gammaf(2*(k-1)+j,:);
            dBc = dBc + dBf(2*(k-1)+j,:);
        end

        bc = -gradU(Xf_old,params1);
        Xc = Xc_old + Vc_old * hc + sig_Lc * Gammac;
        Vc = Vc_old + (bc - kap * Vc_old) * hc + sig * dBc;
        Xc_old = Xc;
        Vc_old = Vc;
    end
    
    uf = [Xf,Vf];
    uc = [Xc,Vc];
end

function [X,Y,identical] = max_coupl(d, p_mu,q_mu, h, sig_L,sig)
    % Note that this very specific to the case
    % where p & q are 2*d - dimensional Gaussian with the same covariance 
    % cov = [sig_L^2 * h * I_{dxd},  0_{dxd}                ;
    %               0_{dxd}       ,  sig^2 * h * I_{dxd}  ]
    % cov is a diagonal matrix
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

function [Xf,Yf,Xc,Yc,identical_f, identical_c] = ...
    fourWay_max_coupl(d, pf_mu, qf_mu,pc_mu, qc_mu, sig_Lf, sig_Lc, sig,...
    hf, hc,identical_f, identical_c)

    % Note that this very specific to the case
    % where pf & qf are 2d - dimensional Gaussians with the same covariance 
    % covf = [sig_Lf^2 * hf * I_{dxd},  0_{dxd}                ;
    %               0_{dxd}       ,  sig^2 * hf * I_{dxd}  ]
    % Af below is the diagonal of chol(cov) = sqrt(diag(covf))
    % Similar with pc & qc
    d1 = 2*d;
    Af = [sig_Lf * sqrt(hf) * ones(1,d), sig * sqrt(hf) * ones(1,d)];
    Ac = [sig_Lc * sqrt(hc) * ones(1,d), sig * sqrt(hc) * ones(1,d)];
  
    U = randn(1,d1) .* Af + pf_mu; %mvnrnd(pf_mu, pf_sig);
    
%     lncf = -0.5 * (d1*log2pi+logdetf);
%     lncc = -0.5 * (d1*log2pi+logdetc);
%     inv_pq_f_sig = inv(pf_sig); %pf_sig = qf_sig
%     inv_pq_c_sig = inv(pc_sig); %pc_sig = qc_sig
%     
%     la1 = lncf -0.5 * sum(((U-pf_mu)./Af).^2,2); %q_l(xf,U)    
%     la2 = lncf -0.5 * ((U-qf_mu)*inv_pq_f_sig*(U-qf_mu)'); %q_l(wf,U)    
%     la3 = lncc -0.5 * ((U-pc_mu)*inv_pq_c_sig*(U-pc_mu)'); %q_{l-1}(xc,U)
%     la4 = lncc -0.5 * ((U-qc_mu)*inv_pq_c_sig*(U-qc_mu)'); %q_{l-1}(wc,U)
%     
    la1 = log_mvnpdf1(d1,U, pf_mu, Af); %q_l(xf,U)
    la2 = log_mvnpdf1(d1,U, qf_mu, Af); %q_l(wf,U)
    la3 = log_mvnpdf1(d1,U, pc_mu, Ac); %q_{l-1}(xc,U)
    la4 = log_mvnpdf1(d1,U, qc_mu, Ac); %q_{l-1}(wc,U)
    
    la = min([la2,la3,la4],[],'all') - la1;
    
    if log(rand) <= la
        Xf = U;
        Yf = U;
        Xc = U;
        Yc = U;
        identical_f = true;
        identical_c = true;
    else
        %(a)
        Xf = U;
        if identical_f %sum(pf_mu == qf_mu) == length(pf_mu)
            Yf = U;
            identical_f = true;
        else
            %(b)
            reject = true;
            while reject
                %Uf =  mvnrnd(qf_mu, qf_sig);
                Uf = randn(1,d1) .* Af + qf_mu;
                la1 = log_mvnpdf1(d1,Uf, qf_mu, Af); %q_l(wf,Uf)
                %la1 = lncf-0.5*((Uf-qf_mu)*inv_pq_f_sig*(Uf-qf_mu)'); 
                la2 = log_mvnpdf1(d1,Uf, pf_mu, Af); %q_l(xf,Uf)
                %la2 = lncf-0.5*((Uf-pf_mu)*inv_pq_f_sig*(Uf-pf_mu)'); 
                la3 = log_mvnpdf1(d1,Uf, pc_mu, Ac); %q_{l-1}(xc,Uf)
                %la3 = lncc-0.5*((Uf-pc_mu)*inv_pq_c_sig*(Uf-pc_mu)');
                la4 = log_mvnpdf1(d1,Uf, qc_mu, Ac); %q_{l-1}(wc,Uf)
                %la4 = lncc-0.5*((Uf-qc_mu)*inv_pq_c_sig*(Uf-qc_mu)');

                la = min([la2,la3,la4],[],'all') - la1;

                reject = (log(rand) < la); %accept with prop = 1-exp(la)
            end
            Yf = Uf;
            identical_f = false;
        end
        %(c)
        reject = true;
        while reject
            %Uc = mvnrnd(pc_mu, pc_sig);
            Uc =  randn(1,d1) .* Ac + pc_mu; 
            la1 = log_mvnpdf1(d1,Uc, pc_mu, Ac); %q_{l-1}(xc,Uc)
            %la1 = lncf-0.5*((Uc-pc_mu)*inv_pq_c_sig*(Uc-pc_mu)'); 
            la2 = log_mvnpdf1(d1,Uc, pf_mu, Af); %q_l(xf,Uc)
            %la2 = lncf-0.5*((Uc-pf_mu)*inv_pq_f_sig*(Uc-pf_mu)'); 
            la3 = log_mvnpdf1(d1,Uc, qf_mu, Af); %q_l(wf,Uc)
            %la3 = lncc-0.5*((Uc-qf_mu)*inv_pq_f_sig*(Uc-qf_mu)');
            la4 = log_mvnpdf1(d1,Uc, qc_mu, Ac); %q_{l-1}(wc,Uc)
            %la4 = lncc-0.5*((Uc-qc_mu)*inv_pq_c_sig*(Uc-qc_mu)');

            la = min([la2,la3,la4],[],'all') - la1;
            reject = (log(rand) < la);
        end
        
        Xc = Uc;
        
        if identical_c %
            Yc = Xc;
            identical_c = true;
        else
            %(d)
            reject = true;
            while reject
                %Uc =  mvnrnd(qc_mu, qc_sig);
                Uc =  randn(1,d1) .* Ac + qc_mu; 

                la1 = log_mvnpdf1(d1,Uc, qc_mu, Ac); %q_{l-1}(wc,Uc)
                %la1 = lncf-0.5*((Uc-qc_mu)*inv_pq_c_sig*(Uc-qc_mu)');  
                la2 = log_mvnpdf1(d1,Uc, pf_mu, Af); %q_l(xf,Uc)
                %la2 = lncf-0.5*((Uc-pf_mu)*inv_pq_f_sig*(Uc-pf_mu)');
                la3 = log_mvnpdf1(d1,Uc, qf_mu, Af); %q_l(wf,Uc)
                %la3 = lncc-0.5*((Uc-qf_mu)*inv_pq_f_sig*(Uc-qf_mu)');
                la4 = log_mvnpdf1(d1,Uc, pc_mu, Ac); %q_{l-1}(xc,Uc)
                %la4 = lncc-0.5*((Uc-pc_mu)*inv_pq_c_sig*(Uc-pc_mu)');

                la = min([la2,la3,la4],[],'all') - la1;
                reject = (log(rand) < la);
            end
            
            Yc = Uc;
            identical_c = false;
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

function U = compute_U(x,params1)
    dim = params1{1}; taua = params1{2}; onemtau=params1{3}; taub=params1{4};
    temp1 = norm(x)^2;
    x = reshape(x, [dim,dim,dim]);
    t1 = circshift1(dim,x,-1,1) - x;
    t1 = reshape(t1,1,[]);
    t2 = circshift1(dim,x,-1,2) - x;
    t2 = reshape(t2,1,[]);
    t3 = circshift1(dim,x,-1,3) - x;
    t3 = reshape(t3,1,[]);
    temp = norm(t1)^2 + norm(t2)^2 + norm(t3)^2;
    
    U = 0.5*onemtau*temp1 + 0.5*taua*temp + 0.25*taub*temp1^2;
end

function g = gradU(x,params1)
    dim = params1{1}; taua = params1{2}; onemtau=params1{3}; taub=params1{4};
    x1 = reshape(x, [dim,dim,dim]);
    temp = circshift1(dim,x1,1,1) + circshift1(dim,x1,-1,1) ...
            +circshift1(dim,x1,1,2) + circshift1(dim,x1,-1,2) ...
            +circshift1(dim,x1,1,3) + circshift1(dim,x1,-1,3);
    g = taua*(6*x1 - temp);
    g = reshape(g,1,[]) + onemtau*x +  taub* x.^3 ;
end

function y = circshift1(d,x,n,dir)
    %input: 3d-matrix x
    index  = mod((0:d-1)-n, d) + 1;
    if dir == 1
        y = x(index,:,:);
    elseif dir == 2
        y = x(:,index,:);
    else
        y = x(:,:,index);
    end
end

% function U = compute_U(x,params1)
%     d = params1{1}; tau = params1{2}; a=params1{3}; b=params1{4};
%     dim = ceil(d^(1./3));
%     temp1 = norm(x)^2;
%     x = reshape(x, [dim,dim,dim]);
%     t1 = circshift(x,-1,1) - circshift(x,1,1);
%     t1 = reshape(t1,1,[]);
%     t2 = circshift(x,-1,2) - circshift(x,1,2);
%     t2 = reshape(t2,1,[]);
%     t3 = circshift(x,-1,3) - circshift(x,1,3);
%     t3 = reshape(t3,1,[]);
%     temp = norm(t1)^2 + norm(t2)^2 + norm(t3)^2;
%     
%     U = 0.5*(1-tau)*temp1 + (1/8)*tau*a*temp + 0.25*tau*b*temp1^2;
% end
% 
% function g = gradU(x,params1)
%     d = params1{1}; tau = params1{2}; a=params1{3}; b=params1{4};
%     dim = ceil(d^(1/3));
%     x = reshape(x, [dim,dim,dim]);
%     temp = -1/12*(circshift(x,2,1) + circshift(x,-2,1)) ...
%                 + (4/3) * (circshift(x,1,1) + circshift(x,-1,1)) ...
%           -1/12*(circshift(x,2,2) + circshift(x,-2,2)) ...
%                 + (4/3) * (circshift(x,1,2) + circshift(x,-1,2)) ...
%           -1/12*(circshift(x,2,3) + circshift(x,-2,3)) ...
%                 + (4/3) * (circshift(x,1,3) + circshift(x,-1,3));
%     g = (1-tau)*x - tau*a*(-7.5*x + temp) + tau*b* x.^3;
%     g = reshape(g,1,[]);
% end


function [m,ar] = RWMH(x0, burn_in, N, step,params1)

    d = params1{1};
    x = x0;
    sqrts = sqrt(step);
    m = zeros(size(x0)); %mean of samples 
    
    lb = -compute_U(x,params1);
    count = 0;
%     figure
%     set(gcf, 'Position', [700 600 1600 1000]);
%     h = animatedline;
%     axis([1, burn_in+N-1, -1, 1])
    for i = 2 : burn_in + N
        xp = x + sqrts * randn(1,d);
        la = -compute_U(xp,params1);
        if log(rand) <= (la-lb)
            x = xp;
            count = count + 1; %accept
            lb = la;
        end
        
        if i > burn_in
            m = m + x;
        end
%         addpoints(h,i,x(1))
%         drawnow limitrate
    end
    m = m / (N+burn_in);
    ar = count/(N+burn_in);
end

function m = TULA(x0, burn_in, N, step, params1)

    d = params1{1};
    x = x0;
    sqrtstep = sqrt(2*step);
    m = zeros(size(x0));
    
%     figure
%     set(gcf, 'Position', [700 600 1600 1000]);
%     h = animatedline;
%     axis([1, burn_in+N-1, -1, 1])

    for i = 2 : burn_in + N
        b = gradU(x,params1);
        %bTamed = b / (1 + step*norm(b));
        bTamed = b ./ (1 + step*abs(b));
        x = x - step * bTamed + sqrtstep * randn(1,d); 
        if i > burn_in
            m = m + x;
        end
        
%         addpoints(h,i,x(1))
%         drawnow limitrate
    end
    m = m /(burn_in + N);
end