%%
% cosnider the following density to sample from:
% pi(x) = exp(-U);
% Thus:
% U(x) = (1/4) * ||x||^4 - (1/2) ||x||^2
% Should choose kappa and sigma such that 
% 2*kappa = sigma^2
% b = - gradU =  -(||X||^2 - 1) * x

close all; 
clear; 
clc;
format long
rng(9)
save_file = 0;
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
Lmin =  4;
Ls = 5;
alpha = 0.9;
sig = 3;
kap = sig^2/2;
mcmc_maxiter = 2e3;
M = 5200;
d = 10^2;
coord = 1; %pick any coordinate
%% Compute increments [\pi_l(\phi) - \pi_{l-1}(\phi)], 
%   where \phi(x) = x^{(coord)} 
        
Al_h = zeros(Lmax-Lmin,d,M);

parfor h = 1:M
    
    uf0 = [zeros(1,d),mvnrnd(zeros(1,d),sig^2/kap * eye(d))]; % x0, v0
    uf0_tilde = uf0;
    uc0 = uf0;
    uc0_tilde = uf0;
    
    Al = zeros(Lmax-Lmin,d);
    for l = Lmin+1 : Lmax
        [Al(l-Lmin,:), ~] = pi_phi_coupled(d,mcmc_maxiter,uf0,uf0_tilde,uc0,...
                          uc0_tilde,l,sig,kap,alpha,k,m,Ls);
    end
    Al_h(:,:,h) = Al;
end

%%
avg = mean(Al_h(:,coord,:),3);
second_moment_incr = mean(Al_h(:,coord,:).^2,3);
LL = (Lmin+1:Lmax)';


%% Fit
%%% fit the variance
log2_var = log2(second_moment_incr);
Polv = polyfit(LL, log2_var,1);
secondM_incr_fit2 = 2.^(Polv(1) *LL + Polv(2)); %Cv2 * 2.^(-beta2*LL);
                     
%% plot
set(0, 'defaultLegendInterpreter','latex');
set(0, 'defaultTextInterpreter','latex');
%set the background of the figure to be white
set(0,'defaultfigurecolor',[1 1 1])

figure('Position', [800 800 1050 600])
txt = sprintf('Fit $O(\\Delta_l^{%.3f})$',-Polv(1));
plot(LL,log2(second_moment_incr),'sk','DisplayName',...
     '$E\left[\xi_l^2 \right]$','MarkerSize',35,...
     'MarkerEdgeColor','k','MarkerFaceColor',[0.5 .5 .5])
hold on
plot(LL,log2(secondM_incr_fit2),'-r','DisplayName',txt, 'LineWidth',5)
hold off
xticks(LL)
title('$2^{nd}$ Moment of $\xi_l$ - Double-Well Model')
yticks = get(gca, 'YTick');
set(gca, 'YTickLabel', strcat('2^{',num2str(yticks'),'}'));
grid on
xlabel('Discretization Levels $l$','FontSize',40)
ylabel('Second Moment','FontSize',45)
legend show
legend('Location','northeast')
set(legend, 'FontSize', 55)
ax = gca;
ax.GridAlpha = 0.35;
ax.FontSize = 45;

if save_file
    export_fig('incr_var_DoubWell.pdf','-m3')
end

%% Functions

function [pi_phi_k0_m, cost] = pi_phi_coupled(d,mcmc_maxiter,uf0,uf0_tilde,uc0,...
                          uc0_tilde,L,sig,kap,alpha,k,m,Ls)
    
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
                               uc0,uc0_tilde,d,L,sig,kap,Ls);
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
            
            cost = cost + 2*(2^L+2^(L-1))-4;
        end
    end

   
    if  identical_f && identical_c
          stop_time = i;
         %fprintf('stopped at iteration i = %d\n', i)
%         fprintf('distf = %.5f, distc = %.5f\n',distf, distc)
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
    
    %U(x) = 1/2 * (x-mu)^2
    %kap = 1/2
    %b = -grad U = -(x-mu)
    
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
    
    %U(x) = 1/2 * (x-mu)^2
    %kap = 1/2
    %b = -grad U = x-mu
    
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

%     [U,U_tilde, identical] = refl_max_coupl(d, p_mu,q_mu, h, sig_L,sig);
    
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
    %sig_Lc = sig_Lf;
    
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

%     [Uf,Uf_tilde, identical_f] = max_coupl(d, muf,muf_tilde, hf, sig_Lf,sig);
%     [Uc,Uc_tilde, identical_c] = max_coupl(d, muc,muc_tilde, hc, sig_Lc,sig);
end

function [uf,uc] = sample_from_Kfc(uf0,uc0,d,L,sig,kap,Ls)
    nf = 2^L;
    hf = 1/nf;
    nc = 2^(L-1);
    hc = 1/nc;
    sig_Lf = 2^(-(L+Ls));
    sig_Lc = 2^(-(L-1+Ls));
    %sig_Lc = sig_Lf;
    
    
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

% function [Xf,Yf,Xc,Yc,identical_f, identical_c] = ...
%           fourWay_max_coupl(d, pf_mu, qf_mu,pc_mu, qc_mu, sig_Lf, sig_Lc,...
%             sig, hf, hc,identical_f, identical_c)
% 
%     % Note that this very specific to the case
%     % where pf & qf are 2d - dimensional Gaussians with the same covariance 
%     % covf = [sig_Lf^2 * hf * I_{dxd},  0_{dxd}                ;
%     %               0_{dxd}       ,  sig^2 * hf * I_{dxd}  ]
%     % Af below is the diagonal of chol(cov) = sqrt(diag(covf))
%     % Similar with pc & qc
%     d1 = 2*d;
%     Af = [sig_Lf * sqrt(hf) * ones(1,d), sig * sqrt(hf) * ones(1,d)];
%     Ac = [sig_Lc * sqrt(hc) * ones(1,d), sig * sqrt(hc) * ones(1,d)];
%   
%     U = randn(1,d1) .* Af + pf_mu; %mvnrnd(pf_mu, pf_sig);
%     
% %     lncf = -0.5 * (d1*log2pi+logdetf);
% %     lncc = -0.5 * (d1*log2pi+logdetc);
% %     inv_pq_f_sig = inv(pf_sig); %pf_sig = qf_sig
% %     inv_pq_c_sig = inv(pc_sig); %pc_sig = qc_sig
% %     
% %     la1 = lncf -0.5 * sum(((U-pf_mu)./Af).^2,2); %q_l(xf,U)    
% %     la2 = lncf -0.5 * ((U-qf_mu)*inv_pq_f_sig*(U-qf_mu)'); %q_l(wf,U)    
% %     la3 = lncc -0.5 * ((U-pc_mu)*inv_pq_c_sig*(U-pc_mu)'); %q_{l-1}(xc,U)
% %     la4 = lncc -0.5 * ((U-qc_mu)*inv_pq_c_sig*(U-qc_mu)'); %q_{l-1}(wc,U)
% %     
%     la1 = log_mvnpdf1(d1,U, pf_mu, Af); %q_l(xf,U)
%     la2 = log_mvnpdf1(d1,U, qf_mu, Af); %q_l(wf,U)
%     la3 = log_mvnpdf1(d1,U, pc_mu, Ac); %q_{l-1}(xc,U)
%     la4 = log_mvnpdf1(d1,U, qc_mu, Ac); %q_{l-1}(wc,U)
%     
%     la = min([la2,la3,la4],[],'all') - la1;
%     
%     if log(rand) <= la
%         Xf = U;
%         Yf = U;
%         Xc = U;
%         Yc = U;
%         identical_f = true;
%         identical_c = true;
%     else
%         %(a)
%         Xf = U;
%         if identical_f %norm(pf_mu-qf_mu) < 1e-4 %sum(pf_mu == qf_mu) == length(pf_mu)
%             Yf = U;
%             identical_f = true;
%         else
%             %(b)
%             reject = true;
%             while reject
%                 %Uf =  mvnrnd(qf_mu, qf_sig);
%                 Uf = randn(1,d1) .* Af + qf_mu;
%                 la1 = log_mvnpdf1(d1,Uf, qf_mu, Af); %q_l(wf,Uf)
%                 %la1 = lncf-0.5*((Uf-qf_mu)*inv_pq_f_sig*(Uf-qf_mu)'); 
%                 la2 = log_mvnpdf1(d1,Uf, pf_mu, Af); %q_l(xf,Uf)
%                 %la2 = lncf-0.5*((Uf-pf_mu)*inv_pq_f_sig*(Uf-pf_mu)'); 
%                 la3 = log_mvnpdf1(d1,Uf, pc_mu, Ac); %q_{l-1}(xc,Uf)
%                 %la3 = lncc-0.5*((Uf-pc_mu)*inv_pq_c_sig*(Uf-pc_mu)');
%                 la4 = log_mvnpdf1(d1,Uf, qc_mu, Ac); %q_{l-1}(wc,Uf)
%                 %la4 = lncc-0.5*((Uf-qc_mu)*inv_pq_c_sig*(Uf-qc_mu)');
% 
%                 la = min([la2,la3,la4],[],'all') - la1;
% 
%                 reject = (log(rand) < la); %accept with prop = 1-exp(la)
%             end
%             Yf = Uf;
%             identical_f = false;
%         end
%         %(c)
%         reject = true;
%         while reject
%             %Uc = mvnrnd(pc_mu, pc_sig);
%             Uc =  randn(1,d1) .* Ac + pc_mu; 
%             la1 = log_mvnpdf1(d1,Uc, pc_mu, Ac); %q_{l-1}(xc,Uc)
%             %la1 = lncf-0.5*((Uc-pc_mu)*inv_pq_c_sig*(Uc-pc_mu)'); 
%             la2 = log_mvnpdf1(d1,Uc, pf_mu, Af); %q_l(xf,Uc)
%             %la2 = lncf-0.5*((Uc-pf_mu)*inv_pq_f_sig*(Uc-pf_mu)'); 
%             la3 = log_mvnpdf1(d1,Uc, qf_mu, Af); %q_l(wf,Uc)
%             %la3 = lncc-0.5*((Uc-qf_mu)*inv_pq_f_sig*(Uc-qf_mu)');
%             la4 = log_mvnpdf1(d1,Uc, qc_mu, Ac); %q_{l-1}(wc,Uc)
%             %la4 = lncc-0.5*((Uc-qc_mu)*inv_pq_c_sig*(Uc-qc_mu)');
% 
%             la = min([la2,la3,la4],[],'all') - la1;
%             reject = (log(rand) < la);
%         end
%         
%         Xc = Uc;
%         
%         if identical_c %norm(pc_mu-qc_mu) < 1e-4 %sum(pc_mu == qc_mu) == length(pc_mu)
%             Yc = Xc;
%             identical_c = true;
%         else
%             %(d)
%             reject = true;
%             while reject
%                 %Uc =  mvnrnd(qc_mu, qc_sig);
%                 Uc =  randn(1,d1) .* Ac + qc_mu; 
% 
%                 la1 = log_mvnpdf1(d1,Uc, qc_mu, Ac); %q_{l-1}(wc,Uc)
%                 %la1 = lncf-0.5*((Uc-qc_mu)*inv_pq_c_sig*(Uc-qc_mu)');  
%                 la2 = log_mvnpdf1(d1,Uc, pf_mu, Af); %q_l(xf,Uc)
%                 %la2 = lncf-0.5*((Uc-pf_mu)*inv_pq_f_sig*(Uc-pf_mu)');
%                 la3 = log_mvnpdf1(d1,Uc, qf_mu, Af); %q_l(wf,Uc)
%                 %la3 = lncc-0.5*((Uc-qf_mu)*inv_pq_f_sig*(Uc-qf_mu)');
%                 la4 = log_mvnpdf1(d1,Uc, pc_mu, Ac); %q_{l-1}(xc,Uc)
%                 %la4 = lncc-0.5*((Uc-pc_mu)*inv_pq_c_sig*(Uc-pc_mu)');
% 
%                 la = min([la2,la3,la4],[],'all') - la1;
%                 reject = (log(rand) < la);
%             end
%             
%             Yc = Uc;
%             identical_c = false;
%         end
%     end
% end

function [Uf1,Uf2,Uc1,Uc2,identical_f, identical_c] = ...
           fourWay_max_coupl(d, muf1, muf2, muc1, muc2, sig_Lf, sig_Lc,...
             sig, hf, hc,identical_f, identical_c)
      
    % Note that this very specific to the case
    % where pf & qf are 2d - dimensional Gaussians with the same covariance 
    % covf = [sig_Lf^2 * hf * I_{dxd},  0_{dxd}                ;
    %               0_{dxd}       ,  sig^2 * hf * I_{dxd}  ]
    % Af below is the diagonal of chol(cov) = sqrt(diag(covf))
    % Similar with pc & qc
    
%     a = norm(muf1 - muc1)
%     b = norm(muf1 - muf2)
%     c = norm(muf1 - muc2)
%     dd = norm(muc1 - muc2)
    
    d1 = 2*d;
    Af = [sig_Lf * sqrt(hf) * ones(1,d), sig * sqrt(hf) * ones(1,d)];
    Ac = [sig_Lc * sqrt(hc) * ones(1,d), sig * sqrt(hc) * ones(1,d)];
  
    Uc1 = randn(1,d1) .* Ac + muc1; %mvnrnd(pc_mu, pc_sig);
    
%      lncf = -0.5 * d1*log(2*pi) - d*(log(sig_Lf*sqrt(hf)) + log(sig*sqrt(hf)));
%      lncc = -0.5 * d1*log(2*pi) - d*(log(sig_Lc*sqrt(hc)) + log(sig*sqrt(hc)));
%      inv_pq_f_sig = inv(diag(Af.^2)); %pf_sig = qf_sig
%      inv_pq_c_sig = inv(diag(Ac.^2)); %pc_sig = qc_sig
%     
%     la1 = lncf -0.5 * sum(((U-pf_mu)./Af).^2,2); %q_l(xf,U)    
%     la2 = lncf -0.5 * ((U-qf_mu)*inv_pq_f_sig*(U-qf_mu)'); %q_l(wf,U)    
%     la3 = lncc -0.5 * ((U-pc_mu)*inv_pq_c_sig*(U-pc_mu)'); %q_{l-1}(xc,U)
%     la4 = lncc -0.5 * ((U-qc_mu)*inv_pq_c_sig*(U-qc_mu)'); %q_{l-1}(wc,U)
%     
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
%     figure
%     set(gcf, 'Position', [700 600 1600 1000]);
%     h = animatedline;
%     axis([1, burn_in+N-1, -1, 1])
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
%         addpoints(h,i,x(1))
%         drawnow limitrate
    end
    m = m / (N+burn_in);
    ar = count/(N+burn_in);
end

function m = TULA(x0, burn_in, N, step)

    d = length(x0);
    x = x0;
    sqrtstep = sqrt(2*step);
    m = zeros(size(x0));
    
%     figure
%     set(gcf, 'Position', [700 600 1600 1000]);
%     h = animatedline;
%     axis([1, burn_in+N-1, -1, 1])

    for i = 2 : burn_in + N
        b = gradU(x);
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