close all; 
clear; 
clc;
format long
rng(9)
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

M = 5200;
k = 100; %starting point -- to avoid burn-in period
m = 2*k; 
Lmax = 10;
lmin =  4;
Ls = 14;
alpha = 0.9;
sig = 3;
kap = sig^2/2;
mcmc_maxiter = 2e3;

%% pi 
dim = 10;
d = dim^3;
tau = 2;
beta = 0.5;
gamma = 0.1;

params1 = {dim,tau*gamma,1-tau,tau*beta};

% for printing: Only print first 5
outputstr = [repmat('%.5f  ', 1, 5) '\n'];

%% ML sampler
        

Al_h = zeros(Lmax-lmin,d,M);

parfor h = 1 :M
    
    uf0 = [zeros(1,d),...
        mvnrnd(zeros(1,d),sig^2/kap * eye(d))]; % x0, v0
    uf0_tilde = uf0;%[zeros(1,d),...
        %mvnrnd(zeros(1,d),sig^2/kap * eye(d))]; % x0_tild, v0_tild 
    uc0 = uf0 ;%[zeros(1,d),...
        %mvnrnd(zeros(1,d),sig^2/kap * eye(d))]; % x0, v0
    uc0_tilde = uf0; %[zeros(1,d),...
        %mvnrnd(zeros(1,d),sig^2/kap * eye(d))]; % x0, v0
    
    Al = zeros(Lmax-lmin,d);
    for l = lmin+1 : Lmax
        [Al(l-lmin,:), ~] = pi_phi_coupled(d,mcmc_maxiter,uf0,uf0_tilde,...
                        uc0, uc0_tilde,l,sig,kap,alpha,k,m,params1,Ls);
    end
    Al_h(:,:,h) = Al;
end

%%
coord = 1;
avg = mean(Al_h(:,coord,:),3);
second_moment_incr = mean(Al_h(:,coord,:).^2,3);
LL = (lmin+1:Lmax)';


% Fit


%%% Variance
%%% fit the variance
log2_var = log2(second_moment_incr);
Polv = polyfit(LL, log2_var,1);

%beta1 = 1;
%beta2 = 2;
%Cv2 = 2^(log2(var_incr(1))+ beta2 *(lmin+1));
secondM_incr_fit2 = 2.^(Polv(1) *LL + Polv(2)); %Cv2 * 2.^(-beta2*LL);

% count = 1;
% k = 0.001;
% while count > 0
%     count  = sum(log2(var_incr_fit2) < log2_var);
%     if count > 0
%         Cv2 = Cv2+k;
%         var_incr_fit2 = Cv2 * 2.^(-beta2*LL); %beta = |Polv(1)|
%         k = k + 0.001;
%     end
% end

                     
% plot
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
title('$2^{nd}$ Moment of $\xi_l$ - Ginzburg-Landau Model')

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

export_fig('incr_var_GinzLandau.pdf','-m3')


%% Functions

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

        %distf = norm(Uf(i,:)-Uf_tilde(i,:));
        %distc = norm(Uc(i,:)-Uc_tilde(i,:));

        
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
                    + sum(temp.* (Uf(k+1:end-1,1:d) - Uf_tilde(k+1:end-1,1:d)));
        pi_phi_k0_m_c = 1/(k0-k+1) * sum(Uc(k:k0,1:d)) ...
                    + sum(temp.* (Uc(k+1:end-1,1:d) - Uc_tilde(k+1:end-1,1:d) ) );
    end

    pi_phi_k0_m = pi_phi_k0_m_f - pi_phi_k0_m_c;
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
    %sig_Lc = sig_Lf;
    
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

%     [Uf,Uf_tilde,identical_f] = refl_max_coupl(d, muf,muf_tilde, hf, sig_Lf,sig);
%     [Uc,Uc_tilde,identical_c] = refl_max_coupl(d, muc,muc_tilde, hc, sig_Lc,sig);

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



function ly = log_stand_mvnpdf(d,X) 
% evaluate the log Multivariate Standard Normal Distribution at X 
    ly = -0.5*sum(X.^2, 2) - d*log(2*pi)/2; 
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


