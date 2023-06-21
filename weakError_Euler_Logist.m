%% phi(u) = u^1 first coordinate.
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

%%
sig = 3;
kap = sig^2/2;
Lmax = 18;
m = 5200;
%% Logistic regression
n = 100;
d = 5;

% generate x
x = (rand(d,n)<.5)*2 - 1;
x = zscore(x,0,2);

inv_sigb = (1/n)*(x*x');
sigb = inv(inv_sigb);

mu = rand(1,d);
% generate y
p = 1./(1+exp(-mu*x));
y = binornd(1,p); %1xn

% term used in grad U
const = sum(y.*x,2)';

%% Euler-strong error for ULD
u0 = [zeros(1,d),mvnrnd(zeros(1,d),sig^2/kap * eye(d))];
LL = 1:6;
Lv = abs(LL-9);
weak_err_em = zeros(length(Lv),1);
for i = 1:length(Lv)
    L = Lv(i);
    n = 2^L;
    %Initiate vectors to store errors and time series
    u_sum = zeros(n+1,1);
    u_exact_sum = zeros(n+1,1);
    
    parfor j = 1:m
        [u_exact, dB] = sample_from_ULD_exact(u0,d,L,Lmax,sig,kap,...
                                        x, const, inv_sigb); 
        
        u_exact_sum = u_exact_sum + u_exact;
        u = sample_from_ULD(u0,d,L,sig,kap,dB,x, const, inv_sigb);
        u_sum = u_sum + u;
    end
    weak_err_em(i) = max(abs(u_sum - u_exact_sum)/m);
end

%% Plotting
dt = 2.^(LL-9)';
logdt = log2(dt);
logStrErr = log2(weak_err_em);
ply = polyfit(logdt, logStrErr,1);
err_fit = 2.^(ply(1) * logdt + ply(2));

set(0, 'defaultLegendInterpreter','latex');
set(0, 'defaultTextInterpreter','latex');
%set the background of the figure to be white
set(0,'defaultfigurecolor',[1 1 1])
figure('Position', [800 800 1000 700])


txt1 = sprintf('Slope $= %.3f$',ply(1));
plot(logdt, logStrErr,'s','DisplayName','EM Weak Error','MarkerSize',30,...
    'MarkerEdgeColor','red','MarkerFaceColor',[1 .0 .2])
hold on
plot(logdt,log2(err_fit) ,'-b','DisplayName',txt1, 'LineWidth',5)

title('Bayesian Logistic Model')

set(gca, 'XTickLabel', strcat('2^{',num2str(logdt),'}'));
yticks = get(gca, 'YTick');
set(gca, 'YTickLabel', strcat('2^{',num2str(yticks'),'}'));


grid on
ax = gca;
ax.GridAlpha = 0.3;
ax.FontSize = 43;

legend('Location','northwest')
set(legend, 'FontSize', 45)
legend('show');
xlabel('$\Delta_l$', 'FontSize', 65)
ylabel('Weak Error','FontSize', 55);
if save_file 
    writematrix(weak_err_em,'weak_err_Logist.dat')
    file_name = 'Logist_EM_weak_err.pdf';
    export_fig(file_name, '-q101')
end

%% Functions
function g = gradU(beta, x, const, inv_sigb)
%beta is 1xd
%x is dxn
%const = sum(y.*x,2)', where y is 1xn; const is 1xd
%inv_sigb is dxd
xi = 0.5;
t = exp(-beta * x); %1xn
g = -const + sum((1./(1+t)) .* x,2)' + 2*xi* beta*inv_sigb;

end


function [u, dB] = sample_from_ULD_exact(u0,d,L,Lmax,sig,kap,x,const,inv_sigb)
   
    nf = 2^Lmax;
    hf = 1/nf;
    nc = 2^L;


    Xf = u0(1:d);
    Vf = u0(d+1:end);
    
    dB = zeros(nc,d);
    Xc = zeros(nc+1,d);
    Vc = zeros(nc+1,d);
    Xc(1,:) = u0(1:d);
    Vc(1,:) = u0(d+1:end);
    step = 2^(Lmax-L);
    
    i = 1;
    for k = 1: nc 
        for m = 1:step
            b = -gradU(Xf,x, const, inv_sigb);
            Xf = Xf + Vf * hf ;
            dBf = sqrt(hf) * randn(1,d);
            Vf = Vf + (b - kap * Vf) * hf + sig * dBf;
            dB(k,:) = dB(k,:) + dBf;
            i = i + 1;
        end
        Xc(k+1,:) = Xf;
        Vc(k+1,:) = Vf;
    end
    
    u = Xc(:,1);
end


function u = sample_from_ULD(u0,d,L,sig,kap,dB,x, const, inv_sigb)
   
    n = 2^L;
    h = 1/n;

    X = zeros(n+1,d);
    V = zeros(n+1,d);
    X(1,:) = u0(1:d);
    V(1,:) = u0(d+1:end);
    
    
    for k = 1: n 
        b = -gradU(X(k,:),x, const, inv_sigb);
        X(k+1,:) = X(k,:) + V(k,:) * h ;
        V(k+1,:) = V(k,:) + (b - kap * V(k,:)) * h + sig * dB(k,:);
    end
    
    u = X(:,1);
end