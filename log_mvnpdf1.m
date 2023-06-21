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
