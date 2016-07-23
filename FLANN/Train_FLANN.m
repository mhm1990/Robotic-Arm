function [W,mse1,epoch,PST2,PS] = Train_FLANN(X,Y,D,alpha,maxepoch,MSEmin)
 
X1 = [ones(1,size(X,2)); X; sin(pi.*X); sin(2*pi.*X); Y; cos(pi.*Y); cos(2*pi.*Y) ] ; 

[Xp1,PS] = mapminmax(X1);
Xp1 = Xp1';
~iscell (Xp1) 
[D1,PST2] = mapminmax(D);

%display (D)
W = 0.5*ones(size(Xp1,2),2);
N = size(X1,1);
 
for i=1:maxepoch
    epoch(i) = i;
    O = Xp1*W;
    O1= tansig_apply(O);
   
    E1 = D1 - O1;
    
    MSE1= 1/(N) * sum(E1.^2); %Mean square error
    mse1(i,:) = MSE1;
    
    fprintf('epoch = %d  mse = %f %f  \n',i,MSE1);
    if (mse1(i,1) < MSEmin)
        break;
    end
    W = W + alpha*Xp1'*E1;
end
 
end
 
  
% Sigmoid Symmetric Transfer Function
function a = tansig_apply(n)
  a = 2 ./ (1 + exp(-2*n)) - 1;
end
