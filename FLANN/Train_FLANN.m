function [W,mse1,epoch,PST1,PST2,PS2] = Train_FLANN(X_2,Y_2,D1,D2,alpha,maxepoch,MSEmin)
%,PSX,PSY
%[X_2,PSX] = mapminmax(X_1);
%[Y_2,PSY] = mapminmax(Y_1);
X1 = [ones(1,size(X_2,2)); X_2; sin(pi.*X_2); sin(2*pi.*X_2); Y_2; cos(pi.*Y_2); cos(2*pi.*Y_2) ] ; 
%X1 = [ones(1,size(X,2)); X; sin(pi.*X); cos(pi.*X); sin(2*pi*X); Y; sin(pi.*Y); cos(pi.*Y); sin(2*pi*Y)] ; 
[Xp_1,PS2] = mapminmax(X1);
display(Xp_1)
Xp1 = Xp_1';
%D1 = [D1 D2];

[D_1,PST1] = mapminmax(D1);
[D_2,PST2] = mapminmax(D2);
%display([D_1' D_2'])

D3= [D_1' D_2'];
display(D3);
W = 0.5*rand(size(Xp1,2),2);
N = size(Xp1,1);

for i=1:maxepoch
    epoch(i) = i;
    O = Xp1*W;
    O1= tanh(O);
   display([D3 O1])
    E1 = D3 - O1;
    
    MSE1= 1/(N) * sum(E1).^2;
    mse1(i,:) = MSE1;
    
    fprintf('epoch = %d  mse = %f %f  \n',i,MSE1);
    if (mse1(i,1) < MSEmin)
        return;
    end
    W = W + alpha*Xp1'*E1;
end

end



% Sigmoid Symmetric Transfer Function
function a = tansig_apply(n)
  a = 2 ./ (1 + exp(-2*n)) - 1;
end



