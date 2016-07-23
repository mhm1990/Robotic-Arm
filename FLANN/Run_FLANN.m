function O1 = Run_FLANN(X,Y,W,PST,PS)%PST1,PST2)
 
 
 R = [ones(1,size(X,2)); X; sin(pi.*X); sin(2*pi.*X); Y; cos(pi.*Y); cos(2*pi.*Y) ] ;
 
%[Xp1,PS] = mapminmax(R);
[Xp1] = mapminmax('apply',R,PS);

Xp1 = Xp1';
O = Xp1*W;
O1= tansig_apply(O);
 
O1 = mapminmax('reverse',O1,PST);


end
 
 
 
% Sigmoid Symmetric Transfer Function
function a = tansig_apply(n)
  a = 2 ./ (1 + exp(-2*n)) - 1;
end
