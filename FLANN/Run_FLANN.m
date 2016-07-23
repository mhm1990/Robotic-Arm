function O2 = Run_FLANN(X_4,Y_4,W,PST1,PST2,PS2)%PST1,PST2,PSX,PSY)
 
    %X_4 = mapminmax('apply',X_3,PSX);
    %Y_4 = mapminmax('apply',Y_3,PSY);
    R = [ones(1,size(X_4,2)); X_4; sin(pi.*X_4); sin(2*pi.*X_4); Y_4; cos(pi.*Y_4); cos(2*pi.*Y_4) ] ;
    % R = [ones(1,size(X,2)); X; sin(pi.*X); cos(pi.*X); sin(2*pi*X); Y; sin(pi.*Y); cos(pi.*Y); sin(2*pi*Y)] ;
    %display(PS2.gain);
    [Xp1] = mapminmax('apply',R,PS2);
    %W = W(1:size(Xp1,2),:);
    %Xp1=R';
    O = Xp1'*W;
    O1= tansig_apply(O);
    o_2 = O1(:,1)';
    o_3 = O1(:,2)';
    %O1 = mapminmax('reverse',O1,PST2);
    O_2 = mapminmax('reverse',o_2,PST1);
    O_3 = mapminmax('reverse',o_3,PST2);
    O2 =[O_2' O_3'];
end



% Sigmoid Symmetric Transfer Function
function a = tansig_apply(n)
  a = 2 ./ (1 + exp(-2*n)) - 1;
end


