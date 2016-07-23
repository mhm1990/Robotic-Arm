clc
clear all
close all
clf('reset')
clf
load 'theta.mat'

X = Z(:,1);
Y = Z(:,2);
D = THETA;
D1 = THETA(:,1);
D2 = THETA(:,2);


X_1= X';
Y_1= Y';

maxepoch = 1000;
alpha =1;
MSEmin = 0.010;
[W,MSE,epoch,PST1,PST2,PS2] = Train_FLANN(X_1,Y_1,D1',D2',alpha,maxepoch,MSEmin);
%,PSX,PSY

figure(1)
plot( epoch,MSE(:,1),'r-');
semilogy(MSE);

xlabel('No. of epochs');
ylabel('Mean-Square error for Theta1');
figure(2)
plot( epoch,MSE(:,2),'g-');
xlabel('No. of epochs');
ylabel('Mean-Square error for Theta2');

%display(W)
X = Z(350:end,1);
Y = Z(350:end,2);
d = D(350:end,:);
X_3= X';
Y_3= Y';
O3 = Run_FLANN(X_3,Y_3,W,PST1,PST2,PS2);%PST1,PST2,PSX,PSY);
o1 = O3(:,1);
o2 = O3(:,2);
sample = 1:size(X_3,2);

figure(3)
plot(sample,O3(:,1),':bs');
hold on;
plot(sample,d(:,1),'--mo');

figure(4)
plot(sample,O3(:,2),':bs');
hold on;
plot(sample,d(:,2),'--mo');
X =X_3';
Y=Y_3';
fprintf('   X        Y      T1      T2      NNT1     NNT2    ET1     ET2\n');
for i=1:1:size(X,1)
    fprintf ('%3.4f  %3.4f  %3.4f  %3.4f   %3.4f  %3.4f  %3.4f  %3.4f\n',X(i), Y(i), d(i,:), O3(i,:), d(i,:)-O3(i,:));
end
%P = D-O;
%save 'test.xls' X Y D O P
fprintf('\nTraining Set Accuracy: %f\n', mean(double(O3 == d)) * 100);

