tic
clc
clear all
close all
clf('reset')
clf
load 'theta.mat'
X = Z (:,1); %input data
Y = Z (:,2); %input data
D = THETA; %desired output data
X= X'; %Re-arrange data for training
Y= Y'; %Re-arrange data for training
maxepoch = 1000;
alpha =0.75;
MSEmin = 0.01;
[W,MSE,epoch,PST,PS] = Train_FLANN(X,Y,D,alpha,maxepoch,MSEmin);%call to training function
figure(1)
plot( epoch,MSE(:,1),'r-');
xlabel('No. of epochs');
ylabel('Mean-Square error for Theta1');
figure(2)
plot( epoch,MSE(:,2),'g-');
xlabel('No. of epochs');
ylabel('Mean-Square error for Theta2');
%display(W)
X = Z(:,1);
Y = Z(:,2);
X= X';
Y= Y';
O = Run_FLANN(X,Y,W,PST,PS);
o1 = O(:,1);
o2 = O(:,2);
sample = 1:size(X,2);
d = D(1:size(X,2),1);
figure(3)
plot(sample,O(:,1),':bs');
hold on;
plot(sample,D(:,1),'--mo');
xlabel('No of samples');
ylabel('Theta 1 in radian');
title('Matching plot for theta1');

figure(4)
plot(sample,O(:,2),':bs');
hold on;
plot(sample,D(:,2),'--mo');
xlabel('No of samples');
ylabel('Theta 2 in radian');
title('Matching plot for theta2');
X =X';
Y=Y';
fprintf(' X Y T1 T2 NNT1 NNT2 ET1 ET2\n');
for i=1:1:size(X,1)
fprintf ('%3.4f %3.4f %3.4f %3.4f %3.4f %3.4f %3.4f %3.4f\n',X(i), Y(i), D(i,:), O(i,:), D(i,:)- O(i,:));
end

fprintf('\nTraining Set Accuracy: %f\n', mean(double(O == D)) * 100);
toc