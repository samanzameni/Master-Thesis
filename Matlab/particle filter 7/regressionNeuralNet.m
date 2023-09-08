clear;
close;

inputs = load('inputData.mat');
inputs=inputs.inputMat;
inputs=inputs';

trainNumbers=round(0.9*size(inputs,2));
testNumbers=size(inputs,2)-trainNumbers;


realTargets = load('outputData1.mat');
realTargets=realTargets.outputMat1;
realTargets=realTargets';
 
% Create a Fitting Network
hiddenLayerSize = 10;
realNet = fitnet(hiddenLayerSize);

% Set up Division of Data for Training, Validation, Testing
realNet.divideParam.trainRatio = 90/100;
realNet.divideParam.testRatio = 10/100;
 
% Train the Network
[realNet] = train(realNet,inputs(:,1:trainNumbers),realTargets(:,1:trainNumbers));
 
% Test the Network
realOutputs = realNet(inputs(:,trainNumbers+1:end));
realPerformance = perform(realNet,realTargets(:,trainNumbers+1:end),realOutputs);

figure(89);
str = strcat('real part prediction MSE = ',num2str(realPerformance));
plot(realTargets(:,trainNumbers+1:end),'-bs');
hold
plot(realOutputs,'-or');title(str);
legend('desired','predicted','Location','northwest');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


imaginaryTargets = load('outputData2.mat');
imaginaryTargets=imaginaryTargets.outputMat2;
imaginaryTargets=imaginaryTargets';
 
% Create a Fitting Network
hiddenLayerSize = 10;
imaginaryNet = fitnet(hiddenLayerSize);

% Set up Division of Data for Training, Validation, Testing
imaginaryNet.divideParam.trainRatio = 90/100;
imaginaryNet.divideParam.testRatio = 10/100;
 
% Train the Network
[imaginaryNet] = train(imaginaryNet,inputs(:,1:trainNumbers),imaginaryTargets(:,1:trainNumbers));
 
% Test the Network
imaginaryOutputs = imaginaryNet(inputs(:,trainNumbers+1:end));
imaginaryPerformance = perform(imaginaryNet,imaginaryTargets(:,trainNumbers+1:end),imaginaryOutputs);

figure(90);
str = strcat('imaginary part prediction MSE = ',num2str(imaginaryPerformance));
plot(imaginaryTargets(:,trainNumbers+1:end),'-bs');
hold
plot(imaginaryOutputs,'-or');title(str);
legend('desired','predicted','Location','northwest');

%realNet([,,]')
%imaginaryNet([,,]')