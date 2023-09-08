clear;
close;


X = load('inputData.mat');
X=X.inputMat;

realPart = load('outputData1.mat');
realPart= realPart.outputMat1;

trainNumbers=round(0.9*size(X,1));
testNumbers=size(X,1)-trainNumbers;

%support vector machine (SVM)
realModel = fitrsvm(X(1:trainNumbers,:),realPart(1:trainNumbers,:));
%Predict Real model
realPredict = predict(realModel,X(trainNumbers+1:end,:));

realError=immse(realPart(trainNumbers+1:end,:),realPredict);

figure(87);
str = strcat('real part prediction MSE = ',num2str(realError));
plot(realPart(trainNumbers+1:end,:),'-bs');
hold
plot(realPredict,'-or');title(str);
legend('desired','predicted','Location','northwest');

imagePart = load('outputData2.mat');
imagePart= imagePart.outputMat2;
%support vector machine (SVM)
imaginaryModel= fitrsvm(X(1:trainNumbers,:),imagePart(1:trainNumbers,:));
%predict imaginary model
imaginaryPredict = predict(imaginaryModel,X(trainNumbers+1:end,:));
imaginaryError=immse(imagePart(trainNumbers+1:end,:),imaginaryPredict);

figure(88);
str = strcat('imaginary part prediction MSE = ',num2str(imaginaryError));
plot(imagePart(trainNumbers+1:end,:),'-bs');
hold
plot(imaginaryPredict,'-or');title(str);
legend('desired','predicted','Location','northwest');

%predict(realModel,[,,])
%predict(imaginaryModel,[,,])