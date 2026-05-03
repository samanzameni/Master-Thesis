
X = load('inputData.mat');
X=X.inputMat;

realPart = load('outputData1.mat');
realPart= realPart.outputMat1;

trainNumbers=round(0.9*size(X,1));
testNumbers=size(X,1)-trainNumbers;

%% binary decision tree
realModel = fitrtree(X(1:trainNumbers,:),realPart(1:trainNumbers,:));

% Perform 10-fold cross-validation
realcv = crossval(realModel, 'KFold', 10 );

% Obtain cross-validated loss
realloss = kfoldLoss(realcv);

disp(['DT Cross-validated realloss: ' num2str(realloss)]);

%Predict Real model
realPredict = predict(realModel,X(trainNumbers+1:end,:));

realError=immse(realPart(trainNumbers+1:end,:),realPredict);

disp(['DT X-coordinate prediction MSE = ' num2str(realError)]);

figure(1015);
str = strcat('DT X-coordinate Cross-validated loss = ',num2str(realloss));
plot(realPart(trainNumbers+1:end,:),'-bs');
hold
plot(realPredict,'-or');title(str);
legend('desired','predicted','Location','northwest');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

imagePart = load('outputData2.mat');
imagePart= imagePart.outputMat2;
%% binary decision tree
imaginaryModel= fitrtree(X(1:trainNumbers,:),imagePart(1:trainNumbers,:));

% Perform 10-fold cross-validation
imaginarycv = crossval(imaginaryModel, 'KFold', 10);

% Obtain cross-validated loss
imaginaryloss = kfoldLoss(imaginarycv);

disp(['DT Cross-validated imaginaryloss: ' num2str(imaginaryloss)]);

%predict imaginary model
imaginaryPredict = predict(imaginaryModel,X(trainNumbers+1:end,:));

imaginaryError=immse(imagePart(trainNumbers+1:end,:),imaginaryPredict);

disp(['DT Y-coordinate prediction MSE = ' num2str(imaginaryError)]);

figure(1016);
str = strcat('DT Y-coordinate Cross-validated loss = ',num2str(imaginaryloss));
plot(imagePart(trainNumbers+1:end,:),'-bs');
hold
plot(imaginaryPredict,'-or');title(str);
legend('desired','predicted','Location','northwest');