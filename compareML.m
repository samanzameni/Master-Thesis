clear all
close all
clc

% Specify the file name for storing the command window output
outputFile = 'output.txt';

% Open the diary file to start recording the command window output
diary(outputFile);

for i = 1:10
    disp(['Running loop ' num2str(i)]);
    main;
    regressionANN;
    regressionDT;
    regressionE;
    regressionSVM;
end

% Close the diary file to stop recording the command window output
diary off;


