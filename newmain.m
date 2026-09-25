%% clear everything
clc; clear; close all;
 
inputfilename = 'InputData.xlsx';
outputfilename = 'OutputData.xlsx';
 
%% initialize the problem variables
%% the region that is serviced by a GDB is divided into m * n square cells,
m=500;
n=500;
max_v= m/500;%%<1/500 m maximum x velocity
max_p=100; %% maximum power an SU can use
max_ch=10; %% maximum channel an SU can use
 
%% ================== STATE REPRESENTATION ==================
%% state = [x; y; vx; vy]  (4x1 real vector), instead of the
%% previous complex(2,1) = [pos_complex; vel_complex] representation
%% =============================================================
 
%% generate real pu locations
PUnum=1;
PUs = zeros(4,PUnum);   % [x;y;vx;vy] per PU
rmse_count = 0;
 
for i=1:PUnum
    PUs(1,i) = fix(rand*m);                       % x
    PUs(2,i) = fix(rand*n);                       % y
    PUs(3,i) = rand*max_v*(-1)^(fix(rand*2));      % vx
    PUs(4,i) = rand*max_v*(-1)^(fix(rand*2));      % vy
end
 
%%initial the filtering parameters
T=1; %% time difference between two queries
N = 1000;
 
sigma= 0.01; %%noise intensity
 
%% ----- 4x4 state transition matrix -----
F = [1 0 T 0;
     0 1 0 T;
     0 0 1 0;
     0 0 0 1];
 
%% ----- 4x4 process noise covariance -----
sys_noise_cov = sigma * [T^3/3   0       T^2/2   0;
                          0       T^3/3   0       T^2/2;
                          T^2/2   0       T       0;
                          0       T^2/2   0       T];
 
sqrtQ = sqrtm(sys_noise_cov);   % computed once, reused every iteration
 
sigma_meas = 5;
particles = zeros(4,N);   % [x;y;vx;vy] per particle
 
for i=1:N
    particles(1,i) = fix(rand*m);
    particles(2,i) = fix(rand*n);
    particles(3,i) = rand*max_v*((-1)^(fix(rand*2)));
    particles(4,i) = rand*max_v*((-1)^(fix(rand*2)));
end
 
entropy = zeros(1, 1000);
gini = zeros(1, 1000);
Merror = zeros(1, 1000);
entropyMax = zeros(1, 1000);
giniMax = zeros(1, 1000);
MerrorMax = zeros(1, 1000);
distance = zeros(1, 1000);
rmse = zeros(1, 1000);
z = zeros(1, N);
Equality = zeros(1, 1000);
Gquality = zeros(1, 1000);
Mquality = zeros(1, 1000);
 
% Entropy impurity
 Ethreshold=0.001;
%  GINI impurity
 Gthreshold=0.001;
%  Misclassification error impurity
 Mthreshold=0.001;
 
it=1000;
inputMat = zeros(1000, 3);
outputMat = zeros(1000, 2);
 
for k=1:1000
 
    %%% initialization, illastration, evaluation , and end condition %%%
    %% initialization
 
    weights=zeros(1,N)+1/N;
 
    % the PUs will move (single 4x4 noise draw instead of two 2x2 draws):
    for j=1:PUnum
        noise = sqrtQ * randn(4,1);   % [x;y;vx;vy] correlated noise
        PUs(:,j) = F*PUs(:,j) + noise;
        PUs(1,j) = min(max(PUs(1,j), 0), m);   % clip x
        PUs(2,j) = min(max(PUs(2,j), 0), n);   % clip y
    end
 
    % generate random location for SU
    locX = fix(rand*m);
    locY = fix(rand*n);
 
    %% evaluation
    % putting particles with fix equal values together and counting the number of them in order to calculate the
    % real weights of particles which shows the concentration in some point.
 
   l = 10;
 
% bin بندی دو بعدی
Bx = fix(particles(1,:) / l) * l;
By = fix(particles(2,:) / l) * l;
 
% Here we only use a helper complex number for easier grouping;
% This is no longer part of the motion/filter model, just a unique key for x,y.
BParticles = complex(Bx, By);
uniqueBParticles = unique(BParticles);
uBParticleNum = length(uniqueBParticles);
 
P = zeros(1, uBParticleNum);
 
for i = 1:uBParticleNum
    P(i) = sum(BParticles == uniqueBParticles(i));
end
 
r = 0.5;
 
    % calculation of Entropy and Gini and Misclassification Error as a parameter to show the end of algorithm
 
    % Entropy impurity
    P_norm = P / N;
    P_norm(P_norm==0) = eps;
    entropy(1,k) =  -sum(P_norm .* log(P_norm));
    entropyMax(k) = log(uBParticleNum);
    if entropy(1,k) < Ethreshold
        it=k;
    %    break;
    end
 
    %  Gini impurity
 
    gini(1,k) =  sum(P/N.*(1 - P/N));
    giniMax(k) = 1 - 1/uBParticleNum;
    if gini(1,k) < Gthreshold
        it=k;
   %     break;
    end
 
    %  Misclassification error impurity
 
    Merror(1,k) =  (1-max(P/N));
    MerrorMax(k) = 1 - 1/uBParticleNum;
    if Merror(1,k) < Mthreshold
        it=k;
    %    break;
    end
 
    % sorting weights in order to get most concentrated points which are the answers
 
    fixedParticles = complex(fix(particles(1,:)), fix(particles(2,:)));
 
    uniquefixedParticles = unique(fixedParticles);
 
    particleNum = size(uniquefixedParticles,2);
 
    particlesDensity = zeros(size(uniquefixedParticles));
 
    for i=1:particleNum
        particlesDensity(i) = sum(uniquefixedParticles(i)==fixedParticles);
    end
 
    [val, sortindx] = sort(particlesDensity);
 
    idx_start = max(1, particleNum-4);
    PU_pos = complex(PUs(1,1), PUs(2,1));
    distance(1,k) = min(abs(uniquefixedParticles(sortindx(idx_start:particleNum)) - PU_pos));
 
 
    % RMSE: root mean square error averaged over all particles
    rmse_count = rmse_count + 1;
    % Position estimate (mean of particles' x and y)
    estimated_pos = complex(mean(particles(1,:)), mean(particles(2,:)));
    rmse(k) = abs(estimated_pos - PU_pos);
 
    %%%%%%%%%%%%%%Here, we do the particle filter%%%%%%%%%%%%%%%%%
 
 
 
    %% # measurment
    [p,channel,time] = DBreq(locX, locY, PUs, max_p, max_ch, sigma_meas);
    % fprintf('power = %d time = %d channel = %d \n',p,time,channel);
    inputMat(k,1:3) = [locX, locY, p];
    outputMat(k,1:2)= [PUs(1,1), PUs(2,1)];
 
      %% # prediction
    particles_star = zeros(4,N);
    for j = 1:N
        % the particles should move with the PU motion model in each iteration
        noise = sqrtQ * randn(4,1);
 
        particles_star(:,j) = F*particles(:,j) + noise;
        particles_star(1,j) = min(max(particles_star(1,j), 0), m);
        particles_star(2,j) = min(max(particles_star(2,j), 0), n);
 
        % Calculate Euclidean distance between the predicted particle and SU position
        dx = particles_star(1,j) - locX;
        dy = particles_star(2,j) - locY;
        d_particle = sqrt(dx^2 + dy^2);
 
        % Predict power using the continuous function
        z(j) = h_continuous(d_particle, max_p);
    end
 
    %% # updating the weights
    for j = 1:N
        % Calculate weight using Gaussian distribution
        weights(j) = exp(-((z(j) - p)^2) / (2 * sigma_meas^2));
    end
 
    % Prevent all weights from becoming zero (Particle Depletion)
    if sum(weights) == 0
        weights = ones(1,N) / N;
    else
        % Normalize to form a probability distribution (i.e. sum to 1).
        weights = weights ./ sum(weights);
    end
 
    %% # Resampling: From this new distribution, now we randomly sample from it to generate our new estimate particles
    for j = 1 : N
        particles(:,j) = particles_star(:,find(rand <= cumsum(weights),1));
    end
 
 
 %% illastration
    if (k == 1 || mod(k, 100) == 0)
         figure(k)
         str = strcat('number of iteration = ', int2str(k));
         plot(particles(1,:), particles(2,:), '.b')
         hold on;
         plot(PUs(1,:), PUs(2,:), 'rs');
         title(str);
         filename = strcat('number of iteration = ', int2str(k));
         print('-dpng', '-r300', filename);
    end
 
end
 
% % Quality
 
 for k=1:it
     if((entropyMax(1,k))~=0 && max(distance(1:it))~=0)
        Equality(1,k)=((1-(entropy(1,k)/entropyMax(1,k)))*(1-(distance(1,k)/max(distance(1:it)))));
 else
        Equality(1,k)=(entropyMax(1,k))+.001;
    end
 end
 
 for k=1:it
     if((giniMax(1,k))~=0 && max(distance(1:it))~=0)
         Gquality(1,k)=((1-(gini(1,k)/giniMax(1,k)))*(1-(distance(1,k)/max(distance(1:it)))));
     else
         Gquality(1,k)=(giniMax(1,k))+.001;
     end
 end
 
 for k=1:it
     if((MerrorMax(1,k))~=0 && max(distance(1:it))~=0)
         Mquality(1,k)=((1-(Merror(1,k)/MerrorMax(1,k)))*(1-(distance(1,k)/max(distance(1:it)))));
     else
         Mquality(1,k)=(MerrorMax(1,k))+.001;
     end
 end
 
 
status = xlswrite(inputfilename, inputMat);
status = xlswrite(outputfilename, outputMat);
 
save('inputData.mat','inputMat');
outputMat1=outputMat(:,1);
outputMat2=outputMat(:,2);
save('outputData1.mat','outputMat1');
save('outputData2.mat','outputMat2');
 
 
 figure(1001)
 idx_start_fig = max(1, particleNum-5);
 topPts = uniquefixedParticles(sortindx(idx_start_fig:particleNum));
 plot(real(topPts), imag(topPts), '.b')
 hold on;
 plot(PUs(1,:), PUs(2,:), 'rs');title('five highest-density particle');
 
 
 figure(1002)
 plot(particles(1,:), particles(2,:), '.b')
 hold on;
 plot(PUs(1,:), PUs(2,:), 'rs');title('last iteration');
 print('-dpng','-r300','last iteration');
 
 figure(1003)
 plot((1:it),entropy(1:it));title('Entropy impurity');
 axis([0 it 0 max(entropy)+0.5])
 print('-dpng','-r300','Entropy impurity');
 
 figure(1004)
 plot((1:it),gini(1:it));title('Gini Impurity');
 axis([0 it 0 max(gini)+0.5])
 print('-dpng','-r300','Gini Impurity');
 
 figure(1005)
 plot((1:it),Merror(1:it));title('Misclassification Error Impurity');
 axis([0 it 0 max(Merror)+0.5])
 print('-dpng','-r300','Misclassification Error Impurity');
 
 figure(1006)
 plot((1:it),distance(1:it));title('Distance');
 print('-dpng','-r300','Distance');
 
 
 figure(1007)
 plot((1:it),Equality(1:it));title('Entropy Quality');
 
 figure(1008)
 plot((1:it),Equality(1:it),'bo:');title('Entropy Quality with Regression Line');
 h = lsline;
 set(h(1),'color','r')
 print('-dpng','-r300','Entropy Quality with Regression Line');
 
 figure(1009)
 plot((1:it),Gquality(1:it));title('Gini Quality');
 
 figure(1010)
 plot((1:it),Gquality(1:it),'bo:');title('Gini Quality with Regression Line');
 h = lsline;
 set(h(1),'color','r')
 print('-dpng','-r300','Gini Quality with Regression Line');
 
 figure(1011)
 plot((1:it),Mquality(1:it));title('Misclassification Error Quality');
 
 figure(1012)
 plot((1:it),Mquality(1:it),'bo:');title('Misclassification Error Quality with Regression Line');
 h = lsline;
 set(h(1),'color','r')
 print('-dpng','-r300','Misclassification Error Quality with Regression Line');
 
 
figure(1013);
subplot(2, 1, 1);
plot(Equality(:), 'LineWidth', 2);
hold on;
plot(Gquality(:), 'LineWidth', 2);
plot(Mquality(:), 'LineWidth', 2);
xlabel('Iteration');
ylabel('Value');
legend('Entropy', 'Gini', 'Misclassification Error');
title('Comparison of the quality of three impurities');
grid on;
subplot(2, 1, 2);
bar([Equality(:), Gquality(:), Mquality(:)], 'grouped');
xlabel('Iteration');
ylabel('Value');
legend('Entropy', 'Gini', 'Misclassification Error');
title('Comparison of the quality of three impurities');
grid on;
set(gcf, 'Position', [100, 100, 800, 600]);
print('-dpng','-r300','Comparison of the quality of three impurities');
 
figure(1014)
plot((1:rmse_count), rmse(1:rmse_count));
title('RMSE of Particle Filter');
xlabel('Iteration');
ylabel('RMSE');
grid on;
print('-dpng','-r300','RMSE of Particle Filter');
 
%% =========================================================================
%% Calculate Statistical Means and Percentage Differences
disp(' ');
disp('=========================================================================');
disp('           Misclassification Error Percentage Difference (%)             ');
disp('=========================================================================');
 
% Extract the valid parts of the arrays (up to iteration 'it')
E_val = entropy(1:it);
G_val = gini(1:it);
M_val = Merror(1:it);
 
% Prevent zero or negative values for harmonic and geometric means
E_val(E_val <= 0) = eps;
G_val(G_val <= 0) = eps;
M_val(M_val <= 0) = eps;
 
% Calculate Arithmetic Mean
mean_E = mean(E_val);
mean_G = mean(G_val);
mean_M = mean(M_val);
 
% Calculate Harmonic Mean
harm_E = harmmean(E_val);
harm_G = harmmean(G_val);
harm_M = harmmean(M_val);
 
% Calculate Geometric Mean
geo_E = geomean(E_val);
geo_G = geomean(G_val);
geo_M = geomean(M_val);
 
% Calculate Median
med_E = median(E_val);
med_G = median(G_val);
med_M = median(M_val);
 
% Calculate Percentage Differences for Merror relative to Entropy and Gini
% Formula: ((Baseline - Merror) / Baseline) * 100
% Positive values mean Misclassification Error is lower (better)
 
% vs Entropy
diff_E_harm = ((harm_E - harm_M) / harm_E) * 100;
diff_E_mean = ((mean_E - mean_M) / mean_E) * 100;
diff_E_med  = ((med_E - med_M) / med_E) * 100;
diff_E_geo  = ((geo_E - geo_M) / geo_E) * 100;
 
% vs Gini
diff_G_harm = ((harm_G - harm_M) / harm_G) * 100;
diff_G_mean = ((mean_G - mean_M) / mean_G) * 100;
diff_G_med  = ((med_G - med_M) / med_G) * 100;
diff_G_geo  = ((geo_G - geo_M) / geo_G) * 100;
 
% Prepare data for the table
Impurities = {'Relative to Entropy'; 'Relative to Gini'};
Harmonic_Mean = round([diff_E_harm; diff_G_harm], 2);
Arithmetic_Mean = round([diff_E_mean; diff_G_mean], 2);
Median = round([diff_E_med; diff_G_med], 2);
Geometric_Mean = round([diff_E_geo; diff_G_geo], 2);
 
% Create and display the table
ResultTable = table(Impurities, Harmonic_Mean, Arithmetic_Mean, Median, Geometric_Mean, ...
    'VariableNames', {'Types_of_impurities', 'Harmonic_Mean', 'Arithmetic_Mean', 'Median', 'Geometric_Mean'});
 
disp(ResultTable);
disp('=========================================================================');