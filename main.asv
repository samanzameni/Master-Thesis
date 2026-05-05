
%% clear everything
clear all
close all
clc

inputfilename = 'InputData.xlsx';
outputfilename = 'OutputData.xlsx';

%% initialize the problem variables
%% the region that is serviced by a GDB is divided into m * n square cells,
m=500;
n=500;
max_v= m/500;%%<1/100 m maximum x velocity
max_p=100; %% maximum power an SU can use

%% initiate the problem:
%% generate real pu locations
PUnum=1;
PUs=zeros(2,PUnum);
rmse_count = 0;

for i=1:PUnum
    PUs(1,i)= complex(fix(rand*m), fix(rand*n));
    PUs(2,i)= complex(rand*max_v*(-1)^(fix(rand*2)) , rand*max_v*(-1)^(fix(rand*2))); %i->fix(rand*2))
end


%%initial the filtering parameters
T=1; %% time difference between two queries
N = 5000;

sigma=complex(0.01, 0.01); %%noise intensity
sys_noise_cov = [1/3*T^3, 1/2*T^2; 1/2*T^2, T]*sigma; % Noise covariance in the system
F=[1 T; 0 1];



sigma_meas = 5;
particles=zeros(2,N);

for i=1:N
    particles(1,i)=complex(fix(rand*m), fix(rand*n));
    particles(2,i)=complex(rand*max_v*((-1)^(fix(rand*2))) , rand*max_v*((-1)^(fix(rand*2))));
end

entropy=zeros(1,1000);
distance=zeros(1,1000);
% Entropy impurity
 Ethreshold=0.1;
%  GINI impurity
 Gthreshold=0.1;
%  Misclassification error impurity
 Mthreshold=0.1;
it=1000;
for k=1:1000

    %%% initialization, illastration, evaluation , and end condition %%%
    %% initialization
    
    weights=zeros(1,N)+1/N;
    
    % the PUs will move:
    for j=1:PUnum
        PUs(:,j)  = F*PUs(:,j) + sqrtm(sys_noise_cov) * [complex(randn, randn); 0];
        PUs(1,j) = complex(min(max(real(PUs(1,j)), 0), m), min(max(imag(PUs(1,j)), 0), n));
    end
    
    
    % generate random location for SU
    loc = complex(fix(rand*m), fix(rand*n));
    
    %% illastration
    if ( fix((k-1)/5) == (k-1)/5 )
        %% figure(k)
        %%plot(particles,'.b',PUs(1,:),'.r')
    end
    
    %% evaluation
    % putting particles with fix equal values together and counting the number of them in order to calculate the
    % real weights of particles which shows the concentration in some point.
    
    l = 10;
    BParticles = (fix(particles(1,:)./l))*l;
    uniqueBParticles = unique(BParticles);
    P = zeros(size(uniqueBParticles));
    uBParticleNum = size(uniqueBParticles,2);
    
    for i = 1:uBParticleNum
        P(i) = sum( uniqueBParticles(i)==BParticles );
    end
    r=0.5;

    % calculation of Entropy and Gini and Misclassification Error as a parameter to show the end of algorithm

    % Entropy impurity

     entropy(1,k) =  -sum((P/N).*log(P/N));
   %  entropyMax =  -(1/2).*log(1/2)*uBParticleNum;
     entropyMax(1,k) =  -(1/2).*log(1/2)*uBParticleNum;
    if entropy(1,k) < Ethreshold
        it=k;
    %    break;
    end
    
    %  Gini impurity

    gini(1,k) =  sum(P/N.*(1 - P/N));
  %  giniMax =  (1/2.*(1 - 1/2))*uBParticleNum;
    giniMax(1,k) =  (1/2.*(1 - 1/2))*uBParticleNum;
    if gini(1,k) < Gthreshold
        it=k;
   %     break;
    end

    %  Misclassification error impurity

    Merror(1,k) =  (1-max(P/N));
 %   MerrorMax =  (1-(1/2))*uBParticleNum;
    MerrorMax(1,k) =  (1-(1/2))*uBParticleNum;
    if Merror(1,k) < Mthreshold
        it=k;
    %    break;
    end
    
    % sorting weights in order to get most concentrated points which are the answers
    
    
    fixedParticles=(fix(particles(1,:)));
    
    uniquefixedParticles = unique(fixedParticles);
    
    particleNum = size(uniquefixedParticles,2);
    
    particlesDensity = zeros(size(uniquefixedParticles));
    
    for i=1:particleNum
        particlesDensity(i) = sum(uniquefixedParticles(i)==fixedParticles);
    end
    
    [val, sortindx] = sort(particlesDensity);
    
    distance(1,k) = min(abs(uniquefixedParticles(sortindx(particleNum-4:particleNum))-PUs(1,1)));
    
    % RMSE: میانگین خطای ریشه مجذور روی همه‌ی ذرات
    rmse_count = rmse_count + 1;
    rmse(1, rmse_count) = sqrt(mean(abs(particles(1,:) - PUs(1,1)).^2));
    
    %%%%%%%%%%%%%%Here, we do the particle filter%%%%%%%%%%%%%%%%%
    
    
      
    
    %% # measurment
    [p,channel,time] = DBreq(loc, PUs, max_p,10);
    % fprintf('power = %d time = %d channel = %d \n',p,time,channel);
    inputMat(k,1:3) = [real(loc),imag(loc),p];
    outputMat(k,1:2)=[real(PUs(1,:)),imag(PUs(1,:))];
    
      %% # prediction
    particles_star=zeros(2,N);
    
    for j = 1:N
        % the particles should move with the PU motion model in each iteration
        particles_star(:,j) = F*particles(:,j)+ sqrtm(sys_noise_cov)*[complex(randn, randn); complex(randn, randn);];
        particles_star(1,j) = complex(min(max(real(particles_star(1,j)), 0), m), min(max(imag(particles_star(1,j)), 0), n));
        % محاسبه فاصله (چون مختصات مختلط است، abs همان فاصله اقلیدسی را می‌دهد)
        d_particle = abs(particles_star(1,j) - loc);
        
        % پیش‌بینی توان با تابع پیوسته
        z(j) = h_continuous(d_particle);
    end
    
    %% # updating the weights
    for j = 1:N
        % محاسبه وزن با استفاده از توزیع گاوسی
        weights(j) = exp(-((z(j) - p)^2) / (2 * sigma_meas^2));
    end

    % جلوگیری از صفر شدن تمام وزن‌ها (Particle Depletion)
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
    
    
 
        if (k == 1 || mod(k, 100) == 0)
         figure(k)
         str = strcat('number of iteration = ', int2str(k));
         plot(particles,'.b')
         hold on;
         plot(PUs(1,:),'rs');
         title(str);
    end  

end

% % Quality

 for k=1:it
     if((entropyMax(1,k))~=0 && max(distance)~=0)
        Equality(1,k)=((1-(entropy(1,k)/entropyMax(1,k)))*(1-(distance(1,k)/max(distance))));
 else
        Equality(1,k)=(entropyMax(1,k))+.001;
    end
 end

 for k=1:it
     if((giniMax(1,k))~=0 && max(distance)~=0)
         Gquality(1,k)=((1-(gini(1,k)/giniMax(1,k)))*(1-(distance(1,k)/max(distance))));
     else
         Gquality(1,k)=(giniMax(1,k))+.001;
     end
 end

 for k=1:it
     if((MerrorMax(1,k))~=0 && max(distance)~=0)
         Mquality(1,k)=((1-(Merror(1,k)/MerrorMax(1,k)))*(1-(distance(1,k)/max(distance))));
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
 plot(uniquefixedParticles(sortindx(particleNum-5:particleNum)),'.b')
 hold on;
 plot(PUs(1,:),'rs');title('five highest-density particle');
  
 figure(1002)
 plot(particles,'.b')
 hold on;
 plot(PUs(1,:),'rs');title('last iteration');
 
 figure(1003)
 plot((1:it),entropy(1:it));title('Entropy impurity');
 axis([0 it 0 max(entropy)+0.5])
 
 figure(1004)
 plot((1:it),gini(1:it));title('Gini Impurity');
 axis([0 it 0 max(gini)+0.5])
 
 figure(1005)
 plot((1:it),Merror(1:it));title('Misclassification Error Impurity');
 axis([0 it 0 max(Merror)+0.5])
 
 figure(1006)
 plot((1:it),distance(1:it));title('Distance');

 
 figure(1007)
 plot((1:it),Equality(1:it));title('Entropy Quality');
 
 figure(1008)
 plot((1:it),Equality(1:it),'bo:');title('Entropy Quality with Regression Line.');
 h = lsline;
 set(h(1),'color','r')
 
 figure(1009)
 plot((1:it),Gquality(1:it));title('Gini Quality');
 
 figure(1010)
 plot((1:it),Gquality(1:it),'bo:');title('Gini Quality with Regression Line.');
 h = lsline;
 set(h(1),'color','r')
 
 figure(1011)
 plot((1:it),Mquality(1:it));title('Misclassification Error Quality');
 
 figure(1012)
 plot((1:it),Mquality(1:it),'bo:');title('Misclassification Error Quality with Regression Line.');
 h = lsline;
 set(h(1),'color','r')


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

figure(1014)
plot((1:rmse_count), rmse(1:rmse_count));
title('RMSE of Particle Filter');
xlabel('Iteration');
ylabel('RMSE');
grid on;

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

