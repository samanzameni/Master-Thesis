%%% clear everything
%clear all
%close all
%clc
%
%%% initialize the problem variables
%%% the region that is serviced by a GDB is divided into m * n square cells,
%m=300;
%n=300; 
%max_v=m/300;%%<1/100 m maximum x velocity
%max_p=100; %% maximum power an SU can use
%
%%% initiate the problem: 
%%%generate real pu locations 
%PUnum=1;
%PUs=zeros(2,PUnum);
%for i=1:PUnum
%  PUs(1,i)= complex(fix(rand*m), fix(rand*n));
%  %PUs(1,i)= complex(0,0);
%  PUs(2,i)= complex(rand*max_v*(-1)^(fix(rand*2)) , rand*max_v*(-1)^(fix(rand*2)));
%end
%
%%%initial the filtering parameters
%T=1; %% time difference between two queries
%N = 2000; % >10% totallparticles  The number of particles the system generates. 
%%nthreshold= 20;
%
%%the functions used are:
%% x = F*x+ PROCESS NOISE --> sqrt(x_N)*randn
%% z = h(x)
%F=[1 T; 0 1];
%
%%%noise
%sigma=complex(0.01, 0.01); %%noise intensity
%sys_noise_cov = [1/3*T^3, 1/2*T^2; 1/2*T^2, T]*sigma; % Noise covariance in the system 
%particles=zeros(2,N);
%
%for i=1:N
%  particles(1,i)=complex(fix(rand*m), fix(rand*n)); 
%  particles(2,i)=complex(rand*max_v*((-1)^(fix(rand*2))) , rand*max_v*((-1)^(fix(rand*2))));
%end
%
%
%
%indice=[1:N]; %%chosen indice for real weight
%
%
%for k=1:70
%    %plot(particles(1,:),'.b',PUs(1,:),'.g')
%    %title(i)
%    %pause
%    
%    %Here, we do the particle filter
%    weights=zeros(1,N)+1/N;
%    
%    %% generate random location for SU
%    loc= complex(fix(rand*m), fix(rand*n));
%    
%    
%    %% the PUs will move:  
%    for j=1:PUnum
%     PUs(:,j) = F*PUs(:,j)+sqrt(sys_noise_cov)*[complex(randn, randn); 0];      
%    end
%    
%    %%get the measurment
%    p = DBreq(loc, PUs, max_p);
%    
%    
%    %%these are needed parameters initialized
%    particles_update=zeros(2,N);  
%    count=0;
%    
%    for j = 1:N
%        %% Predict Phase: how the particles moved!         
%        particles_update(:,j) = F*particles(:,j)+ sqrt(sys_noise_cov)*[complex(randn, randn); complex(randn, randn);]; %%AAAASSSSKKKKK!   
%        %with these new updated particle locations, update the observations
%        %for each of these particles.
%        z(j) = h(abs(particles_update(1,j)-loc), max_p);
%       
%        if p == z(j) %% counting the number of particles in the same region that we know atleast a PU should exist there
%          count++;
%        end           
%    end    
%    
%    %% updating the weight
%    for j = 1:N    
%      if (z(j)==p) %% this particle might be a true locat
%        weights(j)= 1/count;
%      elseif (z(j) < p) %% we know this particle does not exist! 
%        weights(j)=0;
%      end
%      %% otherwise we dont know anything
%    end
%    
%  % Normalize to form a probability distribution (i.e. sum to 1).
%    weights = weights./sum(weights);
%    weights_resample_update=zeros(1,n);
%  
%  %% Resampling: From this new distribution, now we randomly sample from it to generate our new estimate particles
%    for j = 1 : N
%        indice(j)=find(rand <= cumsum(weights),1);
%        particles(:,j) = particles_update(:, indice(j));
%        weights_resample_update(j)=weights(indice(j));    
%    end   
%    
%    weights=weights_resample_update;
%    
%    %figure(k)
%    %plot(particles,'.b',PUs(1,:),'.g', loc , '.r')
%    
%    
%    %neff=1/sum(weights.*weights)
%    
%    %%if(ceil(neff)<nthreshold)
%     %%break;
%    %%end        
%end
%
%x=1
%
%%for 
%%[val, sortindx]=sort(weights);
%%val(N:N)
%
%%plot(particles(1,sortindx(N:N)),'.b',PUs(1,:),'.g')
%uniqueParticles=unique(fix(particles(1,:)));
%
%finalParticlesWeights=zeros(size(uniqueParticles));
%
%finalParticleNum=size(uniqueParticles,2);
%
%for i=1:finalParticleNum
%  finalParticlesWeights(i)=sum(uniqueParticles(i)==fix(particles(1,:)));
%end
%
%[val, sortindx]=sort(finalParticlesWeights);
%
%%sortindx
%%uniqueParticles(sortindx)
%
%figure(1)
%plot(uniqueParticles(sortindx(finalParticleNum-2:finalParticleNum)),'.b',PUs(1,:),'.g')
%figure(2)
%plot(particles,'.b',PUs(1,:),'.g')
%
%title('estimate:')
%PUs(1,:)
%
%



