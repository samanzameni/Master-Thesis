function [power,channel,time] = DBreq (loc, PUs, maxPower, maxChannel)

d=min(abs(loc-PUs(1,:)));
power=h(d,maxPower);
channel=calculateChannel(d,maxChannel);
time=calculateTime();

%     printf("distance = %d \n",d);
fprintf('distance = %d \n',d);
end
