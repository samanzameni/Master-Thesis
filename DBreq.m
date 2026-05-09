function [power,channel,time] = DBreq (loc, PUs, maxPower, maxChannel)

d=min(abs(loc-PUs(1,:)));
power=h_continuous(d,maxPower);
 sigma_meas = 5
 noise = sigma_meas * randn();
 power = max(0, power + noise);
channel=calculateChannel(d,maxChannel);
time=calculateTime();

end
