function [power,channel,time] = DBreq (loc, PUs, maxPower, maxChannel, sigma_meas)

d=min(abs(loc-PUs(1,:)));
power=h_continuous(d,maxPower);
 noise = sigma_meas * randn();
 power = max(0, power + noise);
channel=calculateChannel(d,maxChannel);
time=calculateTime();

end
