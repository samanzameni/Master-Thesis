function [power,channel,time] = DBreq (loc, PUs, maxPower, maxChannel)

d=min(abs(loc-PUs(1,:)));
power=h_continuous(d,maxPower);
% اگر می‌خواهید محیط واقعی‌تر باشد، می‌توانید کمی نویز هم به اندازه‌گیری اضافه کنید:
 % sigma_meas = 5
 noise = 5 * randn();
 power = max(0, power + noise);
channel=calculateChannel(d,maxChannel);
time=calculateTime();

end
