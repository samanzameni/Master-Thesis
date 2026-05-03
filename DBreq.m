function [power,channel,time] = DBreq (loc, PUs, maxPower, maxChannel)

d=min(abs(loc-PUs(1,:)));
% power=h(d,maxPower);
% استفاده از همان فرمول پیوسته افت مسیر برای تولید سیگنال واقعی محیط
gamma = 30; % این ضریب باید دقیقاً همان عددی باشد که در تابع h_continuous نوشتید
power = maxPower * exp(-d / gamma); 

% اگر می‌خواهید محیط واقعی‌تر باشد، می‌توانید کمی نویز هم به اندازه‌گیری اضافه کنید:
 % sigma_meas = 5
 noise = 5 * randn();
 power = power + noise;
channel=calculateChannel(d,maxChannel);
time=calculateTime();


end
