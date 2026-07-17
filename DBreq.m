function [power,channel,time] = DBreq (locX, locY, PUs, maxPower, maxChannel, sigma_meas)
    dx = locX - PUs(1,:);
    dy = locY - PUs(2,:);
    d = min(sqrt(dx.^2 + dy.^2));
    power=h_continuous(d,maxPower);
    noise = sigma_meas * randn();
    power = max(0, power + noise);
    channel=calculateChannel(d,maxChannel);
    time=calculateTime();
end