function z = h_continuous(d , max_p)
    gamma = 30; % ضریب افت
    d = abs(d);
    z = max_p * exp(-d / gamma);
end