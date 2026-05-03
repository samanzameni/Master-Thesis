%% تابع اندازه‌گیری پیوسته
function z = h_continuous(d)
    Pt = 100; % حداکثر توان
    gamma = 30; % ضریب افت
    z = Pt * exp(-d / gamma); 
end