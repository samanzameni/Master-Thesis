function [power] = h(d, maxPower)
    if(d<20)
      power=0;
    elseif (d<40)
      power=40;
    else 
      power=100;
    end
    
end
