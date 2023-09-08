function [channel] = calculateChannel(d, maxChannel) 
    if(d<20)
      channel= maxChannel;
    elseif (d<40)
      channel=int32(maxChannel/2);
    else 
      channel=int32(maxChannel/4);
    end
    
end