function individual = PolynomialMutation(bounds, x)
%GENERATE neighbor by using PM
disI = 20; %distribution index (usually 20)
individual = x;
for i=1:length(bounds)
    r=rand(1);
    dx = bounds(2,i) - bounds(1,i);
    if r <0.5
        bl = (individual(1,i)-bounds(1,i))/dx;
        b = (2*r)+(1-2*r) * (1-bl)^(disI+1);
        delta(i)= b^(1/(disI+1)) -1;
    else
        bu = (bounds(2,i)-individual(1,i))/dx;
        b = 2*(1-r)+2*(r-0.5) * (1-bu)^(disI+1);
        delta(i)= 1-(b^(1/(disI+1)));
    end
    individual(1,i)= individual(1,i)+delta(i)*dx;
    
    if (individual(1,i)<bounds(1,i))
        individual(1,i)=bounds(1,i);
    else
        if (individual(1,i)>bounds(2,i))
        individual(1,i)=bounds(2,i);
        end
    end
    
end

