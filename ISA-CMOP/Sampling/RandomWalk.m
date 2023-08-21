function walk = RandomWalk(bounds, sNumber, sSize)
% <algorithm> <U>
% RandomWalk algorithm
%dimension=bounds.length. sNumbers = 1000. sSize=10%

    d = length(bounds); %#dimensions
    walk =zeros(sNumber,d); %array of ds to store the walk
    x = zeros(1,d); 
    
    for i = 1:d %walk#0
        x(1,i) = bounds(1,i)+(bounds(2,i)-bounds(1,i))*rand(1,1);
    end
    walk(1,:) = x;
    
    for j=2:sNumber %rest of the walk
        curr=walk(j-1,:);
        i=1;
        while i<=d 
            r = (bounds(1,i)*sSize)+((bounds(2,i)*sSize)-(bounds(1,i)*sSize))*rand(1,1);
            temp = curr(1,i)+r;
            if (temp<=bounds(2,i) && temp>=bounds(1,i))
                s=temp;
            else
                s=curr(1,i)-r;
            end
                x(1,i)=s;
                i=i+1;
        end
        walk(j,:)=x;
    end
                
end

