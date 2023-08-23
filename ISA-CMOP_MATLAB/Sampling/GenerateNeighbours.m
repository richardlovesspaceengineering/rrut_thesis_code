function cluster = GenerateNeighbours(bounds, x, n)
%GENERATENEIGHBOURS Summary of this function goes here
%   x=individual, n=number of neighbours

    cluster =zeros(n,length(bounds));
    cluster(1,:)=x;
    for i=2:n
        cluster(i,:)=PolynomialMutation(bounds, x);
    end
%cluster
end

