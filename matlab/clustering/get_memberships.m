function u = get_memberships(data,theta,q,m)

N = size(data,2);

for i = 1:N
    for j = 1:m
        dist(i,j) = (sqrt(sum(sum((data(:,i) - theta(j)).^2))));
    end
    for j = 1:m
        u(i,j) = 1/(sum((dist(i,j)./dist(i,:)).^(1/(q-1))));
    end    
end
%%
