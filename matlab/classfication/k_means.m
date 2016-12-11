function [ k_class ] = k_means( data )
%KMEANS Summary of this function goes here
%   Detailed explanation goes here

N = size(data,2);
m = 3;
rand_idx = randperm(N);
theta_new = zeros(m,1);


theta_new(1) = mean(mean(data(:,rand_idx(1:25))));
theta_new(2) = mean(mean(data(:,rand_idx(26:50))));
theta_new(3) = mean(mean(data(:,rand_idx(51:75))));

theta_old = zeros(m,1);

%i = 1;
%j = 1;

dist = zeros(m,1);
k_class = zeros(N,1);

while not(theta_old(1) == theta_new(1) & theta_old(2) == theta_new(2) & theta_old(3) == theta_new(3))
    
    for i = 1:N
        for j = 1:m
            dist(j) = sqrt(sum(data(:,i) - theta_new(j)).^2);
        end

        [a,idx] = min(dist);
        k_class(i) = idx;

    end

    a = sprintf('class1 = %d, class 2= %d, class3=%d',sum(k_class==1),sum(k_class==2),sum(k_class==3));
    disp(a);
    theta_old = theta_new;

    for j = 1:m
        theta_new(j) = mean(mean(data(:,k_class==j)));
    end
end

end

