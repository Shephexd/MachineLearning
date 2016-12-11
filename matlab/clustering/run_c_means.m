load('irisdata.mat')
N = size(data,2);
m = 3;
rand_idx = randperm(N);
theta_new = zeros(m,1);

q = 5;


theta_new(1) = mean(mean(data(:,rand_idx(1:25))));
theta_new(2) = mean(mean(data(:,rand_idx(26:50))));
theta_new(3) = mean(mean(data(:,rand_idx(51:75))));


theta_old = zeros(m,1);

cost = zeros(m,1);
k_class = zeros(N,1);

while not(theta_old(1) == theta_new(1) & theta_old(2) == theta_new(2) & theta_old(3) == theta_new(3))
    u = get_memberships(data,theta_new,q,m);

    for i = 1:N
        for j = 1:m
            cost(j) = u(i,j)^q*sqrt(sum(data(:,i) - theta_new(j)).^2);
        end

        [a,idx] = min(cost);
        k_class(i) = idx;

    end

    a = sprintf('class1 = %d, class 2= %d, class3=%d',sum(k_class==1),sum(k_class==2),sum(k_class==3));
    disp(a);    
    disp(theta_new);
    theta_old = theta_new;

   
    for j = 1:m
        
        for i = 1:N
            up_eq = mean(data(:,i))*u(i,j)^q;
        end
        
        theta_new(j) = sum(up_eq)/sum(u(:,j).^q);
    end
    break;
end

subplot(2,1,1);
hold on
scatter3(data(1,k_class==1),data(2,k_class==1),data(3,k_class==1),'rx');
scatter3(data(1,k_class==2),data(2,k_class==2),data(3,k_class==2),'gx');
scatter3(data(1,k_class==3),data(2,k_class==3),data(3,k_class==3),'bx');

subplot(2,1,2);
plot(dataclass'-k_class,'x');
