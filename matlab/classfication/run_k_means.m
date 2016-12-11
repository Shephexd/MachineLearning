load('iris_data.mat')
k_class = k_means(data);
figure

subplot(2,1,1);
hold on
scatter3(data(1,k_class==1),data(2,k_class==1),data(3,k_class==1),'rx');
scatter3(data(1,k_class==2),data(2,k_class==2),data(3,k_class==2),'gx');
scatter3(data(1,k_class==3),data(2,k_class==3),data(3,k_class==3),'bx');

subplot(2,1,2);
plot(dataclass'-k_class,'x');

