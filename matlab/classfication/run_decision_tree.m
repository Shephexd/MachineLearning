clear;
load('irisdata.mat')

t = classregtree(traindata',trainclass');
view(t)
sfit = eval(t,data');
mean((sfit-dataclass').^2)
