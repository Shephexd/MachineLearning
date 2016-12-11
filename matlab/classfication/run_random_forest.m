load('irisdata.mat')
N = 100;
M = size(data,2);

m = round(2*M/3);
for n=1:N
    rand_idx = randperm(M);
    subdata = traindata(:,rand_idx(1:m));
    testdata = traindata(:,rand_idx(m+1:M));
    subclass = dataclass(:,rand_idx(1:m));
    tt = classregtree(subdata',subclass');
    ssfit = eval(tt,testdata');
    testtt{n} = tt;
end
k = size(testdata,2);

fitset = zeros(n,M);
for n=1:N
   fitset(n,:) = eval(testtt{n},data');
   chclass = fitset
    check(1,:) = sum(chclass==1)
    check(2,:) = sum(chclass==2)
    check(3,:) = sum(chclass==3)
    [M,I] = max(check)
    

end

plot(I-dataclass,'x');
