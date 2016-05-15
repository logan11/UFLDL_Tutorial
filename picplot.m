close all

n = 1; 

for z = 1:2115;
    n=1;
for j = 1:28
    for k = 1:28
        pic(k,j,z) = train.X(k+n, z);
    end
    n = n+28;
end
end

figure
imshow(pic(:,:,2))

cl = sigmoid(theta'*train.X);
cl = cl > 0.5;
inc = test.y ~= cl;
sum(inc)
[na, sz] = size(inc);

figure;
r = 1;
for z = 1:sz
    if inc(z) == 1
        subplot(5,5,r)
        imshow(pic(:,:,z));
        title(strcat('GT:',num2str(test.y(z)),',Cl:',num2str(cl(z)),',Im:',num2str(z)));
        r = r+1;
    end
end