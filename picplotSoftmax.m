close all

n = 1; 

for z = 1:10000;
    n=1;
for j = 1:28
    for k = 1:28
        pic(k,j,z) = test.X(k+n, z);
    end
    n = n+28;
end
end

[~, label] = max(theta'*test.X, [], 1);
label(2,:) = test.y;
bw = label(2,:) - label(1,:);
bw = bw ~=0;

figure;
r = 1;
for z = 1:1000
    if r < 26
    if bw(z) == 1
        subplot(5,5,r)
        imshow(pic(:,:,z));
        %-1 to renormalize to 0 - 9 rather than indexed
        title(strcat('GT:',num2str(test.y(z)-1),',CL:',num2str(label(1,z)-1),',Im:',num2str(z)));
        r = r+1;
    end
    end
end