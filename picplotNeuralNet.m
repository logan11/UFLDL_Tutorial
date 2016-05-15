close all

n = 1; 
[data_train, labels_train, data_test, labels_test] = load_preprocess_mnist();

pic = zeros(28,28,10000);
for z = 1:10000;
    pic(:,:,z) = reshape(data_test(:,z), [28,28]);
end

[~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_test, [], true);
[~,pred] = max(pred);
% pred = pred+1;
pred(2,:) = labels_test';
bw = pred(2,:) - pred(1,:);
bw = bw ~=0;

figure;
r = 1;
for z = 1:10000
    if r < 26
    if bw(z) == 1
        subplot(5,5,r)
        imshow(pic(:,:,z));
        %-1 to renormalize to 0 - 9 rather than indexed
        title(strcat('GT:',num2str(labels_test(z)-1),',CL:',num2str(pred(1,z)-1),',Im:',num2str(z)));
        r = r+1;
    end
    end
end