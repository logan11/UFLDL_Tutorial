function [ cost, grad, pred_prob] = supervised_dnn_cost( theta, ei, data, labels, pred_only)
%SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation
%   Returns cost broken into cross-entropy, weight norm, and prox reg
%        components (ceCost, wCost, pCost)

%% default values
po = false;
if exist('pred_only','var')
  po = pred_only;
end;

%% reshape into network
stack = params2stack(theta, ei);
numHidden = numel(ei.layer_sizes) - 1;
hAct = cell(numHidden+1, 1);
gradStack = cell(numHidden+1, 1);

%% forward prop
for la = 1:numHidden
    if(la == 1)  
        z = stack{la}.W*data;  
    else   
        z = stack{la}.W*hAct{la-1};  
    end  
    z = bsxfun(@plus, z, stack{la}.b);  
    hAct{la}=sigmoid(z);  
end  

hx = (stack{numHidden+1}.W)*hAct{numHidden};  
hx = bsxfun(@plus, hx, stack{numHidden+1}.b);  
hx = exp(hx);  
pred_prob = bsxfun(@rdivide, hx, sum(hx, 1));
hAct{numHidden+1} = pred_prob;  

%% return here if only predictions desired.
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];  
  return;
end;

%% compute cost
f = log(pred_prob);  
I = sub2ind(size(f), labels', 1:size(f,2));
vals = f(I);  
ceCost = -sum(vals);  

%% compute gradients using backpropagation    
dp = zeros(size(pred_prob));  
dp(I) = 1;  
err = (pred_prob-dp); 
  
for la = numHidden+1: -1 : 1   
    gradStack{la}.b = sum(err, 2);  
    if(la == 1)  
        gradStack{la}.W = err*data';  
        break;
    else   
        gradStack{la}.W = err*hAct{la-1}';  
    end  
    err = (stack{la}.W)'*err.*hAct{la-1}.*(1-hAct{la-1});
end  

%% compute weight penalty cost and gradient for non-bias terms
wCost = 0;  
for la = 1:numHidden+1  
    wCost = wCost+.5*ei.lambda*sum(stack{la}.W(:).^2);
end  
  
cost = ceCost + wCost;  
  
for la = numHidden:-1:1  
    gradStack{la}.W = gradStack{la}.W+ei.lambda*stack{la}.W;
end  

%% reshape gradients into vector
[grad] = stack2params(gradStack);
end