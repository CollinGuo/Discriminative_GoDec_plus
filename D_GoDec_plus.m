function [B,Wi_s,iter_out,Iter_in_s,F,Err,B_norm] = D_GoDec_plus(X,X_s,A_inv_Xis,r,sigma,c,class_num,eta,gamma,M,epsilon1,epsilon2)
[m,n] = size(X);

B = randn(m,r);
B = orth(B);

Wi_s = cell(1,class_num);
Ei_s = cell(1,class_num);
XitWi_s = cell(1,class_num);
Xi_hat_s = cell(1,class_num);
for i = 1:class_num
    Wi_s{i} = randn(m,r);
    Ei_s{i} = zeros(size(X_s{i}));
    Xi_hat_s{i} = X_s{i} - Ei_s{i}/c;
end
Wi_s0 = Wi_s;
iter_out = 0;
iter_in_max = 10;
iter_out_max = 100;
mul_para = (2/sigma);

Iter_in_s = [];
F = [];
Err = [];
B_norm = [];
f = 0;
for i = 1:class_num
    Wi = Wi_s{i};
    Xi = X_s{i};
    XitWi_s{i} = Xi'*Wi;
    f = f + eta*(norm(Wi'*X,'fro')^2 - norm(XitWi_s{i})^2);
    Xi = X_s{i};
    Ti = Xi - B*(XitWi_s{i}');
    Ti_sq = Ti.*Ti;
    f = f + sum(sum(1-exp(-Ti_sq/sigma)));
    f = f + gamma*norm(Wi)^2;
    f = f + trace(Wi'*M*Wi);
end
F = [F,f];
f0 = f;
while 1
    iter_in = 1;
    while 1
        for i = 1:class_num
            Xi = X_s{i};
            Wi = A_inv_Xis{i}*(Xi_hat_s{i}'*B);
            XitWi_s{i} = Xi'*Wi;

            Wi_s{i} = Wi;
            
            Ti = Xi - B*(XitWi_s{i}');
            Ti_sq = Ti.*Ti;
            Ei = Ti.*(c-mul_para*exp(-Ti_sq/sigma));
            Ei_s{i} = Ei;
            Xi_hat_s{i} = Xi - Ei_s{i}/c;
        end
        e = GetStopCri_1(Wi_s,Wi_s0);
        if(e<epsilon1 || iter_in >iter_in_max)
            break;
        end
        iter_in = iter_in + 1;
        Wi_s0 = Wi_s;
    end
        Z = zeros(m,r);
        for i = 1:class_num
            Z = Z + Xi_hat_s{i}*(XitWi_s{i});
        end

        B = Z*(Z'*Z)^(-0.5);
        if mod(iter_out,20) ==0
            
            f0 = 0;
            for i = 1:class_num
                Wi = Wi_s{i};
                f0 = f0 + eta*(norm(Wi'*X,'fro')^2 - norm(XitWi_s{i})^2);
                Xi = X_s{i};
                Ti = Xi - B*(XitWi_s{i}');
                Ti_sq = Ti.*Ti;
                f0 = f0 + sum(sum(1-exp(-Ti_sq/sigma)));
                f0 = f0 + gamma*norm(Wi)^2;
                f0 = f0 + trace(Wi'*M*Wi);
            end
        end
         if mod(iter_out,20) ==1
             
             f = 0;
             for i = 1:class_num
                 Wi = Wi_s{i};
                 f = f + eta*(norm(Wi'*X,'fro')^2 - norm(XitWi_s{i})^2);
                 Xi = X_s{i};
                 Ti = Xi - B*(XitWi_s{i}');
                 Ti_sq = Ti.*Ti;
                 f = f + sum(sum(1-exp(-Ti_sq/sigma)));
                 f = f + gamma*norm(Wi)^2;
                 f = f + trace(Wi'*M*Wi);
             end
             if(abs((f-f0)/f0)<epsilon2 || iter_out>iter_out_max)
                 break;
             end
         end
    
    iter_out = iter_out + 1;

    f0 = f;
end

function e = GetStopCri_1(W,W0)
e = 0;
for i = 1:length(W)
    tmp = W{i} - W0{i};
    e_Wi = norm(tmp(:))/norm(W0{i}(:));
    e = max([e,e_Wi]);
end

