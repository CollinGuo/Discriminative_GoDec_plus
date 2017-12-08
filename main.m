% addpath('/home/ksong/Collin/libsvm-3.20/matlab')
% addpath('/home/kaguo/tools/libsvm-3.20/matlab')
clear

load('yaleB_cropped_48_42.mat')
class_num = 38;
train_num = 10;

% load('/home/kaguo/Videos/PIE_Face_Database/PIE_32x32.mat')
% fea = fea';
% labels = gnd';
% class_num = 68;
% train_num = 20;

% load('/home/kaguo/Videos/spatialpyramidfeatures4scene15/spatialpyramidfeatures4scene15.mat')
% fea = featureMat;
% labels = zeros(1,size(labelMat,2));
% for i = 1:size(labelMat,1)
%     idx = find(labelMat(i,:)==1);
%     labels(idx) = i;
% end
% class_num = 15;
% train_num = 100;

% load('../../Videos/spatialpyramidfeatures4caltech101/spatialpyramidfeatures4caltech101.mat')
% m_fea = mean(featureMat,2);
% fea_tmp = bsxfun(@minus,featureMat,m_fea);
% [U,D,V]=svd(fea_tmp,0);
% U = U(:,1:1500);
% featureMat = U'*featureMat;
% fea = featureMat;
% clear featureMat
% labels = zeros(1,size(labelMat,2));
% for i = 1:size(labelMat,1)
%     idx = find(labelMat(i,:)==1);
%     labels(idx) = i;
% end
% class_num = 102;
% train_num = 30;

% load('../../Videos/caltech101Caltech101.mat','fea_pool_pca','labels')
% fea = fea_pool_pca;
% clear fea_pool_pca;
% class_num = 102;
% train_num = 30;


rep_times = 10;

tic


alpha = 1;
eta1 = 1e-3;


Mu = 0.01;
Mu_g = 0.1;

R = 190;

Mu1 = 1;
Eta1 = 100;

Mu1 = 10.^(-5:4);
Eta1 = 10.^(-5:4);

Sigma = 0.01;
gamma = 0.01;



C = 2./Sigma;

total_result0 = zeros(length(Mu),length(Mu_g),length(Sigma),length(Eta1),length(Mu1),length(R),rep_times);
total_result1 = zeros(length(Mu),length(Mu_g),length(Sigma),length(Eta1),length(Mu1),length(R),rep_times);
total_result2 = zeros(length(Mu),length(Mu_g),length(Sigma),length(Eta1),length(Mu1),length(R),rep_times);
total_result3 = zeros(length(Mu),length(Mu_g),length(Sigma),length(Eta1),length(Mu1),length(R),rep_times);
total_result4 = zeros(length(Mu),length(Mu_g),length(Sigma),length(Eta1),length(Mu1),length(R),rep_times);
for t_i = 1:rep_times
        trainSet = cell(1,length(dataset));
        testSet = cell(1,length(dataset));
        trainLabel = cell(length(dataset),1);
        testLabel = cell(length(dataset),1);
    
        for i = 1:length(dataset)
            idx = randperm(size(dataset{i},2));
            trainSet{i} = dataset{i}(:,idx(1:train_num));
            testSet{i} = dataset{i}(:,idx(train_num+1:end));
            trainLabel{i} = datalabel{i}(idx(1:train_num));
            testLabel{i} = datalabel{i}(idx(train_num+1:end));
        end
        trainSet = cell2mat(trainSet);
        testSet = cell2mat(testSet);
        trainLabel = cell2mat(trainLabel);
        trainLabel = trainLabel';
        testLabel = cell2mat(testLabel);
        testLabel = testLabel';
    
        for i = 1:size(trainSet,2)
            trainSet(:,i) = trainSet(:,i)/norm(trainSet(:,i));
        end
        for i = 1:size(testSet,2)
            testSet(:,i) = testSet(:,i)/norm(testSet(:,i));
        end
    
    
    trainLabelMat = zeros(length(trainLabel),class_num);
    for i = 1:length(trainLabel)
        trainLabelMat(i,trainLabel(i)) = 1;
    end
    X = trainSet;
    
    [m,n] = size(X);
    XXt = X*X';
    mean_vc = zeros(m,class_num);
    mean_v = mean(X,2);
    S_w = zeros(m);
    S_b = zeros(m);
    X_s = cell(1,class_num);
    XiXit = cell(1,class_num);
    
    for i = 1:class_num
        V_tmp = X(:,trainLabel==i);
        X_s{i} = V_tmp;
        XiXit{i} = V_tmp*V_tmp';
        mean_vc(:,i) = mean(V_tmp,2);
        diff1 = bsxfun(@minus,V_tmp,mean_vc(:,i));
        S_w = S_w + diff1*diff1';
        diff2 = mean_v - mean_vc(:,i);
        S_b = S_b + size(V_tmp,2)*(diff2*diff2');
    end
    
    [U0,D0,V0]=svd(S_w);
    d0 = diag(D0);
    S_w=S_w+d0(1)*1e-5*eye(size(S_w));
    temp = pinv(S_w)*S_b;
    [U,D]=eig(temp);
    d = real(diag(D));
    U(:,isinf(d)) = [];
    d(isinf(d)) = [];
    [d,idx] = sort(d,'descend');
    lambda1=real(d(1));
    M = lambda1*S_w-S_b;
    clear S_w S_b
    
    
    temp_result0 = zeros(length(Mu),length(Mu_g),length(Sigma),length(Eta1),length(Mu1),length(R));
    temp_result2 = zeros(length(Mu),length(Mu_g),length(Sigma),length(Eta1),length(Mu1),length(R));
    temp_result1 = zeros(length(Mu),length(Mu_g),length(Sigma),length(Eta1),length(Mu1),length(R));
    temp_result3 = zeros(length(Mu),length(Mu_g),length(Sigma),length(Eta1),length(Mu1),length(R));
    temp_result4 = zeros(length(Mu),length(Mu_g),length(Sigma),length(Eta1),length(Mu1),length(R));
    for r_i = 1:length(R)
        r = R(r_i);
        
        mu_result0 = zeros(length(Mu),length(Mu_g),length(Sigma),length(Eta1),length(Mu1));
        mu_result2 = zeros(length(Mu),length(Mu_g),length(Sigma),length(Eta1),length(Mu1));
        mu_result1 = zeros(length(Mu),length(Mu_g),length(Sigma),length(Eta1),length(Mu1));
        mu_result3 = zeros(length(Mu),length(Mu_g),length(Sigma),length(Eta1),length(Mu1));
        mu_result4 = zeros(length(Mu),length(Mu_g),length(Sigma),length(Eta1),length(Mu1));
        for m_i = 1
            mu = Mu(m_i);
            
            mu_g_result0 = zeros(length(Mu_g),length(Sigma),length(Eta1),length(Mu1));
            mu_g_result2 =zeros(length(Mu_g),length(Sigma),length(Eta1),length(Mu1));
            mu_g_result1 = zeros(length(Mu_g),length(Sigma),length(Eta1),length(Mu1));
            mu_g_result3 = zeros(length(Mu_g),length(Sigma),length(Eta1),length(Mu1));
            mu_g_result4 = zeros(length(Mu_g),length(Sigma),length(Eta1),length(Mu1));
            %             tic
            for m_i_g = 1:length(Mu_g)
                
                eta = Mu_g(m_i_g);
                M_ = mu*M;
                
                gammaI = gamma*eye(size(M_,1));
                etaXXt = eta*XXt;
                add_part = etaXXt + M_ + gammaI;
%                 clear M_
                
                s_result0 = zeros(length(Sigma),length(Eta1),length(Mu1));
                s_result2 = zeros(length(Sigma),length(Eta1),length(Mu1));
                s_result1 = zeros(length(Sigma),length(Eta1),length(Mu1));
                s_result3 = zeros(length(Sigma),length(Eta1),length(Mu1));
                s_result4 = zeros(length(Sigma),length(Eta1),length(Mu1));
                tic
                for s_i = 1:length(Sigma)
                    
                    c = C(s_i);
                    sigma = Sigma(s_i);
                    I_c = eye(size(X,1))*c;
                    mul_para = c-eta;
                    A_inv_Xis = cell(1,class_num);
                    
                    for i = 1:class_num
                        A1 = add_part + mul_para*XiXit{i};
                        A_inv_Xis{i} = A1\X_s{i}*c;
                    end
                    [B,Wi_s,iter_out,Iter_in_s,F,Err,B_norm] = D_GoDec_plus(X,X_s,A_inv_Xis,r,sigma,c,class_num,eta,gamma,M_,1e-3,1e-4);
%                     [B,Wi_s,iter_out,Iter_in_s,F,Err,B_norm] = D_Borth_Lgraph_corr_express_C2_new2(X,X_s,A_inv_Xis,r,sigma,c,class_num,eta,gamma,M_);
                    W = cell2mat(Wi_s);
                    H_test = W'*testSet;
                    WitX_s = cell(class_num,size(testSet,2));
                    coef_term = zeros(class_num,size(testSet,2));
                    fisher_term = zeros(class_num,size(testSet,2));
                    fisher_term1 = zeros(class_num,size(testSet,2));
                    for i = 1:class_num
                        WitX = Wi_s{i}'*testSet;
                        WitX_s{i} = WitX;
                        WitX_sq = WitX.*WitX;
                        coef_term(i,:) = sum(WitX_sq,1);
                        fisher = abs(bsxfun(@minus,H_test,W'*mean_vc(:,i)));
                        fisher_term(i,:) = sum(fisher,1);
                        fisher1 = abs(bsxfun(@minus,WitX,Wi_s{i}'*mean_vc(:,i)));
                        fisher_term1(i,:) = sum(fisher1,1);
                    end
                    coef_sum = sum(coef_term,1);
                    coef_term1 = bsxfun(@minus,coef_sum,coef_term);
                    
                    corr = zeros(class_num,size(testSet,2));
                    for i = 1:class_num
                        temp = testSet - B*WitX_s{i};
                        temp = temp.*temp;
                        temp = exp(-temp/sigma);
                        corr(i,:) = sum(temp,1);
                    end
                    for e_i = 1:length(Eta1)
                        eta1 = Eta1(e_i);
                        for mu1_i = 1:length(Mu1)
                            mu1 = Mu1(mu1_i);
                            error = eta1*coef_term1 - corr;
                            fisher_error = error + mu1*fisher_term;
                            fisher_error1 = error + mu1*fisher_term1;
                            [~,result_label] = max(corr,[],1);
                            result0 = sum(result_label==testLabel)/length(testLabel);
                            s_result0(s_i,e_i,mu1_i) = result0;
                            
                            [~,result_label] = min(error,[],1);
                            result1 = sum(result_label==testLabel)/length(testLabel);
                            s_result1(s_i,e_i,mu1_i) = result1;
                            
                            [~,result_label] = min(fisher_error,[],1);
                            result2 = sum(result_label==testLabel)/length(testLabel);
                            s_result2(s_i,e_i,mu1_i) = result2;
                            
                            [~,result_label] = min(fisher_error1,[],1);
                            result3 = sum(result_label==testLabel)/length(testLabel);
                            s_result3(s_i,e_i,mu1_i) = result3;
                        end
                    end
                end
                toc
                
                mu_g_result0(m_i_g,:,:,:) = s_result0;
                mu_g_result2(m_i_g,:,:,:) = s_result2;
                mu_g_result1(m_i_g,:,:,:) = s_result1;
                mu_g_result3(m_i_g,:,:,:) = s_result3;
                mu_g_result4(m_i_g,:,:,:) = s_result4;
            end
            
            mu_result0(m_i,:,:,:,:) = mu_g_result0;
            mu_result2(m_i,:,:,:,:) = mu_g_result2;
            mu_result1(m_i,:,:,:,:) = mu_g_result1;
            mu_result3(m_i,:,:,:,:) = mu_g_result3;
            mu_result4(m_i,:,:,:,:) = mu_g_result4;
            
        end

        %         temp_result(:,:,:,:,:,r_i) = mu_result;
        temp_result0(:,:,:,:,:,r_i) = mu_result0;
        temp_result2(:,:,:,:,:,r_i) = mu_result2;
        temp_result1(:,:,:,:,:,r_i) = mu_result1;
        temp_result3(:,:,:,:,:,r_i) = mu_result3;
        temp_result4(:,:,:,:,:,r_i) = mu_result4;
    end
    %     total_result(:,:,:,:,:,:,t_i) = temp_result;
    total_result0(:,:,:,:,:,:,t_i) = temp_result0;
    total_result1(:,:,:,:,:,:,t_i) = temp_result1;
    total_result2(:,:,:,:,:,:,t_i) = temp_result2;
    total_result3(:,:,:,:,:,:,t_i) = temp_result3;
    total_result4(:,:,:,:,:,:,t_i) = temp_result4;
end
toc
final_result = mean(total_result1,7)
std(total_result1,0,7)
