clear;clc;
tic;

%% The results statistial part of main.m
load Features_aal;
load group_test;
load IDX_FS;
load('index.mat','ACC');
times = 100;
Matrix = zeros(90,90,69);     %aal
% Matrix = zeros(246,246,69); %BN
label = [zeros(30,1);ones(39,1)]; % normal label

ACC_permu = zeros(1000,1);  % 1000: 1000 times permutation
SPEC_permu = zeros(1000,1);
SEN_permu = zeros(1000,1);
DSC_permu = zeros(1000,1);
AUC_permu = zeros(1000,1);

ACC_tmp = zeros(10,times);  
SPEC_tmp = zeros(10,times);
SEN_tmp = zeros(10,times);
DSC_tmp = zeros(10,times);
DEV_tmp = cell([10,times]);

ACCC = zeros(20,1);
SPEC = zeros(20,1);
SEN = zeros(20,1);
DSC = zeros(20,1);
DEV = cell([20,1]);

group = [];
for m = 1 : times
    for i = 1 : 10   
        group = [group;group_test{i,m}'];    
    end
end

% for i = 1 : 750
%     randperm(69);  % Update random seed for runing on different matlab
%                    % e.g., if you want to run 1000 times permutation on 4
%                    % matlab, you can open 4 matlab running this code, each 
%                    % matlab running 250 times, and finally combine the 
%                    % results. But for each matlab, this randperm should
%                    % run 0,250,500,750, respectivly, to updata the seed,
%                    % or the results will be same, since the random in
%                    % matlab is pseudorandom. 
% end

for l = 1 : 1000    
    fprintf('%d\n',l);
    label_permu = label(randperm(69)); % permuted label
    dev = [];
    for m = 1 : times
        for i = 1 : 10
            % finding the corresponding features to bulid the classificaiton
            % model in permutation phase 10-folds CV
            Matrix_num = ACC(i,:,:,:,m);
            IDX = find(Matrix_num==max(Matrix_num(:)));  % Selecting the best
            [~,X,Y,Z] = ind2sub(size(Matrix_num),IDX);
            if length(IDX) > 1
                IDXX = find(X==max(X));   % Try to reduce the number of selected features
                X = X(IDXX(1));
                Y = Y(IDXX(1));       
            end       
            %
            IDX = IDX_FS{i,X,Y,m};
            Features_permu = Features(IDX,:);
            group_train = 1 : 69;
            group_train(group_test{i,m}) = [];
            % SVM permutation
            Train = Features_permu(:,group_train)';
            Test = Features_permu(:,group_test{i,m})';
            Train = ( Train - repmat( mean( Train ) , size( Train , 1 ) , 1 ) ) ./ repmat( std( Train ) , size( Train , 1 ) , 1 );
            Test = ( Test - repmat( mean( Train ) , size( Test , 1 ) , 1 ) ) ./ repmat( std( Train ) , size( Test , 1 ) , 1 );
            Train_G = label_permu(group_train);
            Test_G = label_permu(group_test{i,m});
            
            for k = 1 : 20
                SVM = svmtrain( Train_G , Train , sprintf('-c %.1f -t 0 -b 1 -q 1',k*0.1) ); % libsvm svmtrain function
                [TG,~,DEV{k}] = svmpredict( Test_G , Test , SVM , '-b 1 -q 1' ); 

                TP = sum( TG( TG == Test_G ) == 1 );
                TN = sum( TG( TG == Test_G ) == 0 );
                FP = sum( TG( TG ~= Test_G ) == 1 );
                FN = sum( TG( TG ~= Test_G ) == 0 );
                ACCC(k) = ( TP + TN ) / ( TP + TN + FP + FN + 1e-6 );
                SEN(k) = TP / ( TP + FN + 1e-6 );
                SPEC(k) = TN / ( TN + FP + 1e-6 );
                DSC(k) = 2 * TP / ( 2 * TP + FP + FN + 1e-6 );  % avoiding NAN
            end  
            [~,IDX] = sort(ACCC);
            ACC_tmp(i,m) = ACCC(IDX(end));
            SPEC_tmp(i,m) = SPEC(IDX(end));
            SEN_tmp(i,m) = SEN(IDX(end));
            DSC_tmp(i,m) = DSC(IDX(end));
            DEV_tmp{i,m} = DEV{IDX(end)};
            tmp = DEV_tmp{i,m}(:,1);
            if abs(( sum( (tmp>=0.5) == Test_G ) / length(tmp) - ACC_tmp(i,m) )) < 0.0001
                dev = [dev;tmp];
            else
                dev = [dev;1-tmp];
            end
        end
    end
    ACC_permu(l) = mean(ACC_tmp(:));
    SPEC_permu(l) = mean(SPEC_tmp(:));
    SEN_permu(l) = mean(SEN_tmp(:));
    DSC_permu(l) = mean(DSC_tmp(:));
    
    % AUC
    label_AUC = label_permu(group);
    tmp = sort(dev);
    for i = 1 : length(tmp)

        test = dev >= tmp(i);

        TP = sum( test( test == label_AUC ) == 1 );
        TN = sum( test( test == label_AUC ) == 0 );
        FP = sum( test( test ~= label_AUC ) == 1 );
        FN = sum( test( test ~= label_AUC ) == 0 );

        TPR(i) = TP / ( TP + FN );
        FPR(i) = FP / ( FP + TN );

    end

    AA = diff( FPR );
    IDX = find( AA == 0 );
    IDX = [1,IDX];
    for i = 1 : length(IDX) - 1
        AUC_permu(l) = AUC_permu(l) - ( FPR( IDX(i+1) ) - FPR( IDX(i) ) ) * TPR( IDX(i+1) );
    end  

end
save permute_aal ACC_permu SPEC_permu SEN_permu DSC_permu AUC_permu

%% Statistic
load permute_aal;
AUC_permu(AUC_permu<0.5) = 1 - AUC_permu(AUC_permu<0.5);
P = [length(find(ACC_permu>=0.8490)), length(find(SEN_permu>=0.9238)), ...
     length(find(SPEC_permu>=0.7250)), length(find(DSC_permu>=0.8506)), ... 
     length(find(AUC_permu>=0.8047))];

toc;