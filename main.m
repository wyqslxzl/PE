clear;clc;
tic;

% P.root = 'E:\Data Processing\PEE\HC\results\preprocessing';
% file = spm_select( 'FPList' , P.root , 'ROI\w*001.mat' );
% 
% %% ECT Mask Functional connectivity Matrix
% Matrix = zeros( 90 , 90 , 69 );
% %Matrix = zeros( 246 , 256 , 69 );
% AAL = 4 : 93; % Accroding to CONN software
% BN = 120:365;
% 
% for i = 1 : 69
%     
%     load( file(i,:) , 'data' );
%     time_series_before = zeros( 206 , 90 );
%     num = 0;
%     for j = 1 : 246
%         time_series_before(:,j) = data{AAL(j)}(1:206);
% %         time_series_before(:,j) = data{BN(j)}(1:206);
%     end
% 
%     Matrix(:,:,i) = corr( time_series_before ) - eye(90);
% %     Matrix(:,:,i) = corr( time_series_before ) - eye(246);
% end
      
%% Feature Arrange
% Features = zeros( size(Matrix,1)*(size(Matrix,1)-1)/2 , 69 );
% for k = 1 : 69    
%     tmp = triu( Matrix(:,:,k) );
%     Features(:,k) = tmp(tmp~=0);
% end
%save Features_aal Features;
%save Features_BN246 Features;

% load Features_aal;
% Matrix = zeros(90,90,69);
load Features_BN246;
Matrix = zeros(246,246,69);
group = [ zeros( 1 , 30 ) , ones( 1 , 39 ) ]';

times = 100;
pv = [0.005,0.01,0.025:0.025:0.2];
IDX_CV = cell(0);

ACC = zeros( 10 , 10 , length(pv) , 20 , times );
SEN = ACC; 
SPEC = ACC;
DSC = ACC;
DEV = cell([10 , 10 , length(pv), 20 , times]);
group_test = cell([10 , times]);
IDX_FS = cell([10 , 10 , length(pv), times]);

for m = 1 : times
    fprintf('%d\n',m);
    %% Random Grouping
    while 1
        IDX = randperm( 69 );
        group_train = cell(0);
        for i = 1 : 10
            group_train{i} = IDX;
            if i < 10
                group_test{i,m} = IDX((i-1)*7+1:i*7);  % 69*0.1¡Ö7
                group_train{i}((i-1)*7+1:i*7) = [];
            else
                group_test{i,m} = IDX((i-1)*7+1:end);
                group_train{i}((i-1)*7+1:end) = [];
            end
        end

        num = 0;    
        for i = 1 : 10
            if sum( group(group_test{i,m}) == 1 ) > 0 && sum( group(group_test{i,m}) == 0 ) > 0
                num = num + 1;
            end
        end 
        if num == 10
            break;
        end
    end

    %% Ten-Fold CV
    GTR = cell(0);

    % random grouping in training data
    for i = 1 : 10
        IDX = randperm( length(group_train{i}) );
        for j = 1 : 10
            GTR{i,j} = IDX;
            if j < 10
                GTR{i,j}((j-1)*6+1:j*6) = [];
            else
                GTR{i,j}((j-1)*6+1:end) = [];
            end
        end 
    end

    for i = 1 : 10

        % T-test for original selecion in training data
        Feature_F = Features(:,group_train{i});
        [C,p] = corr( Feature_F' , group(group_train{i}) );

        for l = 1 : length(pv)

            IDX_CV{i,l} = find( p <= pv(l) );
            Feature_CV = Feature_F(IDX_CV{i,l},:);    
            IDXX = cell(0);

            % 10-fold LASSO
            for j = 1 : 10

                Train = Feature_CV(:,GTR{i,j})';
                Train_G = group(group_train{i}(GTR{i,j}));

               
                while 1
                    [b,info] = lasso( ( Train - repmat( mean( Train ) , size( Train , 1 ) , 1 ) ) ./ repmat( std( Train ) , size( Train , 1 ) , 1 )...
                                      , Train_G , 'cv' , 10 );  
                    IDX = find( b( : , info.Index1SE ) ~= 0 );
                    if ~isempty( IDX ) 
                        break;
                    end
                end   

                IDXX{j,l} = IDX;    
            end

            % features occuring times
            num = zeros( 1 , size(Features,1) );
            for j = 1 : 10
                 num(IDX_CV{i,l}(IDXX{j,l})) = num(IDX_CV{i,l}(IDXX{j,l})) + 1;
            end

            for j = 1 : 10

                IDX_FS{i,j,l,m} = find(num>=j);

                % SVM model
                Train = Features(IDX_FS{i,j,l,m},group_train{i})';
                Test = Features(IDX_FS{i,j,l,m},group_test{i,m})';
                Train = ( Train - repmat( mean( Train ) , size( Train , 1 ) , 1 ) ) ./ repmat( std( Train ) , size( Train , 1 ) , 1 );
                Test = ( Test - repmat( mean( Train ) , size( Test , 1 ) , 1 ) ) ./ repmat( std( Train ) , size( Test , 1 ) , 1 );
                Train_G = group(group_train{i});
                Test_G = group(group_test{i,m});

                for k = 1 : 20
                    SVM = svmtrain( Train_G , Train , sprintf('-c %.1f -t 0 -b 1 -q 1',k*0.1) ); % libsvm svmtrain function
                    [TG,~,DEV{i,j,l,k,m}] = svmpredict( Test_G , Test , SVM , '-b 1 -q 1' ); 

                    TP = sum( TG( TG == Test_G ) == 1 );
                    TN = sum( TG( TG == Test_G ) == 0 );
                    FP = sum( TG( TG ~= Test_G ) == 1 );
                    FN = sum( TG( TG ~= Test_G ) == 0 );
                    ACC(i,j,l,k,m) = ( TP + TN ) / ( TP + TN + FP + FN + 1e-6 );
                    SEN(i,j,l,k,m) = TP / ( TP + FN + 1e-6 );
                    SPEC(i,j,l,k,m) = TN / ( TN + FP + 1e-6 );
                    DSC(i,j,l,k,m) = 2 * TP / ( 2 * TP + FP + FN + 1e-6 );  % avoiding NAN
                end
            end       
        end
    end
end
save index ACC SEN SPEC DSC DEV;
save IDX_FS IDX_FS;
save group_test group_test;

%% Results records
AC = zeros(1,1000); 
SE = zeros(1,1000);  
SP = zeros(1,1000);
DS = zeros(1,1000);
DE = [];
num = zeros(1,size(Features,1));
label = [zeros(30,1);ones(39,1)];
for m = 1 : times
    for i = 1 : 10

        Matrix_num = ACC(i,:,:,:,m);
        IDX = find(Matrix_num==max(Matrix_num(:)));  % Selecting the best
        [~,X,Y,Z] = ind2sub(size(Matrix_num),IDX);
        if length(IDX) > 1
            IDXX = find(X==max(X));   % Try to reduce the number of selected features
            IDX = IDX(IDXX(1));
            X = X(IDXX(1));
            Y = Y(IDXX(1));       
        end
        AC((m-1)*10+i) = Matrix_num(IDX);
        Matrix_num = SEN(i,:,:,:,m);
        SE((m-1)*10+i) = Matrix_num(IDX);
        Matrix_num = SPEC(i,:,:,:,m);
        SP((m-1)*10+i) = Matrix_num(IDX);
        Matrix_num = DSC(i,:,:,:,m);
        DS((m-1)*10+i) = Matrix_num(IDX);
        tmp = DEV(i,:,:,:,m);
        tmp = tmp(IDX);
        num(IDX_FS{i,X,Y,m})=num(IDX_FS{i,X,Y,m})+1; 
% The ROC curve is sometimes reversed, i.e., the curve is under the line 
% from (0,0) to (1,1). Thus, soft-output of SVM should be fixed. The reason
% for this problem is currently unknown. Maybe due to the lable of the 
% first sample in training group.
        tmp2 = tmp{1}(:,2);
        if abs(( sum( (tmp2>=0.5) == label(group_test{i,m}) ) / length(tmp2) - AC((m-1)*10+i) )) < 0.0001 
            dev = [dev;tmp2];
        else
            dev = [dev;1-tmp2];
        end
    end
end
M = mean([AC',SE',SP',DS']);
S = std([AC',SE',SP',DS']);
clear X;

%% Feature projection
num2 = num;
num(num2>=500) = 10000;

Matrix_num = ones( size(Matrix,1) , size(Matrix,2) );
Matrix_num = triu( Matrix_num , 1 );
Matrix_num(Matrix_num==1) = num;

IDX = find( Matrix_num == 10000 );
[X(1,:),X(2,:)] = ind2sub( [size(Matrix,1) , size(Matrix,2)] , IDX );

load names;
% names(1:3) = [];
names(1:119) = [];
fid = fopen( 'edge_BN1.txt' , 'wb' );
for i = 1 : size( X , 2 )
%     fprintf( fid , '%s\r\n' , names{X(1,i)}(5:end)  ); % aal: 5:end
    fprintf( fid , '%s\r\n' , names{X(1,i)}(7:end)  ); % BN: 7:end
end
fclose( fid );

fid = fopen( 'edge_BN2.txt' , 'wb' );
for i = 1 : size( X , 2 )
%     fprintf( fid , '%s\r\n' , names{X(2,i)}(5:end) );
    fprintf( fid , '%s\r\n' , names{X(2,i)}(7:end)  );
end
fclose( fid );

%% Weights Computing using all subjects
IDX = find(num2>=500);
group = [ zeros( 1 , 30 ) , ones( 1 , 39 ) ]';
IDX2 = randperm(69);
Features = Features(IDX,IDX2);
group = group(IDX2);

SVM = fitcsvm( Features' , group );
W = sum( repmat( SVM.Alpha , 1 , length(IDX) ) .* SVM.SupportVectors .* repmat( group(SVM.IsSupportVector) , 1 , length(IDX) ) );

W = abs(W) / sum( abs(W) );
save weights W;

%% Roc
group = [];
dev = [];
for m = 1 : times
    for i = 1 : 10   
        group = [group;group_test{i,m}'];    
    end
end

label = label( group );
tmp = sort(dev);
for i = 1 : 69*times
    
    test = dev >= tmp(i);
    
    TP = sum( test( test == label ) == 1 );
    TN = sum( test( test == label ) == 0 );
    FP = sum( test( test ~= label ) == 1 );
    FN = sum( test( test ~= label ) == 0 );
    
    TPR(i) = TP / ( TP + FN );
    FPR(i) = FP / ( FP + TN );
    
end

AA = diff( FPR );
IDX = find( AA == 0 );
IDX = [1,IDX];
AUC = 0;
for i = 1 : length(IDX) - 1
    AUC = AUC - ( FPR( IDX(i+1) ) - FPR( IDX(i) ) ) * TPR( IDX(i+1) );
end

plot( FPR , TPR );

toc;