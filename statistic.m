clear;clc;
tic;

%% The results statistial part of main.m
load Features_aal;
load group_test;
load IDX_FS;
load index;
times = 100;
pv = [0.001,0.005,0.01,0.025:0.025:0.2];
Matrix = zeros(90,90,69);     %aal
% Matrix = zeros(246,246,69); %BN
label = [zeros(30,1);ones(39,1)];

AC = zeros(1,1000); 
SE = zeros(1,1000);  
SP = zeros(1,1000);
DS = zeros(1,1000);
dev = [];

num = zeros(1,size(Features,1));
num2 = 0;
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
        num2 = num2 + length(IDX_FS{i,X,Y,m});
        
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
names(1:3) = [];
%names(1:119) = [];
fid = fopen( 'edge_aal_1.txt' , 'wb' );
for i = 1 : size( X , 2 )
    fprintf( fid , '%s\r\n' , names{X(1,i)}(5:end)  ); % aal: 5:end
%     fprintf( fid , '%s\r\n' , names{X(1,i)}(7:end)  ); % BN: 7:end
end
fclose( fid );

fid = fopen( 'edge_aal_2.txt' , 'wb' );
for i = 1 : size( X , 2 )
    fprintf( fid , '%s\r\n' , names{X(2,i)}(5:end) );
%     fprintf( fid , '%s\r\n' , names{X(2,i)}(7:end)  );
end
fclose( fid );

%% Weights Computing using all subjects
IDX = find(num2>=500);
group = [ zeros( 1 , 30 ) , ones( 1 , 39 ) ]';
IDX2 = randperm(69);
Features = Features(IDX,IDX2);
group = group(IDX2);

Features = Features';
SVM = fitcsvm( Features , group );
W = sum( repmat( SVM.Alpha , 1 , length(IDX) ) .* SVM.SupportVectors .* repmat( group(SVM.IsSupportVector) , 1 , length(IDX) ) ); 

W = abs(W) / sum( abs(W) );
save weights W;

%% Two sample T test
for i = 1 : size( Features , 2 )
    [~,P(i),~,State] = ttest2( Features(group==0,i) , Features(group==1,i) );
    T(i) = State.tstat;
end
save T_test T P;

%% Roc
group = [];
for m = 1 : times
    for i = 1 : 10   
        group = [group;group_test{i,m}'];    
    end
end

% The ROC curve is sometimes reversed, i.e., the curve is under the line 
% from (0,0) to (1,1). Thus, soft-output of SVM should be fixed. The reason
% for this problem is currently unknown. Maybe due to the lable of the 
% first sample in training group.

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