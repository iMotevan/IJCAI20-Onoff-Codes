clear;
topk = 5;
delta = 3;
load('/Users/imote/Documents/MATLAB/IJCAI/M.mat');
load('/Users/imote/Documents/MATLAB/IJCAI/mt.mat')
M(M < 0) = 0;
M = M(1:100,1:50);
% M = normr(M);
M = M ./ max(M,[],2);
[B,I] = maxk(M,topk,2);
% I = sort(I, 2);

user_num = size(M,1);
movie_num = size(M,2);
% G1 = zeros(user_num,movie_num);
% for i=1:user_num
%     temprow_index = zeros(1,movie_num);
%     temprow_index(I(i,:)) = B(i,:);
%     G1(i,:) = temprow_index;
% end

bundle_num_per_user = 0;
for i=1:delta
    bundle_num_per_user = bundle_num_per_user + nchoosek(topk,i);
end
w_f = zeros(user_num,bundle_num_per_user);
d_f = zeros(user_num,bundle_num_per_user);
b_f = cell(user_num,bundle_num_per_user);
sigma_i = zeros(user_num*bundle_num_per_user, movie_num);

for i=1:user_num
    c_index = 1;
    for j=1:delta
        b_num = nchoosek(topk,j);
        b_f(i,c_index:c_index+b_num-1) = num2cell(nchoosek(I(i,:),j),2)';
        c_index = c_index+b_num;
    end
end
%%Calculate efficiency and diversity and sigma_i
for i=1:user_num
    for j=1:bundle_num_per_user
        movie_list = cell2mat(b_f(i,j));
        w_f(i,j) = sum(M(i,movie_list));
        diversity = zeros(1,18);
        for k = 1:numel(movie_list)
            diversity = diversity | mt(movie_list(k),:);
        end
        d_f(i,j) = sum(diversity);
        sigma_temp = zeros(1,movie_num);
        sigma_temp(movie_list) = 1;
        sigma_i((i-1)*bundle_num_per_user + j,:) = sigma_temp;
    end
end

%%%Normalized w_f and d_f
% w_f = w_f ./ max(w_f,[],2);
% d_f = d_f ./ max(d_f,[],2);
w_f = w_f ./ max(max(w_f));
d_f = d_f ./ max(max(d_f));

T = 500;
load('/Users/imote/Documents/MATLAB/IJCAI/p_j.mat');
%%1 * 1000
p_j = p_j(1:user_num);
% p_j = rand(1,user_num);
p_j = p_j / sum(p_j);
r_j = T * p_j;
c_i = 1 * ones(1,movie_num);
% c_i = randi([5,10],[1,movie_num]);

%%%%Solve the first LP
cvx_begin
    variables x(user_num,bundle_num_per_user);
    maximize(sum(sum(x .* w_f)));
    subject to
        sum(x,2) <= r_j';
        reshape(x',1, []) * sigma_i <= c_i;
        x >= 0;
cvx_end
x1 = x;
opt_1 = cvx_optval;
cvx_clear

%%%%Solve the second LP
cvx_begin
    variables x(user_num,bundle_num_per_user);
    maximize(sum(sum(x .* d_f)));
    subject to
        sum(x,2) <= r_j';
        reshape(x',1, []) * sigma_i <= c_i;
        x >= 0;
cvx_end

x2 = x;
opt_2 = cvx_optval;

% opt = opt_1 + opt_2;
% wd_f = w_f + d_f;

gamma = 1;
alpha_1 = 0.5;
alpha_2 = 0.5;

EXP_TIMES = 100;
SAMPLE_TIMES = 10;

alg_cr = zeros(5, 1);
alg_cr_1 = zeros(5, 1);
alg_cr_2 = zeros(5, 1);
%%%Online phase test
for et = 1:EXP_TIMES
    disp(et);
    j_seq = randsample(user_num,T,true,p_j);
    
    for st = 1:SAMPLE_TIMES
        %%%SAMP
        B_c_i = c_i;
        x_samp = zeros(user_num,bundle_num_per_user);
        %arrival at time t
        for t = 1:numel(j_seq)
            %sample an assignment
            j_id = j_seq(t);
            x1_f = x1(j_id,:);
            x2_f = x2(j_id,:);
            p_sample = gamma * (alpha_1 * x1_f + alpha_2 * x2_f) / r_j(j_id);
            p_sample(p_sample < 0) = 0;
            f_sample = randsample([0,1:bundle_num_per_user],1,true,[1-sum(p_sample),p_sample]);
            if(f_sample == 0)
                continue;
            end
            %safe constraint
            movie_list = cell2mat(b_f(j_id,f_sample));
        %             flag = B_c_i(movie_list);
            if(all(B_c_i(movie_list) >= 1))
                B_c_i(movie_list) = B_c_i(movie_list) - 1;
                x_samp(j_id,f_sample) = x_samp(j_id,f_sample) + 1;
            end
        end
%         alg_cr(1) = sum(sum(wd_f .* x_samp)) / opt + alg_cr(1);
        alg_cr_1(1) = sum(sum(w_f .* x_samp)) / opt_1 + alg_cr_1(1);
        alg_cr_2(1) = sum(sum(d_f .* x_samp)) / opt_2 + alg_cr_2(1);
       
        %%%ATT
        
        
    end
    
    %%%Greedy-Efficiency
    B_c_i = c_i;
    x_greedy_e = zeros(user_num,bundle_num_per_user);
    %arrival at time t
    for t = 1:numel(j_seq)
        j_id = j_seq(t);
        bundle_list = b_f(j_id,:);
        w_f_j = w_f(j_id,:);
        for i=1:numel(w_f_j)
            [v, I] = max(w_f_j);
            movie_list = cell2mat(bundle_list(I));
            if(all(B_c_i(movie_list) >= 1))
                B_c_i(movie_list) = B_c_i(movie_list) - 1;
                x_greedy_e(j_id,I) = x_greedy_e(j_id,I) + 1;
                break;
            end
            w_f_j(I) = -1; %%-1 is more safe than 0
        end
    end
%         alg_cr(3) = sum(sum(wd_f .* x_greedy_e)) / opt + alg_cr(3);
    alg_cr_1(3) = sum(sum(w_f .* x_greedy_e)) / opt_1 + alg_cr_1(3);
    alg_cr_2(3) = sum(sum(d_f .* x_greedy_e)) / opt_2 + alg_cr_2(3);

    %%%Greedy-Diversity
    B_c_i = c_i;
    x_greedy_d = zeros(user_num,bundle_num_per_user);
    %arrival at time t
    for t = 1:numel(j_seq)
        j_id = j_seq(t);
        bundle_list = b_f(j_id,:);
        w_f_j = d_f(j_id,:);
        for i=1:numel(w_f_j)
            [v, I] = max(w_f_j);
            movie_list = cell2mat(bundle_list(I));
            if(all(B_c_i(movie_list) >= 1))
                B_c_i(movie_list) = B_c_i(movie_list) - 1;
                x_greedy_d(j_id,I) = x_greedy_d(j_id,I) + 1;
                break;
            end
            w_f_j(I) = -1; %%-1 is more safe than 0
        end
    end
%         alg_cr(4) = sum(sum(wd_f .* x_greedy_d)) / opt + alg_cr(4);
    alg_cr_1(4) = sum(sum(w_f .* x_greedy_d)) / opt_1 + alg_cr_1(4);
    alg_cr_2(4) = sum(sum(d_f .* x_greedy_d)) / opt_2 + alg_cr_2(4);

%     %%%Greedy-Combine
%     B_c_i = c_i;
%     x_greedy_c = zeros(user_num,bundle_num_per_user);
%     %arrival at time t
%     for t = 1:numel(j_seq)
%         j_id = j_seq(t);
%         bundle_list = b_f(j_id,:);
%         w_f_j = wd_f(j_id,:);
%         for i=1:numel(w_f_j)
%             [v, I] = max(w_f_j);
%             movie_list = cell2mat(bundle_list(I));
%             if(all(B_c_i(movie_list) >= 1))
%                 B_c_i(movie_list) = B_c_i(movie_list) - 1;
%                 x_greedy_c(j_id,I) = x_greedy_c(j_id,I) + 1;
%                 break;
%             end
%             w_f_j(I) = -1; %%-1 is more safe than 0
%         end
%     end
% %         alg_cr(5) = sum(sum(wd_f .* x_greedy_c)) / opt + alg_cr(5);
%     alg_cr_1(5) = sum(sum(w_f .* x_greedy_c)) / opt_1 + alg_cr_1(5);
%     alg_cr_2(5) = sum(sum(d_f .* x_greedy_c)) / opt_2 + alg_cr_2(5);
end

alg_cr_1(1) = alg_cr_1(1) / (EXP_TIMES * SAMPLE_TIMES);
alg_cr_1(3:5) = alg_cr_1(3:5) / EXP_TIMES;

alg_cr_2(1) = alg_cr_2(1) / (EXP_TIMES * SAMPLE_TIMES);
alg_cr_2(3:5) = alg_cr_2(3:5) / EXP_TIMES;


