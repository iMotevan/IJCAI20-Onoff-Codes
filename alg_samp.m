function cr = alg_samp(x_1,x_2,B_c_i,j_seq,alpha_1,alpha_2)

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
alg_cr = sum(sum(w_f .* x_samp + d_f .* x_samp)) / opt;


