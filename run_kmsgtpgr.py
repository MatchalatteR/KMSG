import numpy as np
import tensorflow as tf
import pandas as pd
import datetime
import os

from model.KmsgSharedTree import *

np.random.seed(4)
tf.set_random_seed(4)

cur_dir = os.path.dirname(os.path.realpath(__file__)).replace("\\", "/")
dataset_path = cur_dir + '/dataset/ml-1m/'
ratings_path =           'ratings.dat'
kg_dataset_path =        'triplet.dat'
movie_path = 'movies.dat'

batch_size      = 16
max_seq_length  = 8
hidden_size     = 64
feature_size    = 64
learning_rate   = 1e-4
discount_factor = 0.8
gnn_layer       = 1
epsilon         = 3
global_topk     = 10
kg_topK         = 10

kg = pd.read_table(dataset_path + kg_dataset_path, sep='\t', names=['h', 'r', 't'])
interactive_data = pd.read_table(dataset_path + ratings_path, sep='::', names=['u_id', 'i_id', 'rating', 'timestep'])
if ratings_path != 'netflix_1_64.dat':
    movie = pd.read_table(dataset_path + movie_path, sep='::', names=['i_id', 'title', 'genre'])

user_ids = list(interactive_data['u_id'].unique())

np.random.shuffle(user_ids)
training_count = int(len(user_ids) * 0.8)
train_id = user_ids[:training_count]
test_id = user_ids[training_count:]

movie_ids = []
for idx1 in user_ids:
    user_record = interactive_data[interactive_data['u_id'] == idx1]
    for idx2, row in user_record.iterrows():
        if row['i_id'] in movie_ids:
            idx = movie_ids.index(row['i_id'])
        else:
            movie_ids.append(row['i_id'])
            
user_count = len(user_ids)
item_count = len(movie_ids)
print('Total user  count:', len(user_ids))
print('Total movie count:', item_count)

if ratings_path != 'netflix_1_64.dat':
    y_min = 1900
    y_max = 2100
    movie_years = {}
    movie_genre = {}
    cnt_genre = 0
    genres = []
    for _, row in movie.iterrows():
        if row['i_id'] in movie_ids:
            idx = movie_ids.index(row['i_id'])
            movie_years[idx] = int(str(row['title'])[-5:-1])
            genre_list = str(row['genre']).split('|')
            movie_genre[idx] = genre_list
            for g in genre_list:
                if g not in genres:
                    genres.append(g)
                    cnt_genre += 1
            if movie_years[idx]>y_max:
                y_max=movie_years[idx]
            if movie_years[idx]<y_min:
                y_min=movie_years[idx]

def get_all_adj_kg(kg):
    item_id_to_index = {str(item_id): idx for idx, item_id in enumerate(movie_ids)}

    idx_relation = {idx: set() for idx in range(len(movie_ids))}
    for h, t in zip(kg['h'], kg['t']):
        h_index = item_id_to_index.get(h)
        t_index = item_id_to_index.get(t)
        
        if h_index in idx_relation:
            idx_relation[h_index].add(t_index)

        if t_index in idx_relation:
            idx_relation[t_index].add(h_index) 

    adj_list_weight = np.zeros([item_count, item_count])
    for i in range(item_count):
        for j in range(i+1, item_count):
            shared_relations = len(idx_relation[i].intersection(idx_relation[j]))
            adj_list_weight[i][j] = adj_list_weight[j][i] = shared_relations

    adj_list = []
    for i in range(item_count):
        related_movies_indices = np.argsort(-adj_list_weight[i])
        related_movies_indices = [index for index in related_movies_indices if adj_list_weight[i][index] > 0 and index != i][:kg_topK]
        related_movies_ids = [train_id[j] for j in related_movies_indices if j < len(train_id)]
        adj_list.append(related_movies_ids)

    adj_list_padded = []
    for related_movies_ids in adj_list:
        padded_list = related_movies_ids + [-1] * (kg_topK - len(related_movies_ids))
        adj_list_padded.append(padded_list)

    adj_list_np = np.array(adj_list_padded)
    return adj_list_np

def get_all_adj(item_id):
    adj_list_weight = np.zeros([item_count, item_count])
    adj_list = []
    for id in item_id:
        user_record = interactive_data[interactive_data['u_id'] == id]
        user_record = user_record.sort_values(by='timestep')
        item_list = []
        rating_list = []
        for _, row in user_record.iterrows():
            item_list.append(movie_ids.index(row['i_id']))
            rating_list.append(row['rating'])
        for i in range(len(item_list) - epsilon):
            for j in range(i+1,i+epsilon+1):
                if item_relation_ASG(item_list[i],item_list[j])>0:
                    adj_list_weight[item_list[i]][item_list[j]] += 1
                    adj_list_weight[item_list[j]][item_list[i]] += 1

    for i in range(len(adj_list_weight)):
        tmp = adj_list_weight[i].copy()
        tmp.sort()
        tmp = tmp[::-1]
        last = tmp[min(len(tmp), global_topk) - 1]
        left = []
        for j in range(len(adj_list_weight[i])):
            if adj_list_weight[i][j] >= last and adj_list_weight[i][j] > 0:
                left.append(j)
            else:
                adj_list_weight[i][j] = 0
        left += [0] * (global_topk - len(left))
        left = left[:global_topk]
        adj_list.append(left)

    return np.array(adj_list)

def normalize(rating):
    max_rating = 5
    min_rating = 0
    return -1 + 2 * (rating - min_rating) / (max_rating - min_rating)

def item_relation_ASG(item_1,item_2):
    if item_1 == 0 or item_2 == 0 or item_1 == item_2:
        return 0
    score = 0

    if ratings_path == 'netflix_1_64.dat':
        return 1

    for g in movie_genre[item_1]:
        if g in movie_genre[item_2]:
            score += 1

    if abs(movie_years[item_1]-movie_years[item_2]) < 5:
        score += 1

    return score

def item_relation_WSG(item_1,item_2):
    if item_1 == 0 or item_2 == 0 or item_1 == item_2:
        return 0
    score = 0

    if ratings_path == 'netflix_1_64.dat':
        return 1
    
    for g in movie_genre[item_1]:
        if g in movie_genre[item_2]:
            score += 1

    if abs(movie_years[item_1]-movie_years[item_2]) < 5:
        score += 1

    return score

def process_data(item_list, rating_list, Q_value):
    if len(item_list)>0:
        action = item_list.pop()
        reward = Q_value.pop()
    else:
        action = 0
        reward = 0
    mask = [1.] * len(item_list)
    state_len=max_seq_length-1
    while len(item_list)<state_len:
        item_list.append(0)
        rating_list.append(-1)
        mask.append(0.)

    items = np.unique(item_list).tolist()
    items = items + [0]*(state_len-len(items))
    alias_inputs = [items.index(i) for i in item_list]
    WSG = np.zeros((state_len, state_len), dtype=np.float32)
    AIG = np.zeros((state_len, state_len), dtype=np.float32)
    for i in range(state_len-1):
        u = alias_inputs[i]
        v = alias_inputs[i+1]
        AIG[u][v] += 1

    for i in range(state_len-1):
        u = alias_inputs[i]
        for j in range(i+1,state_len):
            v = alias_inputs[j]
            if item_relation_WSG(item_list[i],item_list[j]) > 0 or (rating_list[i] >= 3 and rating_list[j] >= 3):
                WSG[u][v] = 1

    sum_in = np.sum(WSG, 0)
    sum_in[np.where(sum_in == 0)] = 1
    WSG_in = np.divide(WSG, sum_in)
    sum_out = np.sum(WSG, 1)
    sum_out[np.where(sum_out == 0)] = 1
    WSG_out = np.divide(WSG.transpose(), sum_out)

    sum_in = np.sum(AIG, 0)
    sum_in[np.where(sum_in == 0)] = 1
    AIG_in = np.divide(AIG, sum_in)
    sum_out = np.sum(AIG, 1)
    sum_out[np.where(sum_out == 0)] = 1
    AIG_out = np.divide(AIG.transpose(), sum_out)

    return WSG_in, WSG_out, AIG_in, AIG_out, items, alias_inputs, mask, action, reward

def evaluate(recommend_id, item_id, rating, top_N):
    '''
    evalute the recommend result for each user.
    :param recommend_id: the recommend_result for each item, a list that contains the results for each item. 
    :param item_id: item id.
    :param rating: user's rating on item.
    :param top_N: N, a real number of N for evaluation.
    :return: reward@N, recall@N, MRR@N
    '''
    session_length = len(recommend_id)
    relevant = 0
    recommend_relevant = 0
    selected = 0
    output_reward = 0
    mrr = 0
    if session_length == 0:
        return 0, 0, 0, 0
    for ti in range(session_length):
        current_recommend_id = list(recommend_id[ti])[:top_N]
        current_item = item_id[ti]
        current_rating = rating[ti]
        if current_rating > 3.5:
            relevant += 1
            if current_item in current_recommend_id:
                recommend_relevant += 1
        if current_item in current_recommend_id:
            selected += 1
            output_reward += normalize(current_rating)
            rank = current_recommend_id.index(current_item)
            mrr += 1.0 / (rank + 1)
    recall = recommend_relevant / relevant if relevant != 0 else 0
    precision = recommend_relevant / session_length
    return output_reward / session_length, precision, recall, mrr / session_length



print('Begin training the tree policy.')
start = datetime.datetime.now()
train_step = 0
Loss_list = []
asg_list = get_all_adj(train_id)
keg_list = get_all_adj_kg(kg)

agent = SharedTreePolicy(keg_list=keg_list, asg_list=asg_list, layer=3, branch=int(np.ceil(item_count ** (1 / 3))), learning_rate=learning_rate,
                         max_seq_length=max_seq_length-1, hidden_size=hidden_size, batch_size=batch_size,
                         feature_size=feature_size, gnn_layer=gnn_layer, topK=global_topk)

LIST_WSG_in = []
LIST_WSG_out = []
LIST_AIG_in = []
LIST_AIG_out = []
LIST_item = []
LIST_alias_inputs = []
LIST_mask = []
LIST_reward = []
LIST_action = []
for id1 in train_id:
    user_record = interactive_data[interactive_data['u_id'] == id1]
    user_record = user_record.sort_values(by='timestep')
    item_list = []
    rating_list = []
    Q_value = []
    step_time_list = []
    for _, row in user_record.iterrows():
        item_list.append(movie_ids.index(row['i_id']))
        rating_list.append(row['rating'])
    for rating in rating_list:
        Q_value.append(normalize(rating))
    for i in range(len(Q_value)-1,-1,-1):
        Q_value[i-1] += discount_factor*Q_value[i]

    Loss_list = []
    for i in range(len(item_list)-max_seq_length+1):
        WSG_in, WSG_out, AIG_in, AIG_out, items, alias_inputs, mask, action, reward = process_data(item_list[i:i+max_seq_length],
                                                                      rating_list[i:i+max_seq_length],
                                                                      Q_value[i:i+max_seq_length])
        LIST_WSG_in.append(WSG_in)
        LIST_WSG_out.append(WSG_out)
        LIST_AIG_in.append(AIG_in)
        LIST_AIG_out.append(AIG_out)
        LIST_item.append(items)
        LIST_alias_inputs.append(alias_inputs)
        LIST_mask.append(mask)
        LIST_reward.append(reward)
        LIST_action.append(action)
        if len(LIST_action)==batch_size:
            step_start = datetime.datetime.now()
            loss = agent.learn(LIST_WSG_in, LIST_WSG_out, LIST_AIG_in, LIST_AIG_out, LIST_item, LIST_alias_inputs, LIST_mask, LIST_reward, LIST_action)
            step_end = datetime.datetime.now()
            Loss_list.append(loss)
            step_time = (step_end - step_start).seconds
            step_time_list.append(step_time)
            LIST_WSG_in = []
            LIST_WSG_out = []
            LIST_AIG_in = []
            LIST_AIG_out = []
            LIST_item = []
            LIST_alias_inputs = []
            LIST_mask = []
            LIST_reward = []
            LIST_action = []

    train_step += 1
    print('User ', train_step, 'Loss: ', np.mean(Loss_list))

while(len(LIST_action)>0 and len(LIST_action)<batch_size):
    WSG_in, WSG_out, AIG_in, AIG_out, items, alias_inputs, mask, action, reward = process_data([], [], [])
    LIST_WSG_in.append(WSG_in)
    LIST_WSG_out.append(WSG_out)
    LIST_AIG_in.append(AIG_in)
    LIST_AIG_out.append(AIG_out)
    LIST_item.append(items)
    LIST_alias_inputs.append(alias_inputs)
    LIST_mask.append(mask)
    LIST_reward.append(reward)
    LIST_action.append(action)
if len(LIST_action)==batch_size:
    step_start = datetime.datetime.now()
    loss = agent.learn(LIST_WSG_in, LIST_WSG_out, LIST_AIG_in, LIST_AIG_out, LIST_item, LIST_alias_inputs, LIST_mask, LIST_reward, LIST_action)
    step_end = datetime.datetime.now()
    step_time = (step_end - step_start).seconds
    step_time_list.append(step_time) 

end = datetime.datetime.now()
training_time = (end - start).seconds
print('training_time:', training_time)

print('Begin Test')
test_count = 0
result = []
total_testing_steps = 0
start = datetime.datetime.now()

for id1 in test_id:
    user_record = interactive_data[interactive_data['u_id'] == id1]
    user_record = user_record.sort_values(by='timestep')
    item_list = []
    rating_list = []
    Q_value = []
    test_count += 1
    all_item = []
    all_rating = []
    recommend = []

    LIST_WSG_in = []
    LIST_WSG_out = []
    LIST_AIG_in = []
    LIST_AIG_out = []
    LIST_item = []
    LIST_alias_inputs = []
    LIST_mask = []
    LIST_rating = []
    LIST_action = []

    for _, row in user_record.iterrows():
        item_list.append(movie_ids.index(row['i_id']))
        rating_list.append(row['rating'])
    for rating in rating_list:
        Q_value.append(normalize(rating))

    for i in range(len(item_list)-max_seq_length+1):
        WSG_in, WSG_out, AIG_in, AIG_out, items, alias_inputs, mask, action, reward = process_data(item_list[i:i+max_seq_length],
                                                                      rating_list[i:i+max_seq_length],
                                                                      Q_value[i:i+max_seq_length])
        LIST_WSG_in.append(WSG_in)
        LIST_WSG_out.append(WSG_out)
        LIST_AIG_in.append(AIG_in)
        LIST_AIG_out.append(AIG_out)
        LIST_item.append(items)
        LIST_alias_inputs.append(alias_inputs)
        LIST_mask.append(mask)
        LIST_rating.append(rating_list[i+max_seq_length-1])
        LIST_action.append(action)
        if len(LIST_action) == batch_size:
            #  [batch_size, branch**layer]
            output_action = agent.get_action_prob(LIST_WSG_in, LIST_WSG_out, LIST_AIG_in, LIST_AIG_out, LIST_item, LIST_alias_inputs, LIST_mask)
            for j in range(len(output_action)):
                if LIST_action[j] == 0:
                    break
                recommend_idx = np.argsort(-output_action[j])[:50]
                recommend.append(recommend_idx)
                all_item.append(LIST_action[j])
                all_rating.append(LIST_rating[j])
            LIST_WSG_in = []
            LIST_WSG_out = []
            LIST_AIG_in = []
            LIST_AIG_out = []
            LIST_item = []
            LIST_alias_inputs = []
            LIST_mask = []
            LIST_rating = []
            LIST_action = []

    while (len(LIST_action) > 0 and len(LIST_action) < batch_size):
        WSG_in, WSG_out, AIG_in, AIG_out, items, alias_inputs, mask, action, reward = process_data([], [], [])
        LIST_WSG_in.append(WSG_in)
        LIST_WSG_out.append(WSG_out)
        LIST_AIG_in.append(AIG_in)
        LIST_AIG_out.append(AIG_out)
        LIST_item.append(items)
        LIST_alias_inputs.append(alias_inputs)
        LIST_mask.append(mask)
        LIST_rating.append(0)
        LIST_action.append(action)
    if len(LIST_action) == batch_size:
        output_action = agent.get_action_prob(LIST_WSG_in, LIST_WSG_out, LIST_AIG_in, LIST_AIG_out, LIST_item, LIST_alias_inputs, LIST_mask)
        for j in range(len(output_action)):
            if LIST_action[j] == 0:
                break
            recommend_idx = np.argsort(-output_action[j])[:50]
            recommend.append(recommend_idx)
            all_item.append(LIST_action[j])
            all_rating.append(LIST_rating[j])

    if len(all_rating) > 0:
        reward_10, precision_10, recall_10, mkk_10 = evaluate(recommend, all_item, all_rating, 10)
        reward_30, precision_30, recall_30, mkk_30 = evaluate(recommend, all_item, all_rating, 30)
        print('Test user #', test_count, '/', len(test_id))
        print('Reward@10: %.4f, Precision@10: %.4f, Recall@10: %.4f, MRR@10: %4f'
              % (reward_10, precision_10, recall_10, mkk_10))
        print('Reward@30: %.4f, Precision@30: %.4f, Recall@30: %.4f, MRR@30: %4f'
              % (reward_30, precision_30, recall_30, mkk_30))
        result.append([reward_10, precision_10, recall_10, mkk_10, reward_30, precision_30, recall_30, mkk_30])
end = datetime.datetime.now()
testing_time = (end - start).seconds
print('testing_time',testing_time)
print('###############')
print('Learning finished')
print('Result:')
display = np.mean(np.array(result).reshape([-1, 8]), axis=0)
eval_mat = ["Reward@10", "Precision@10", "Recall@10", "MRR@10", "Reward@30", "Precision@30", "Recall@30", "MRR@30"]
for i in range(len(display)):
    print('%.5f' % display[i])