import os

import numpy as np
import torch
import pickle
from model import LightGCL
from utils import metrics, scipy_sparse_mat_to_torch_sparse_tensor
import pandas as pd
from Aparser import args
from tqdm import tqdm
import time
import torch.utils.data as data
from utils import TrnData
import time
# device = 'cuda:' + args.cuda
# hyperparameters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
d = args.d
l = args.gnn_layer
temp = args.temp
batch_user = args.batch
epoch_no = args.epoch
max_samp = 40
lambda_1 = args.lambda1
lambda_2 = args.lambda2
lambda_3 = args.lambda3
dropout = args.dropout
lr = args.lr
decay = args.decay
svd_q = args.q
gnn_layer=args.gnn_layer

# load data
path = 'data/'
f = open(path+'trnMat.pkl','rb')
train = pickle.load(f)
# print(type(train))  # 检查 train 的类型
# print(train)        # 查看 train 的内容
train_csr = (train!=0).astype(np.float32)
f = open(path+'tstMat.pkl','rb')
test = pickle.load(f)
print('Data loaded.')

print('user_num:',train.shape[0],'item_num:',train.shape[1],'lambda_1:',lambda_1,'lambda_2:',lambda_2,'lambda_3:',lambda_3,'temp:',temp,'q:',svd_q,'GNN_layer',gnn_layer)

epoch_user = min(train.shape[0], 30000)

# normalizing the adj matrix
rowD = np.array(train.sum(1)).squeeze()
colD = np.array(train.sum(0)).squeeze()
for i in range(len(train.data)):
    train.data[i] = train.data[i] / pow(rowD[train.row[i]]*colD[train.col[i]], 0.5)

# construct data loader
train = train.tocoo()
train_data = TrnData(train)
train_loader = data.DataLoader(train_data, batch_size=args.inter_batch, shuffle=True, num_workers=0)

adj_norm = scipy_sparse_mat_to_torch_sparse_tensor(train)
adj_norm = adj_norm.coalesce().to(device)
print('Adj matrix normalized.')

# perform svd reconstruction
adj = scipy_sparse_mat_to_torch_sparse_tensor(train).coalesce().to(device)
print('Performing SVD...')
svd_u,s,svd_v = torch.svd_lowrank(adj, q=svd_q)
u_mul_s = svd_u @ (torch.diag(s))
v_mul_s = svd_v @ (torch.diag(s))
del s
print('SVD done.')

# process test set
test_labels = [[] for i in range(test.shape[0])]
for i in range(len(test.data)):
    row = test.row[i]
    col = test.col[i]
    test_labels[row].append(col)
print('Test data processed.')


loss_list = []
loss_r_list = []
loss_s_list = []
# recall_10_x = []
# recall_10_y = []
# ndcg_20_y = []
# recall_20_y = []
# ndcg_40_y = []

model = LightGCL(adj_norm.shape[0], adj_norm.shape[1], d, u_mul_s, v_mul_s, svd_u.T, svd_v.T, train_csr, adj_norm, l, temp, lambda_1, lambda_2,lambda_3, dropout, batch_user, device)
#model.load_state_dict(torch.load('saved_model.pt'))

model.to(device)

model.preprocessing(2289,956)

print(model.Amax,model.Mmax)

optimizer = torch.optim.Adam(model.parameters(),weight_decay=0,lr=lr)
#optimizer.load_state_dict(torch.load('saved_optim.pt'))

current_lr = lr

for epoch in range(epoch_no):
    # if (epoch+1)%50 == 0:
    #     torch.save(model.state_dict(),'saved_model/saved_model_epoch_'+str(epoch)+'.pt')
    #     torch.save(optimizer.state_dict(),'saved_model/saved_optim_epoch_'+str(epoch)+'.pt')

    epoch_loss = 0
    epoch_loss_r = 0
    epoch_loss_s = 0
    train_loader.dataset.neg_sampling()

    for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch} Training")):
        uids, pos, neg = batch
        uids = uids.long().to(device)
        pos = pos.long().to(device)
        neg = neg.long().to(device)

        iids = torch.cat([pos, neg], dim=0)

        # print(uids.shape, iids.shape)
        # feed
        optimizer.zero_grad()
        loss, loss_r, loss_s= model(uids, iids, pos, neg)
        # print("loss=",loss.item())
        # print("loss_cal_ok!")
        loss.backward()
        optimizer.step()
        #print('batch',batch)
        # print("loss_back_w_ok")
        epoch_loss += loss.cpu().item()
        epoch_loss_r += loss_r.cpu().item()
        epoch_loss_s += loss_s.cpu().item()

        torch.cuda.empty_cache()
        #print(i, len(train_loader), end='\r')

    batch_no = len(train_loader)
    epoch_loss = epoch_loss/batch_no
    epoch_loss_r = epoch_loss_r/batch_no
    epoch_loss_s = epoch_loss_s/batch_no
    loss_list.append(epoch_loss)
    loss_r_list.append(epoch_loss_r)
    loss_s_list.append(epoch_loss_s)
    tqdm.write(f"Epoch {epoch} | Loss: {epoch_loss:.4f} | Loss_r: {epoch_loss_r:.4f} | Loss_s: {epoch_loss_s:.4f}")

    if epoch % 1 == 0:  # test every 10 epochs
        test_uids = np.array([i for i in range(adj_norm.shape[0])])
        batch_no = int(np.ceil(len(test_uids)/batch_user))
        all_recall_10 = 0
        all_ndcg_10 = 0
        all_recall_20 = 0
        all_ndcg_20 = 0
        all_precision_10 = 0
        all_precision_20 = 0
        all_tail_10 = 0
        all_tail_20 = 0

        for batch in tqdm(range(batch_no), desc=f"Epoch {epoch} Testing"):
            start = batch*batch_user
            end = min((batch+1)*batch_user,len(test_uids))

            test_uids_input = torch.LongTensor(test_uids[start:end]).to(device)
            predictions = model(test_uids_input,None,None,None,test=True)
            predictions = np.array(predictions.cpu())

            #top@20
            recall_10, ndcg_10, precision_10 = metrics(test_uids[start:end],predictions,10,test_labels)
            #top@40
            recall_20, ndcg_20 ,precision_20= metrics(test_uids[start:end],predictions,20,test_labels)

            # ===== Tail@K =====
            tail_flags = np.array(model.api_head_tail)  # 0 = tail, 1 = head

            # Top-10
            top10 = predictions[:, :10]
            tail_10 = (tail_flags[top10] == 0).sum(axis=1) / 10
            all_tail_10 += tail_10.mean()

            # Top-20
            top20 = predictions[:, :20]
            tail_20 = (tail_flags[top20] == 0).sum(axis=1) / 20
            all_tail_20 += tail_20.mean()

            all_recall_10+=recall_10
            all_ndcg_10+=ndcg_10
            all_recall_20+=recall_20
            all_ndcg_20+=ndcg_20
            all_precision_10+=precision_10
            all_precision_20+=precision_20
            #print('batch',batch,'recall@20',recall_10,'ndcg@20',ndcg_20,'recall@40',recall_20,'ndcg@40',ndcg_40)
        # print('Test of epoch',epoch,':','Recall@10:',all_recall_10/batch_no,'Ndcg@10:',all_ndcg_10/batch_no,"Precision@10",all_precision_10/batch_no,'Recall@20:',all_recall_20/batch_no,'Ndcg@20:',all_ndcg_20/batch_no,"Precision@20",all_precision_20/batch_no)
        print('Test of epoch', epoch, ':',
              'Recall@10:', all_recall_10 / batch_no,
              'Ndcg@10:', all_ndcg_10 / batch_no,
              'Precision@10:', all_precision_10 / batch_no,
              'Tail@10:', all_tail_10 / batch_no,
              'Recall@20:', all_recall_20 / batch_no,
              'Ndcg@20:', all_ndcg_20 / batch_no,
              'Precision@20:', all_precision_20 / batch_no,
              'Tail@20:', all_tail_20 / batch_no)
        print('-------------------------------------------')
        # recall_10_x.append(epoch)
        # recall_10_y.append(all_recall_10/batch_no)
        # ndcg_10_y.append(all_ndcg_10/batch_no)
        # recall_20_y.append(all_recall_20/batch_no)
        # ndcg_20_y.append(all_ndcg_20/batch_no)
    if epoch == 129:
        result_dict = {
            "epoch": epoch,
            "Recall@10": all_recall_10 / batch_no,
            "NDCG@10": all_ndcg_10 / batch_no,
            "Precision@10": all_precision_10 / batch_no,
            "Recall@20": all_recall_20 / batch_no,
            "NDCG@20": all_ndcg_20 / batch_no,
            "Precision@20": all_precision_20 / batch_no,
            "Tail@10": all_tail_10 / batch_no,
            "Tail@20": all_tail_20 / batch_no,
            # 记录超参数
            "embedding_dim": d,
            "gnn_layer": gnn_layer,
            "temp": temp,
            "lambda2": lambda_2,
        }

        df = pd.DataFrame([result_dict])  # 注意加中括号表示一行
        os.makedirs("log", exist_ok=True)
        save_path = "log/final_results.csv"
        if not os.path.exists(save_path):
            df.to_csv(save_path, index=False)
        else:
            df.to_csv(save_path, mode="a", header=False, index=False)

        print(f"第 {epoch} epoch 结果已保存到 {save_path}")
# # === Final test after training ===
# test_uids = np.array([i for i in range(adj_norm.shape[0])])
# batch_no = int(np.ceil(len(test_uids)/batch_user))
#
# all_recall_20 = 0
# all_ndcg_20 = 0
# all_recall_40 = 0
# all_ndcg_40 = 0
#
# for batch in tqdm(range(batch_no), desc="Final Testing"):
#     start = batch * batch_user
#     end = min((batch+1) * batch_user, len(test_uids))
#
#     test_uids_input = torch.LongTensor(test_uids[start:end]).to(device)
#     predictions = model(test_uids_input, None, None, None, test=True)
#     predictions = np.array(predictions.cpu())
#
#     # top@20
#     recall_20, ndcg_20 ,precision_20= metrics(test_uids[start:end], predictions, 20, test_labels)
#     # top@40
#     recall_40, ndcg_40, precision_40 = metrics(test_uids[start:end], predictions, 40, test_labels)
#
#     all_recall_20 += recall_20
#     all_ndcg_20 += ndcg_20
#     all_recall_40 += recall_40
#     all_ndcg_40 += ndcg_40
#
# tqdm.write('-------------------------------------------')
# tqdm.write(f"Final test results:\n"
#            f"Recall@20: {all_recall_20/batch_no:.4f} | "
#            f"Ndcg@20: {all_ndcg_20/batch_no:.4f}\n"
#            f"Recall@40: {all_recall_40/batch_no:.4f} | "
#            f"Ndcg@40: {all_ndcg_40/batch_no:.4f}")
#
# # === Save metrics to CSV ===
# metric = pd.DataFrame({
#     'epoch': list(range(epoch_no)) + ['Final'],
#     'recall@20': loss_r_list + [all_recall_20/batch_no],
#     'ndcg@20': loss_s_list + [all_ndcg_20/batch_no],
#     'recall@40': loss_list + [all_recall_40 /batch_no],
#     'ndcg@40': [None] * epoch_no + [all_ndcg_40/batch_no]  # 注意：这里只存 Final 的 ndcg@40
# })
#
# os.makedirs("log", exist_ok=True)
# os.makedirs("saved_model", exist_ok=True)
#
# current_t = time.gmtime()
# metric.to_csv(f'log/result_{args.data}_{time.strftime("%Y-%m-%d-%H", current_t)}.csv')
#
# # === Save model and optimizer ===
# torch.save(model.state_dict(),
#            f'saved_model/saved_model_{args.data}_{time.strftime("%Y-%m-%d-%H", current_t)}.pt')
# torch.save(optimizer.state_dict(),
#            f'saved_model/saved_optim_{args.data}_{time.strftime("%Y-%m-%d-%H", current_t)}.pt')
#
# # torch.save(optimizer.state_dict(),'saved_model/saved_optim_'+args.data+'_'+time.strftime('%Y-%m-%d-%H',current_t)+'.pt')