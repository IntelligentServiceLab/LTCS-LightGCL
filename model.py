import torch
import torch.nn as nn
import random
import pickle
def linalg_norm(input, ord=2, dim=-1, keepdim=False):
    return torch.linalg.norm(input, ord=ord, dim=dim, keepdim=keepdim)
import math
from utils import sparse_dropout, spmm,get_weight
import numpy as np
from scipy.sparse import csr_matrix
def random_sample_alternative(x, y):
    if y > x + 1:
        raise ValueError("y cannot be greater than the range size (x + 1)")
    nums = list(range(x + 1))
    random.shuffle(nums)  # 随机打乱列表
    return nums[:y]       # 返回前 y 个数


class LightGCL(nn.Module):
    def __init__(self, n_u, n_i, d, u_mul_s, v_mul_s, ut, vt, train_csr, adj_norm, l, temp, lambda_1, lambda_2,
                 lambda_3, dropout, batch_user, device):
        super(LightGCL, self).__init__()

        # 新增：长尾判断相关属性
        self.api_popularity = None  # 存储每个API的流行度（被调用次数）
        self.head_threshold = None  # 头部API的阈值
        self.api_head_tail = None

        self.E_u_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_u,d)))
        self.E_i_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_i,d)))

        self.train_csr = train_csr
        self.adj_norm = adj_norm

        self.l = l
        self.E_u_list = [None] * (l+1)
        self.E_i_list = [None] * (l+1)
        self.E_u_list[0] = self.E_u_0
        self.E_i_list[0] = self.E_i_0
        self.Z_u_list = [None] * (l+1)
        self.Z_i_list = [None] * (l+1)
        self.G_u_list = [None] * (l+1)
        self.G_i_list = [None] * (l+1)
        self.G_u_list[0] = self.E_u_0
        self.G_i_list[0] = self.E_i_0
        self.temp = temp

        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3

        self.dropout = dropout
        self.act = nn.LeakyReLU(0.5)
        self.batch_user = batch_user

        self.E_u = None
        self.E_i = None

        self.u_mul_s = u_mul_s
        self.v_mul_s = v_mul_s
        self.ut = ut
        self.vt = vt

        self.device = device

        # api_flag will be set in preprocessing()
        self.api_flag = None

    def preprocessing(self, mashup_num, api_num):

        # 确保是 CSR 格式，可以方便地按行取数据
        train_mat = self.train_csr
        if not isinstance(train_mat, csr_matrix):
            train_mat = train_mat.tocsr()

        # ====== 新增：计算API流行度并判断头部/尾部 ======
        # 计算每个API的被调用次数
        self.api_popularity = [0] * api_num
        for i in range(mashup_num):
            items = train_mat.getrow(i).indices
            for api_id in items:
                self.api_popularity[api_id] += 1

        # 设置头部API阈值（前20%为头部）
        popularity_sorted = sorted(self.api_popularity, reverse=True)
        self.head_threshold = popularity_sorted[int(api_num * 0.2)]  # 前20%作为头部

        # 标记每个API是头部(1)还是尾部(0)
        self.api_head_tail = [1 if count >= self.head_threshold else 0 for count in self.api_popularity]

        print(f"头部API阈值: {self.head_threshold}")
        print(f"头部API数量: {sum(self.api_head_tail)}, 尾部API数量: {api_num - sum(self.api_head_tail)}")

        # 初始化矩阵
        Amap = [[0] * api_num for _ in range(api_num)]  # API-API 共现次数
        Mmap = [[0] * mashup_num for _ in range(mashup_num)]  # Mashup-Mashup 相似度

        Amax, Mmax = 1, 1

        # ====== 构建 Amap (API-API 共现) ======
        for i in range(mashup_num):
            items = train_mat.getrow(i).indices  # mashup i 调用的所有 API
            for m in range(len(items)):
                for n in range(len(items)):
                    if items[m] == items[n]:
                        continue
                    Amap[items[m]][items[n]] += 1
                    Amax = max(Amax, Amap[items[m]][items[n]])

        # ====== 构建 Mmap (Mashup-Mashup 相似度) ======
        for i in range(mashup_num):
            items_i = set(train_mat.getrow(i).indices)
            for j in range(mashup_num):
                if i == j:
                    continue
                items_j = set(train_mat.getrow(j).indices)
                common = len(items_i & items_j)
                if common > 0:
                    Mmap[i][j] = common
                    Mmax = max(Mmax, Mmap[i][j])

        # 转换成邻接表形式
        Alist = [[j for j in range(api_num) if Amap[i][j] > 0] for i in range(api_num)]
        Mlist = [[j for j in range(mashup_num) if Mmap[i][j] > 0] for i in range(mashup_num)]

        # 保存结果到对象
        self.Amap = Amap
        self.Mmap = Mmap
        self.Amax = Amax
        self.Mmax = Mmax
        self.Alist = Alist
        self.Mlist = Mlist
        self.mashup_num = mashup_num
        self.api_num = api_num

        print("示例 Amap[0][:5] =", Amap[0][:5])
        print("示例 Mmap[0][:5] =", Mmap[0][:5])
        print("Alist[0] =", Alist[0][:10])
        print("Mlist[0] =", Mlist[0][:10])

    def forward(self, uids, iids, pos, neg, test=False):
        if test==True:  # testing phase
            preds = self.E_u[uids] @ self.E_i.T
            mask = self.train_csr[uids.cpu().numpy()].toarray()
            mask = torch.Tensor(mask).to(self.device)
            preds = preds * (1-mask) - 1e8 * mask
            predictions = preds.argsort(descending=True)
            return predictions
        else:  # training phase
            for layer in range(1,self.l+1):
                # GNN propagation
                # self.G_u 和 self.G_i：聚合所有层次的 SVD 增强特征。
                # self.E_u 和 self.E_i：聚合所有层次的 GNN 传播特征。
                self.Z_u_list[layer] = (torch.spmm(sparse_dropout(self.adj_norm,self.dropout), self.E_i_list[layer-1]))
                self.Z_i_list[layer] = (torch.spmm(sparse_dropout(self.adj_norm,self.dropout).transpose(0,1), self.E_u_list[layer-1]))

                # svd_adj propagation
                vt_ei = self.vt @ self.E_i_list[layer-1]
                self.G_u_list[layer] = (self.u_mul_s @ vt_ei)
                ut_eu = self.ut @ self.E_u_list[layer-1]
                self.G_i_list[layer] = (self.v_mul_s @ ut_eu)

                # aggregate
                self.E_u_list[layer] = self.Z_u_list[layer]
                self.E_i_list[layer] = self.Z_i_list[layer]

            self.G_u = sum(self.G_u_list)
            self.G_i = sum(self.G_i_list)

            # aggregate across layers
            self.E_u = sum(self.E_u_list)
            self.E_i = sum(self.E_i_list)

            # cl loss
            G_u_norm = self.G_u
            E_u_norm = self.E_u
            G_i_norm = self.G_i
            E_i_norm = self.E_i
            neg_score = torch.log(torch.exp(G_u_norm[uids] @ E_u_norm.T / self.temp).sum(1) + 1e-8).mean()
            neg_score += torch.log(torch.exp(G_i_norm[iids] @ E_i_norm.T / self.temp).sum(1) + 1e-8).mean()
            pos_score = (torch.clamp((G_u_norm[uids] * E_u_norm[uids]).sum(1) / self.temp,-5.0,5.0)).mean() + (torch.clamp((G_i_norm[iids] * E_i_norm[iids]).sum(1) / self.temp,-5.0,5.0)).mean()
            loss_s = -pos_score + neg_score

            # bpr loss
            u_emb = self.E_u[uids]
            pos_emb = self.E_i[pos]
            neg_emb = self.E_i[neg]
            pos_scores = (u_emb * pos_emb).sum(-1)
            neg_scores = (u_emb * neg_emb).sum(-1)

            loss_r = -(pos_scores - neg_scores).sigmoid().log().mean()

            # reg loss
            loss_reg = 0
            for param in self.parameters():
                loss_reg += param.norm(2).square()
            loss_reg *= self.lambda_3
            # print(loss_r,loss_reg,loss_s)
            # cacl loss
            Acacl = 0
            Mcacl = 0

            # for i in range(len(uids)):
            #     Mvis[uids[i]] = 1
            # for i in range(len(iids)):
            #     Avis[iids[i]] = 1
            Mashup_list = random_sample_alternative(self.mashup_num-1,100)
            Api_list = random_sample_alternative(self.api_num-1,100)
            Mvis = [0 for _ in range(self.mashup_num)]
            Avis = [0 for _ in range(self.api_num)]
            for i in Mashup_list:
                Mvis[i] = 1
            for i in Api_list:
                Avis[i] = 1
            for i in Mashup_list:
                for j in range(len(self.Mlist[i])):
                    uid2 = self.Mlist[i][j]
                    if Mvis[uid2] == 0:
                        continue
                    uid1 = i
                    Num = self.Mmap[uid1][uid2]
                    Weight = 1/(1+math.exp((-5)*(Num/self.Mmax)))
                    # Mcacl += 1/(1+math.exp(self.Mmax-5-Num)) * torch.norm(self.E_u[uid1]-self.E_u[uid2],p=2)
                    temp  = torch.norm(self.E_u[uid1] - self.E_u[uid2],p=2)
                    Mcacl += Weight * temp
                    #     # total+= temp
                    #     # times1 += Weight
                    #     # times2 += 1/(1+math.exp(self.Mmax-5-Num))

            # 在计算Acacl的部分，完整修改如下：
            for i in Api_list:
                for j in range(len(self.Alist[i])):
                    iid2 = self.Alist[i][j]
                    if Avis[iid2] == 0:
                        continue
                    iid1 = i
                    Num = self.Amap[iid1][iid2]  # 这两个API被同一个Mashup调用的次数

                    # 原有的共现权重计算
                    Weight = 1 / (1 + math.exp((-5) * (Num / self.Amax)))

                    # ====== 新增：长尾调节系数（基于这两个有共现关系的API） ======
                    head_tail_i = self.api_head_tail[iid1]
                    head_tail_j = self.api_head_tail[iid2]

                    if head_tail_i == 1 and head_tail_j == 1:  # 都是头部API
                        alpha = 1.0
                    elif head_tail_i == 0 and head_tail_j == 0:  # 都是尾部API
                        alpha = 0.5
                    else:  # 一个头部一个尾部
                        alpha = 2.0

                    adjusted_weight = Weight * alpha
                    temp = torch.norm(self.E_i[iid1] - self.E_i[iid2], p=2)
                    Acacl += adjusted_weight * temp

            # cacl_loss = self.lambda_2 * (Acacl + Mcacl)
            # cacl_loss = self.lambda_2 * (Acacl)   #消融实验方法2
            cacl_loss = self.lambda_2 * (Mcacl)   #消融实验方法3
            # print(times1,times2,total)
            # # print(times)
            # # print(cacl_loss)
            # # total loss
            # print(cacl_loss)
            loss = loss_r + self.lambda_1 * loss_s + cacl_loss + loss_reg
            # print("loss:",loss)
            #print('loss',loss.item(),'loss_r',loss_r.item(),'loss_s',loss_s.item())
            return loss, loss_r, self.lambda_1 * loss_s
