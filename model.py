# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class BaseClass(torch.nn.Module):
    def __init__(self):
        super(BaseClass, self).__init__()
        self.cur_itr = torch.nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)
        self.best_mrr = torch.nn.Parameter(torch.tensor(0, dtype=torch.float64), requires_grad=False)
        self.best_itr = torch.nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)
        self.best_hit1 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float64), requires_grad=False)



class MyLoss(torch.nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        return

    def forward(self, pred1, tar1):
        pred1 = F.softmax(pred1, dim=1)
        loss = -torch.log(pred1[tar1 == 1]).sum()
        return loss



class HyMRformer(BaseClass):

    def __init__(self, n_ent, n_rel, input_drop, hidden_drop, feature_drop, RAD_Size, num_heads,
                 emb_dim, emb_dim1, max_arity, device):
        super(HyMRformer, self).__init__()
        self.loss = MyLoss()
        self.n_ent = n_ent
        self.n_rel = n_rel
        self.device = device
        
        self.emb_dim = emb_dim
        self.emb_dim1 = emb_dim1
        self.emb_dim2 = emb_dim // emb_dim1
        
        self.max_arity = max_arity
        
        self.input_drop = nn.Dropout(input_drop) # input_drop
        self.hidden_drop = nn.Dropout(hidden_drop) # hidden_drop
        self.feature_drop = nn.Dropout(feature_drop) # feature_drop
 
        self.ent_embeddings = nn.Parameter(torch.Tensor(self.n_ent, self.emb_dim))
        self.rel_embeddings = nn.Parameter(torch.Tensor(self.n_rel, self.emb_dim))
        self.pos_embeddings = nn.Embedding(self.max_arity, self.emb_dim)


        ### The Parameters of Relation-Aware Deformable Convolution
        self.RAP_Size = 16
        self.RAD_Size = RAD_Size # The size coefficient of the relation-aware internal convolution kernels
        
        self.Rel_PosW = nn.Linear(in_features=self.emb_dim, out_features=self.RAP_Size)
        self.Rel_PosInvW = nn.Linear(in_features=self.emb_dim * self.RAP_Size, out_features=self.emb_dim)

        self.Rel_W2 = nn.Linear(in_features=self.emb_dim, out_features=1*self.RAD_Size)
        self.Rel_W3 = nn.Linear(in_features=self.emb_dim, out_features=2*self.RAD_Size)
        self.Rel_W4 = nn.Linear(in_features=self.emb_dim, out_features=3*self.RAD_Size)
        self.Rel_W5 = nn.Linear(in_features=self.emb_dim, out_features=4*self.RAD_Size)
        self.Rel_W6 = nn.Linear(in_features=self.emb_dim, out_features=5*self.RAD_Size)
        self.Rel_W7 = nn.Linear(in_features=self.emb_dim, out_features=6*self.RAD_Size)
        self.Rel_W8 = nn.Linear(in_features=self.emb_dim, out_features=7*self.RAD_Size)
        self.Rel_W9 = nn.Linear(in_features=self.emb_dim, out_features=8*self.RAD_Size)

        self.pool = torch.nn.MaxPool2d((2, 1))
        self.conv_size = self.emb_dim * self.RAD_Size // 2

        self.fc_layer = nn.Linear(in_features=self.conv_size, out_features=self.emb_dim)

        self.bn = nn.BatchNorm2d(num_features=1)
        self.Posbn = nn.BatchNorm1d(num_features=1)        
        # self.bn1 = nn.BatchNorm3d(num_features=1)
        # self.bn2 = nn.BatchNorm3d(num_features=4)
        # self.bn3 = nn.BatchNorm2d(num_features=32)
        # self.bn4 = nn.BatchNorm1d(num_features=self.conv_size)
        
        

        ### The Parameters of Relation-Aware Multi-Head Self-Attention
        self.heads = num_heads
        self.d_model = emb_dim
        self.hidden_size = self.d_model * 2
        
        self.q_linear = nn.Linear(in_features=self.d_model, out_features=self.heads * self.d_model)
        self.k_linear = nn.Linear(in_features=self.d_model, out_features=self.heads * self.d_model)
        self.v_linear = nn.Linear(in_features=self.d_model, out_features=self.heads * self.d_model)

        self.fc_attn = nn.Linear(in_features=self.heads * self.d_model, out_features=self.d_model)
        self.layer_norm = nn.LayerNorm(self.d_model)
        
        # self.w_1 = nn.Linear(in_features=self.d_model, out_features=self.hidden_size)
        # self.w_2 = nn.Linear(in_features=self.hidden_size, out_features=self.d_model)
        
        self.w_1 = nn.Conv1d(self.d_model, self.hidden_size, 1)
        self.w_2 = nn.Conv1d(self.hidden_size, self.d_model, 1)

        

        self.register_parameter('b', nn.Parameter(torch.zeros(n_ent)))

        nn.init.xavier_uniform_(self.ent_embeddings.data)
        nn.init.xavier_uniform_(self.rel_embeddings.data)
        nn.init.xavier_uniform_(self.pos_embeddings.weight.data)

        
        nn.init.xavier_uniform_(self.Rel_PosW.weight.data)
        nn.init.xavier_uniform_(self.Rel_PosInvW.weight.data)        
        nn.init.xavier_uniform_(self.Rel_W2.weight.data)
        nn.init.xavier_uniform_(self.Rel_W3.weight.data)
        nn.init.xavier_uniform_(self.Rel_W4.weight.data)
        nn.init.xavier_uniform_(self.Rel_W5.weight.data)
        nn.init.xavier_uniform_(self.Rel_W6.weight.data)
        nn.init.xavier_uniform_(self.Rel_W7.weight.data)
        nn.init.xavier_uniform_(self.Rel_W8.weight.data)
        nn.init.xavier_uniform_(self.Rel_W9.weight.data)
        
        nn.init.xavier_uniform_(self.q_linear.weight.data)
        nn.init.xavier_uniform_(self.k_linear.weight.data)
        nn.init.xavier_uniform_(self.v_linear.weight.data)  
         
        nn.init.xavier_uniform_(self.fc_layer.weight.data)
        nn.init.xavier_uniform_(self.fc_attn.weight.data)
        
        # nn.init.xavier_uniform_(self.w_1.weight.data)
        # nn.init.xavier_uniform_(self.w_2.weight.data)
        
        nn.init.kaiming_uniform_(self.w_1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.w_2.weight, mode='fan_in', nonlinearity='relu')



    def RA_Position(self, rel_embedding, ent_embedding, position):
        ''' Relation-Aware Position Embedding Module '''
        ent_embedding = ent_embedding.view(ent_embedding.shape[0], 1, -1)
        
        RA_PosConv1 = self.Rel_PosW(rel_embedding).view(ent_embedding.shape[0]*self.RAP_Size, 1, 1)
        RA_PosConv2 = self.Rel_PosW(rel_embedding).view(ent_embedding.shape[0]*self.RAP_Size, 1, 1)
        RA_PosConv3 = self.Rel_PosW(rel_embedding).view(ent_embedding.shape[0]*self.RAP_Size, 1, 1)
        RA_PosConv4 = self.Rel_PosW(rel_embedding).view(ent_embedding.shape[0]*self.RAP_Size, 1, 1)
        RA_PosConv5 = self.Rel_PosW(rel_embedding).view(ent_embedding.shape[0]*self.RAP_Size, 1, 1)
        RA_PosConv6 = self.Rel_PosW(rel_embedding).view(ent_embedding.shape[0]*self.RAP_Size, 1, 1)
        RA_PosConv7 = self.Rel_PosW(rel_embedding).view(ent_embedding.shape[0]*self.RAP_Size, 1, 1)
        RA_PosConv8 = self.Rel_PosW(rel_embedding).view(ent_embedding.shape[0]*self.RAP_Size, 1, 1)
        RA_PosConv9 = self.Rel_PosW(rel_embedding).view(ent_embedding.shape[0]*self.RAP_Size, 1, 1)
        
        x = self.Posbn(ent_embedding)
        x =  x.permute(1, 0, 2)

        if position == 1:
            x = F.conv1d(x, RA_PosConv1, groups=ent_embedding.size(0))
        if position == 2:
            x = F.conv1d(x, RA_PosConv2, groups=ent_embedding.size(0))
        if position == 3:
            x = F.conv1d(x, RA_PosConv3, groups=ent_embedding.size(0))
        if position == 4:
            x = F.conv1d(x, RA_PosConv4, groups=ent_embedding.size(0))
        if position == 5:
            x = F.conv1d(x, RA_PosConv5, groups=ent_embedding.size(0))
        if position == 6:
            x = F.conv1d(x, RA_PosConv6, groups=ent_embedding.size(0))
        if position == 7:
            x = F.conv1d(x, RA_PosConv7, groups=ent_embedding.size(0))
        if position == 8:
            x = F.conv1d(x, RA_PosConv8, groups=ent_embedding.size(0))
        if position == 9:
            x = F.conv1d(x, RA_PosConv9, groups=ent_embedding.size(0))
        
        x = x.contiguous().view(-1, self.RAP_Size * self.emb_dim)
        x = self.Rel_PosInvW(x)
        
        x = x.view(-1, 1, self.emb_dim)

        return x




    def RADConv(self, ent_embedding, rel_embedding, arity):
        ''' Relation-Aware Deformable Convolution Module '''
        
        ent_embedding = ent_embedding.view(ent_embedding.shape[0], 1, -1,  arity-1)
        
        ## self.RAD_Size * (arity - 1)
        rad_kernel2 = self.Rel_W2(rel_embedding).view(ent_embedding.size(0)*self.RAD_Size, 1, 1, 1)
        rad_kernel3 = self.Rel_W3(rel_embedding).view(ent_embedding.size(0)*self.RAD_Size, 1, 1, 2)
        rad_kernel4 = self.Rel_W4(rel_embedding).view(ent_embedding.size(0)*self.RAD_Size, 1, 1, 3)
        rad_kernel5 = self.Rel_W5(rel_embedding).view(ent_embedding.size(0)*self.RAD_Size, 1, 1, 4)
        rad_kernel6 = self.Rel_W6(rel_embedding).view(ent_embedding.size(0)*self.RAD_Size, 1, 1, 5)
        rad_kernel7 = self.Rel_W7(rel_embedding).view(ent_embedding.size(0)*self.RAD_Size, 1, 1, 6)
        rad_kernel8 = self.Rel_W8(rel_embedding).view(ent_embedding.size(0)*self.RAD_Size, 1, 1, 7)
        rad_kernel9 = self.Rel_W9(rel_embedding).view(ent_embedding.size(0)*self.RAD_Size, 1, 1, 8)

        x = ent_embedding
        x = self.bn(x)
        x = x.permute(1, 0, 2, 3)

        if arity == 2:
            x = F.conv2d(x, rad_kernel2, groups=ent_embedding.size(0))
        if arity == 3:
            x = F.conv2d(x, rad_kernel3, groups=ent_embedding.size(0))
        if arity == 4:
            x = F.conv2d(x, rad_kernel4, groups=ent_embedding.size(0))
        if arity == 5:
            x = F.conv2d(x, rad_kernel5, groups=ent_embedding.size(0))
        if arity == 6:
            x = F.conv2d(x, rad_kernel6, groups=ent_embedding.size(0))
        if arity == 7:
            x = F.conv2d(x, rad_kernel7, groups=ent_embedding.size(0))
        if arity == 8:
            x = F.conv2d(x, rad_kernel8, groups=ent_embedding.size(0))          
        if arity == 9:
            x = F.conv2d(x, rad_kernel9, groups=ent_embedding.size(0))
        
        x = self.pool(x)
        x = x.contiguous().view(-1, self.conv_size)
        x = self.feature_drop(x)

        return x


    def RADConv_Process(self, concat_input):
        ''' The Feature Process of Relation-Aware Deformable Convolution Module '''        
        r = concat_input[:, 0, :]
        
        if concat_input.shape[1] == 2:
            e1 = concat_input[:, 1, :]

            e1 = self.RA_Position(r, e1, position = 1)
            
            ent_features = e1
            x = self.RADConv(ent_features, r, arity = 2)
                    
        if concat_input.shape[1] == 3:
            e1 = concat_input[:, 1, :]
            e2 = concat_input[:, 2, :]
            
            e1 = self.RA_Position(r, e1, position = 1)
            e2 = self.RA_Position(r, e2, position = 2)

            ent_features = torch.cat((e1, e2), dim=1)
            x = self.RADConv(ent_features, r, arity = 3)

        if concat_input.shape[1] == 4:
            e1 = concat_input[:, 1, :]
            e2 = concat_input[:, 2, :]
            e3 = concat_input[:, 3, :]
            
            e1 = self.RA_Position(r, e1, position = 1)
            e2 = self.RA_Position(r, e2, position = 2)
            e3 = self.RA_Position(r, e3, position = 3)
            
            ent_features = torch.cat((e1, e2, e3), dim=1)
            x = self.RADConv(ent_features, r, arity = 4)
 
        if concat_input.shape[1] == 5:
            e1 = concat_input[:, 1, :]
            e2 = concat_input[:, 2, :]
            e3 = concat_input[:, 3, :]
            e4 = concat_input[:, 4, :]
            
            e1 = self.RA_Position(r, e1, position = 1)
            e2 = self.RA_Position(r, e2, position = 2)
            e3 = self.RA_Position(r, e3, position = 3)
            e4 = self.RA_Position(r, e4, position = 4)
            
            ent_features = torch.cat((e1, e2, e3, e4), dim=1)           
            x = self.RADConv(ent_features, r, arity = 5)
            
        if concat_input.shape[1] == 6:
            e1 = concat_input[:, 1, :]
            e2 = concat_input[:, 2, :]
            e3 = concat_input[:, 3, :]
            e4 = concat_input[:, 4, :]
            e5 = concat_input[:, 5, :]
            
            e1 = self.RA_Position(r, e1, position = 1)
            e2 = self.RA_Position(r, e2, position = 2)
            e3 = self.RA_Position(r, e3, position = 3)
            e4 = self.RA_Position(r, e4, position = 4)
            e5 = self.RA_Position(r, e5, position = 5)
                        
            ent_features = torch.cat((e1, e2, e3, e4, e5), dim=1)           
            x = self.RADConv(ent_features, r, arity = 6)
            
        if concat_input.shape[1] == 7:
            e1 = concat_input[:, 1, :]
            e2 = concat_input[:, 2, :]
            e3 = concat_input[:, 3, :]
            e4 = concat_input[:, 4, :]
            e5 = concat_input[:, 5, :]
            e6 = concat_input[:, 6, :]
            
            e1 = self.RA_Position(r, e1, position = 1)
            e2 = self.RA_Position(r, e2, position = 2)
            e3 = self.RA_Position(r, e3, position = 3)
            e4 = self.RA_Position(r, e4, position = 4)
            e5 = self.RA_Position(r, e5, position = 5)
            e6 = self.RA_Position(r, e6, position = 6)
            
            ent_features = torch.cat((e1, e2, e3, e4, e5, e6), dim=1)           
            x = self.RADConv(ent_features, r, arity = 7)            

        if concat_input.shape[1] == 8:
            e1 = concat_input[:, 1, :]
            e2 = concat_input[:, 2, :]
            e3 = concat_input[:, 3, :]
            e4 = concat_input[:, 4, :]
            e5 = concat_input[:, 5, :]
            e6 = concat_input[:, 6, :]
            e7 = concat_input[:, 7, :]
            
            e1 = self.RA_Position(r, e1, position = 1)
            e2 = self.RA_Position(r, e2, position = 2)
            e3 = self.RA_Position(r, e3, position = 3)
            e4 = self.RA_Position(r, e4, position = 4)
            e5 = self.RA_Position(r, e5, position = 5)
            e6 = self.RA_Position(r, e6, position = 6)
            e7 = self.RA_Position(r, e7, position = 7)
                       
            ent_features = torch.cat((e1, e2, e3, e4, e5, e6, e7), dim=1)           
            x = self.RADConv(ent_features, r, arity = 8)
        
        if concat_input.shape[1] == 9:
            e1 = concat_input[:, 1, :]
            e2 = concat_input[:, 2, :]
            e3 = concat_input[:, 3, :]
            e4 = concat_input[:, 4, :]
            e5 = concat_input[:, 5, :]
            e6 = concat_input[:, 6, :]
            e7 = concat_input[:, 7, :]
            e8 = concat_input[:, 8, :]
            
            e1 = self.RA_Position(r, e1, position = 1)
            e2 = self.RA_Position(r, e2, position = 2)
            e3 = self.RA_Position(r, e3, position = 3)
            e4 = self.RA_Position(r, e4, position = 4)
            e5 = self.RA_Position(r, e5, position = 5)
            e6 = self.RA_Position(r, e6, position = 6)
            e7 = self.RA_Position(r, e7, position = 7)
            e8 = self.RA_Position(r, e8, position = 8)
            
            ent_features = torch.cat((e1, e2, e3, e4, e5, e6, e7, e8), dim=1)           
            x = self.RADConv(ent_features, r, arity = 9) 

        return x






    def Attention(self, Q, K, V):
        ## scores = (Q*K^T)/sqrt(d)
        attn = torch.bmm(Q, K.transpose(1, 2))
        attn = attn / np.power(self.d_model, 0.5)
        attn = F.softmax(attn, dim = 2)

        output = torch.bmm(attn, V)

        return output



    def MultiHeadAttention(self, feature_Q, feature_K, feature_V):
        ''' The Feature Process of Relation-Aware Multi-Head Self-Attention Module '''         
        # r = concat_input[:, 0, :]
        
        q = feature_Q.view(-1, 1, self.d_model)
        k = feature_K.view(-1, 1, self.d_model)
        v = feature_V.view(-1, 1, self.d_model)

        residual = q.view(-1, self.d_model)

        sz_b, len_q, _ = q.size() # batch size, len_size
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        Q = self.q_linear(q).view(sz_b, -1, self.heads, self.d_model)
        K = self.k_linear(k).view(sz_b, -1, self.heads, self.d_model)
        V = self.v_linear(v).view(sz_b, -1, self.heads, self.d_model)

        Q = Q.transpose(1, 2).contiguous().view(-1, len_q, self.d_model)  # (batch*heads) x lq x dk
        K = K.transpose(1, 2).contiguous().view(-1, len_k, self.d_model)  # (batch) x lk x dk
        V = V.transpose(1, 2).contiguous().view(-1, len_v, self.d_model)  # (batch) x lv x dv

        scores = self.Attention(Q, K, V)
        
        x = scores.view(sz_b, self.heads, len_q, self.d_model)
        
        # x = x.transpose(1, 2).contiguous().view(sz_b, len_q, -1)  # batch x lq x (heads*dv)
        # x = self.fc_attn(x)
        # x = x.view(-1, self.emb_dim)
        
        x = x.transpose(1, 2).contiguous().view(sz_b, -1)  # batch x (lq x heads*dv)
        x = self.fc_attn(x)
        x = self.layer_norm(x + residual)

        return x



    def HyMRformer(self, concat_input):        
        
        v_convQ = self.RADConv_Process(concat_input)
        v_convK = self.RADConv_Process(concat_input)
        v_convV = self.RADConv_Process(concat_input)
        
        v_convQ = self.fc_layer(v_convQ)
        v_convK = self.fc_layer(v_convK)      
        v_convV = self.fc_layer(v_convV)      

        v_out = self.MultiHeadAttention(v_convQ, v_convK, v_convV)
        
        x = v_out
        residual = x

        x = x.transpose(0, 1)

        x = self.w_2(F.relu(self.w_1(x)))
        
        x = x.transpose(0, 1)
        
        x = self.hidden_drop(x)
        x = self.layer_norm(x + residual)

        return x







    def forward(self, rel_idx, ent_idx, miss_ent_domain):

        r = self.rel_embeddings[rel_idx].unsqueeze(1)
        ents = self.ent_embeddings[ent_idx]
        
        concat_input = torch.cat((r, ents), dim=1)   
        concat_input = self.input_drop(concat_input) # input_drop    
        
        x = self.HyMRformer(concat_input)

        miss_ent_domain = torch.LongTensor([miss_ent_domain-1]).to(self.device)
        mis_pos = self.pos_embeddings(miss_ent_domain)
        tar_emb = self.ent_embeddings + mis_pos
        
        scores = torch.mm(x, tar_emb.transpose(0, 1))
        scores += self.b.expand_as(scores)

        return scores