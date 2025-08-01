import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# class Autoencoder1(nn.Module):
#     def __init__(self, y_in, y1, y2, y3, y4):
#         super(Autoencoder, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(y_in, y1),
#             nn.ReLU(inplace=True),
#             nn.Linear(y1, y2),
#             nn.ReLU(inplace=True),
#             nn.Linear(y2, y3),
#             nn.ReLU(inplace=True),
#             nn.Linear(y3, y4),
#             #nn.Sigmoid()
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(y4, y3),
#             nn.ReLU(inplace=True),
#             nn.Linear(y3, y2),
#             nn.ReLU(inplace=True),
#             nn.Linear(y2, y1),
#             nn.ReLU(inplace=True),
#             nn.Linear(y1, y_in),
#             #nn.Sigmoid()
#         )
#         #self.soft = nn.Softmax(dim = 0)

#     def forward(self, x):
#         x = self.encoder(x)
#         #print(x.shape, '-'*3, type(x))
#         #print(tags.shape, '-'*3, type(tags))
#         x = self.decoder(x)
#         #x = self.soft(x)
#         return x


class Autoencoder(nn.Module):
    def __init__(self, y_in, y1, y2, y3, y4):
        super(Autoencoder, self).__init__()
        self.encoder1 = nn.Sequential(nn.Linear(y_in, y1), nn.ReLU(inplace=True))
        self.encoder2 = nn.Sequential(nn.Linear(y1, y2), nn.ReLU(inplace=True))
        self.encoder3 = nn.Sequential(nn.Linear(y2, y3), nn.ReLU(inplace=True))
        self.encoder4 = nn.Linear(y3, y4)

        self.decoder1 = nn.Sequential(nn.Linear(y4, y3), nn.ReLU(inplace=True))
        self.decoder2 = nn.Sequential(nn.Linear(y3, y2), nn.ReLU(inplace=True))
        self.decoder3 = nn.Sequential(nn.Linear(y2, y1), nn.ReLU(inplace=True))
        self.decoder4 = nn.Sequential(nn.Linear(y1, y_in))

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        d1 = self.decoder1(e4)
        d2 = self.decoder2(d1 + e3)
        d3 = self.decoder3(d2 + e2)
        x = self.decoder4(d3 + e1)
        return x

class RNNAutoencoder(nn.Module):
    def __init__(self, y_in, y1, y2, y3, y4):
        super(RNNAutoencoder, self).__init__()
        self.y_in = y_in  # d-dimensional packet feature
        self.hidden_size = y1  # RNN hidden state size
        self.embedding_dim = y2  # embedding dimension
        self.T = min(10, max(5, y_in // 4))  # 自适应时间步数
        
        # Embedding layer to reduce impact of different feature value ranges
        self.embedding = nn.Sequential(
            nn.Linear(y_in, self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Encoder RNN unit (U^E_RNN)
        self.encoder_rnn = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=1,  # 减少层数避免复杂度
            batch_first=True,
            dropout=0.0 if y1 < 128 else 0.1
        )
        
        # Decoder RNN unit (U^D_RNN)  
        self.decoder_rnn = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0.0 if y1 < 128 else 0.1
        )
        
        # Output projection layer to reconstruct original features
        self.output_projection = nn.Sequential(
            nn.Linear(self.hidden_size, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, y_in)
        )
        
        # Linear layer to project decoder hidden states back to embedding space
        self.hidden_to_embedding = nn.Linear(self.hidden_size, self.embedding_dim)
        
    def forward(self, x):
        original_shape_1d = False
        
        # Handle 1D input
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            original_shape_1d = True
            
        batch_size = x.size(0)
        
        # 处理空批次
        if batch_size == 0:
            if original_shape_1d:
                return torch.zeros(self.y_in)
            else:
                return torch.zeros(0, self.y_in)
        
        # 简化的序列创建：直接重复输入特征作为时间序列
        # 将输入复制T次作为时间序列：(batch_size, T, y_in)
        x_repeated = x.unsqueeze(1).repeat(1, self.T, 1)  # (batch_size, T, y_in)
        
        # Apply embedding layer to each time step
        batch_size, seq_len, feature_dim = x_repeated.shape
        x_flat = x_repeated.view(-1, feature_dim)  # (batch_size * T, y_in)
        embedded_flat = self.embedding(x_flat)  # (batch_size * T, embedding_dim)
        embedded_seq = embedded_flat.view(batch_size, seq_len, -1)  # (batch_size, T, embedding_dim)
        
        # Encoder: Pass embedded sequence through encoder RNN
        encoder_outputs, (encoder_hidden, encoder_cell) = self.encoder_rnn(embedded_seq)
        # encoder_outputs: (batch_size, T, hidden_size)
        
        # Use the last time step's hidden state as initial state for decoder
        decoder_hidden = encoder_hidden
        decoder_cell = encoder_cell
        
        # Decoder: Generate sequence step by step
        decoder_outputs = []
        decoder_input = embedded_seq[:, 0:1, :]  # Start with first time step (batch_size, 1, embedding_dim)
        
        for t in range(self.T):
            # Pass through decoder RNN
            decoder_output, (decoder_hidden, decoder_cell) = self.decoder_rnn(
                decoder_input, (decoder_hidden, decoder_cell)
            )
            decoder_outputs.append(decoder_output)
            
            # Use output as next input
            if t < self.T - 1:
                # Convert decoder output back to embedding space for next input
                decoder_input = self.hidden_to_embedding(decoder_output)  # (batch_size, 1, embedding_dim)
        
        # Collect all decoder outputs
        decoder_outputs = torch.cat(decoder_outputs, dim=1)  # (batch_size, T, hidden_size)
        
        # Project decoder outputs back to original feature space
        reconstructed_seq = self.output_projection(decoder_outputs)  # (batch_size, T, y_in)
        
        # Aggregate time steps back to original input format
        # 取时间维度的平均值作为最终输出
        output = reconstructed_seq.mean(dim=1)  # (batch_size, y_in)
        
        # Handle 1D output
        if original_shape_1d:
            output = output.squeeze(0)
            
        return output

# class GNNAutoencoder(nn.Module):
#     def __init__(self, y_in, y1, y2, y3, y4):
#         super(GNNAutoencoder, self).__init__()
#         self.y_in = y_in
        
#         # 编码器GNN层 - 简单的图卷积实现
#         self.encoder_gnn1 = nn.Linear(1, y1)  # 节点特征变换
#         self.encoder_gnn2 = nn.Linear(y1, y2)
#         self.encoder_gnn3 = nn.Linear(y2, y3)
#         self.encoder_gnn4 = nn.Linear(y3, y4)
        
#         # 解码器GNN层
#         self.decoder_gnn1 = nn.Linear(y4, y3)
#         self.decoder_gnn2 = nn.Linear(y3, y2)
#         self.decoder_gnn3 = nn.Linear(y2, y1)
#         self.decoder_gnn4 = nn.Linear(y1, 1)
        
#         self.relu = nn.ReLU()
        
#     def simple_graph_conv(self, x, adj, linear_layer):
#         """简单的图卷积操作：先邻接矩阵聚合，再线性变换"""
#         # x: (batch_size, num_nodes, features)
#         # adj: (batch_size, num_nodes, num_nodes) 邻接矩阵
        
#         # 图卷积：AXW (邻接矩阵 × 节点特征 × 权重)
#         aggregated = torch.bmm(adj, x)  # 邻接矩阵聚合
#         output = linear_layer(aggregated)  # 线性变换
#         return output
    
#     def create_adjacency_matrix(self, batch_size):
#         """创建简单的线性图邻接矩阵"""
#         adj = torch.zeros(batch_size, self.y_in, self.y_in)
        
#         for b in range(batch_size):
#             # 创建线性连接：每个节点连接相邻节点
#             for i in range(self.y_in):
#                 adj[b, i, i] = 1.0  # 自连接
#                 if i > 0:
#                     adj[b, i, i-1] = 1.0  # 连接前一个节点
#                 if i < self.y_in - 1:
#                     adj[b, i, i+1] = 1.0  # 连接后一个节点
            
#             # 归一化（度矩阵的逆）
#             degree = adj[b].sum(dim=1, keepdim=True)
#             degree[degree == 0] = 1  # 避免除0
#             adj[b] = adj[b] / degree
            
#         return adj
    
#     def forward(self, x):
#         original_shape_1d = False
        
#         # 如果输入是1D，增加batch维度
#         if len(x.shape) == 1:
#             x = x.unsqueeze(0)
#             original_shape_1d = True
            
#         batch_size = x.size(0)
        
#         # 将输入重塑为图节点：(batch_size, num_nodes=y_in, node_features=1)
#         x_nodes = x.unsqueeze(-1)  # (batch_size, y_in, 1)
        
#         # 创建邻接矩阵
#         adj = self.create_adjacency_matrix(batch_size)
        
#         # 编码过程（图卷积）
#         e1 = self.relu(self.simple_graph_conv(x_nodes, adj, self.encoder_gnn1))
#         e2 = self.relu(self.simple_graph_conv(e1, adj, self.encoder_gnn2))
#         e3 = self.relu(self.simple_graph_conv(e2, adj, self.encoder_gnn3))
#         e4 = self.simple_graph_conv(e3, adj, self.encoder_gnn4)  # 潜在表示
        
#         # 解码过程（带跳跃连接）
#         d1 = self.relu(self.simple_graph_conv(e4, adj, self.decoder_gnn1))
#         d2 = self.relu(self.simple_graph_conv(d1 + e3, adj, self.decoder_gnn2))  # 跳跃连接
#         d3 = self.relu(self.simple_graph_conv(d2 + e2, adj, self.decoder_gnn3))  # 跳跃连接
#         d4 = self.simple_graph_conv(d3 + e1, adj, self.decoder_gnn4)  # 跳跃连接
        
#         # 重塑回原始1D形状：(batch_size, y_in)
#         output = d4.squeeze(-1)
        
#         # 如果原输入是1D，输出也返回1D
#         if original_shape_1d:
#             output = output.squeeze(0)
            
#         return output

class GNNAutoencoder(nn.Module):
    def __init__(self, y_in, y1, y2, y3, y4):
        super(GNNAutoencoder, self).__init__()
        self.y_in = y_in
        self.node_feature_dim = 3  # [received bytes, sent bytes, duration]
        
        # 确保节点数量合理
        self.num_nodes = max(8, y_in // 8)  # 至少8个节点
        
        # 初始特征变换层 - 将输入特征转换为节点特征
        self.feature_transform = nn.Sequential(
            nn.Linear(y_in, self.num_nodes * self.node_feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 输入层：GCN层将节点特征变换到低维表示
        self.input_gcn = nn.Linear(self.node_feature_dim, y1)
        
        # 编码器块1：gPool + GCN
        self.encoder_gcn1 = nn.Linear(y1, y2)
        self.encoder_pool1 = nn.Linear(y1, 1)  # 简化的gPool
        
        # 编码器块2：gPool + GCN  
        self.encoder_gcn2 = nn.Linear(y2, y3)
        self.encoder_pool2 = nn.Linear(y2, 1)
        
        # 潜在表示层
        self.latent_gcn = nn.Linear(y3, y4)
        
        # 解码器块1：gUnpool + GCN
        self.decoder_unpool1 = nn.Linear(y4, y3)
        self.decoder_gcn1 = nn.Linear(y3, y3)  # 保持维度一致
        
        # 解码器块2：gUnpool + GCN
        self.decoder_unpool2 = nn.Linear(y3, y2) 
        self.decoder_gcn2 = nn.Linear(y2, y2)  # 保持维度一致
        
        # 输出层：重构原始特征
        self.decoder_final = nn.Linear(y2, y1)  # 将y2降维到y1
        self.output_gcn = nn.Linear(y1, y1)  # 保持维度一致
        self.output_projection = nn.Sequential(
            nn.Linear(self.num_nodes * y1, y_in),
            nn.ReLU()
        )
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def create_bipartite_graph(self, batch_size):
        """
        创建双向流量交互的二部图邻接矩阵
        """
        adj = torch.zeros(batch_size, self.num_nodes, self.num_nodes)
        
        for b in range(batch_size):
            # 将节点分为客户端节点(前半部分)和服务器节点(后半部分)
            client_nodes = self.num_nodes // 2
            
            # 创建二部图：客户端节点只连接服务器节点
            for i in range(client_nodes):
                for j in range(client_nodes, self.num_nodes):
                    # 双向连接，模拟数据包聚合
                    adj[b, i, j] = 1.0  # 客户端到服务器
                    adj[b, j, i] = 1.0  # 服务器到客户端
            
            # 添加自连接
            for i in range(self.num_nodes):
                adj[b, i, i] = 1.0
            
            # 度归一化
            degree = adj[b].sum(dim=1, keepdim=True)
            degree[degree == 0] = 1
            adj[b] = adj[b] / degree
            
        return adj
    
    def session_graph_conv(self, x, adj, linear_layer):
        """
        会话图卷积：考虑相邻数据包聚合的图卷积操作
        """
        # 邻接矩阵聚合：模拟相邻数据包在同一方向的聚合
        aggregated = torch.bmm(adj, x)  # (batch_size, num_nodes, features)
        
        # 线性变换
        output = linear_layer(aggregated)
        return output
    
    def gpool_operation(self, x, pool_layer):
        """
        简化的gPool操作：选择重要节点
        """
        # 计算节点重要性分数
        scores = pool_layer(x)  # (batch_size, num_nodes, 1)
        scores = torch.sigmoid(scores)
        
        # 加权节点特征
        pooled = x * scores  # 重要性加权
        return pooled, scores
    
    def forward(self, x):
        original_shape_1d = False
        
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            original_shape_1d = True
            
        batch_size = x.size(0)
        
        # 处理空批次
        if batch_size == 0:
            if original_shape_1d:
                return torch.zeros(self.y_in)
            else:
                return torch.zeros(0, self.y_in)
        
        # 特征提取：将输入转换为会话图节点特征
        transformed_features = self.feature_transform(x)  # (batch_size, num_nodes * 3)
        node_features = transformed_features.view(batch_size, self.num_nodes, self.node_feature_dim)
        
        # 创建二部图邻接矩阵
        adj = self.create_bipartite_graph(batch_size)
        
        # 输入层：GCN变换到低维表示
        x_input = self.relu(self.session_graph_conv(node_features, adj, self.input_gcn))
        skip1 = x_input  # 保存用于skip connection (batch_size, num_nodes, y1)
        
        # 编码器块1：gPool + GCN
        x_pool1, pool_scores1 = self.gpool_operation(x_input, self.encoder_pool1)
        e1 = self.relu(self.session_graph_conv(x_pool1, adj, self.encoder_gcn1))
        e1 = self.dropout(e1)
        skip2 = e1  # 保存用于skip connection (batch_size, num_nodes, y2)
        
        # 编码器块2：gPool + GCN
        x_pool2, pool_scores2 = self.gpool_operation(e1, self.encoder_pool2)
        e2 = self.relu(self.session_graph_conv(x_pool2, adj, self.encoder_gcn2))
        e2 = self.dropout(e2)
        skip3 = e2  # 保存用于skip connection (batch_size, num_nodes, y3)
        
        # 潜在表示
        latent = self.session_graph_conv(e2, adj, self.latent_gcn)  # (batch_size, num_nodes, y4)
        
        # 解码器块1：gUnpool + GCN + skip connection
        d1_unpool = self.relu(self.decoder_unpool1(latent))  # (batch_size, num_nodes, y3)
        d1 = self.relu(self.session_graph_conv(d1_unpool + skip3, adj, self.decoder_gcn1))  # 相加，维度匹配
        d1 = self.dropout(d1)  # (batch_size, num_nodes, y3)
        
        # 解码器块2：gUnpool + GCN + skip connection
        d2_unpool = self.relu(self.decoder_unpool2(d1))  # (batch_size, num_nodes, y2)
        d2 = self.relu(self.session_graph_conv(d2_unpool + skip2, adj, self.decoder_gcn2))  # 相加，维度匹配
        d2 = self.dropout(d2)  # (batch_size, num_nodes, y2)
        
        # 最终解码层：将y2降维到y1，然后与skip1相加
        d_final = self.relu(self.decoder_final(d2))  # (batch_size, num_nodes, y1)
        output_features = self.relu(self.session_graph_conv(d_final + skip1, adj, self.output_gcn))  # 相加，维度匹配
        
        # 将节点特征展平并投影回原始输入格式
        flattened_features = output_features.view(batch_size, -1)  # (batch_size, num_nodes * y1)
        final_output = self.output_projection(flattened_features)  # (batch_size, y_in)
        
        if original_shape_1d:
            final_output = final_output.squeeze(0)
            
        return final_output