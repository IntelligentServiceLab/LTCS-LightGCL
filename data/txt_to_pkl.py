import scipy.sparse as sp
import numpy as np
import pickle


def load_data(file_path):
    """加载用户-物品交互数据，从指定文件读取

    参数:
        file_path (str): 包含用户-物品交互数据的文件路径。

    返回:
        tuple: 三个列表（rows, cols, data），分别代表稀疏矩阵中的行、列和数据部分。
    """
    rows, cols, data = [], [], []  # 初始化行、列和数据列表，用于存储交互信息
    with open(file_path, 'r') as file:  # 打开文件进行读取
        for line in file:  # 遍历文件中的每一行
            parts = line.strip().split()  # 去除多余的空格并将行数据按空格分割
            if len(parts) < 2:  # 如果该行数据不足（至少包含用户和一个物品），则跳过此行
                continue
            user = int(parts[0])  # 第一部分为用户ID
            items = map(int, parts[1:])  # 后续部分为用户交互的物品ID
            for item in items:  # 遍历每个物品ID
                rows.append(user)  # 将用户ID添加到行列表中
                cols.append(item)  # 将物品ID添加到列列表中
                data.append(1.0)  # 交互值设置为1.0，表示用户与物品之间有交互
    return rows, cols, data  # 返回三个列表，分别表示用户、物品和交互值


def save_sparse_matrix(file_path, rows, cols, data):
    """创建稀疏矩阵并保存为.pkl文件

    参数:
        file_path (str): 保存稀疏矩阵的文件路径。
        rows (list): 行索引（用户ID列表）。
        cols (list): 列索引（物品ID列表）。
        data (list): 数据值（交互值列表）。
    """
    # 使用COO格式创建稀疏矩阵
    matrix = sp.coo_matrix((data, (rows, cols)), dtype=np.float32)

    # 将稀疏矩阵保存到pickle文件中
    with open(file_path, 'wb') as f:
        pickle.dump(matrix, f)
    print(f"Saved sparse matrix to {file_path}")  # 打印保存成功的消息


# 加载训练数据和测试数据
train_rows, train_cols, train_data = load_data('train.txt')  # 从文件加载训练集数据
test_rows, test_cols, test_data = load_data('test.txt')  # 从文件加载测试集数据

# 打印训练集和测试集的数据长度
print(len(train_rows), len(train_cols), len(train_data), len(test_data), len(test_data))

# 将训练集和测试集的稀疏矩阵分别保存为pickle文件
save_sparse_matrix('trnMat.pkl', train_rows, train_cols, train_data)  # 保存训练集矩阵
save_sparse_matrix('tstMat.pkl', test_rows, test_cols, test_data)  # 保存测试集矩阵
