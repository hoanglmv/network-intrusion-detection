root_path = "/home/myvh07/hoanglmv/Project/network-intrusion-detection/data"
data_path = "CIC2023"
import matplotlib.pyplot as plt
import os
import pandas as pd

path = os.path.join(root_path, data_path)
print(f"Data path: {path}")
# Lấy danh sách tên file trong thư mục data_path
file_list = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
for file_name in file_list:
    print(file_name)

# dùng vòng for để concat các file và lưu vào trong dataframe df_all, có thể điều chỉnh số file load vào
df= pd.DataFrame()
for i in range (0,10):
    file_path = os.path.join(path, file_list[i])
    df_temp = pd.read_csv(file_path)
    df= pd.concat([df, df_temp], ignore_index=True)    

df.info()
# print số nhãn trong cột label

print(df['label'].value_counts())

# Lọc các mẫu có nhãn chứa "DDOS", "DOS" hoặc "MITM" (không phân biệt hoa thường)
df = df[df['label'].str.contains('DDOS|DOS|MITM', case=False, regex=True)]

# print số nhãn trong cột label

print(df['label'].value_counts())

# import pandas as pd
# from sklearn.feature_selection import RFE
# from sklearn.tree import DecisionTreeClassifier
# # tách nhãn Label với features
# target_col = 'label'  
# X = df.drop(columns=[target_col])
# y = df[target_col]

# # Loại bỏ cột string không phải số
# numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
# X = X[numeric_features]

# print("Số cột numeric:", len(numeric_features))

# # khởi tạo mô hình
# clf = DecisionTreeClassifier(random_state=0)
# rfe = RFE(estimator=clf, n_features_to_select=20, step=1)

# # Huấn luyện RFE
# rfe.fit(X, y)   # ép nhãn về kiểu số nếu cần

# # danh sách feature quan trọng nhất
# selected_mask = rfe.support_
# selected_features = [numeric_features[i] for i, keep in enumerate(selected_mask) if keep]

# # lưu 
# df_selected = df[selected_features + [target_col]]

# print("Tổng số feature được chọn:", len(selected_features))
# print("Danh sách feature quan trọng nhất:")
# print(selected_features)
target_col = 'label'
# Danh sách các nhãn cần loại bỏ
labels_to_remove = ['DoS-HTTP_Flood', 'DDoS-HTTP_Flood', 'DDoS-SlowLoris']

# Danh sách các cột cần chọn (đặt trong dấu nháy)
selected_columns = [
    'flow_duration', 'Header_Length', 'Protocol Type', 'Rate',
    'fin_flag_number', 'syn_flag_number', 'psh_flag_number', 'syn_count',
    'urg_count', 'rst_count', 'HTTP', 'HTTPS', 'UDP', 'Min', 'IAT',
    'Number', 'Magnitue', 'Radius', 'Variance', 'Weight', 'label'
]

# Lọc giữ lại các hàng không nằm trong danh sách nhãn cần loại bỏ
df_selected = df[selected_columns]
df_selected = df_selected[~df_selected['label'].isin(labels_to_remove)]

# Kiểm tra lại các nhãn còn lại

import re
# giả sử df_selected đã có cột target_col (ví dụ 'label')
max_per_label = 10000
random_state = 0

#tạo 1 DataFrame gộp, mỗi label tối đa 20000 mẫu -----
df_capped = (
    df_selected
    .groupby(target_col, group_keys=False)
    .apply(lambda g: g.sample(n=min(len(g), max_per_label), random_state=random_state))
    .reset_index(drop=True)
)

print("Số mẫu mỗi label trong df_capped:")
print(df_capped[target_col].value_counts())

# Lưu file gộp
df_capped.to_csv("data/processed/data.csv", index=False)
print("Saved combined file: data/processed/data.csv")

