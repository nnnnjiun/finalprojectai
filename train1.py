import os
import cv2
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
left_path = os.path.join('imgs', 'left')    # 左邊的根目錄
right_path = os.path.join('imgs', 'right')    # 右邊的根目錄
# 所有图片的全路径
files = [os.path.join(left_path, left) for left in os.listdir(left_path)] + \
        [os.path.join(right_path, right) for right in os.listdir(right_path)]
X = []
y = [0] * len(os.listdir(left_path)) + [1] * len(os.listdir(right_path)) #最後輸出0是左1是右

for file in files:
    x = cv2.imread(file,0).reshape(-1)
    X.append(x)
    
    

# 2、拆分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# 3、定义模型
model = LogisticRegression(max_iter=500)
# 4、训练模型
model.fit(X_train, y_train)
# 5、评估模型
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(train_score, test_score)
# 保存模型
joblib.dump(model, 'left_or_right.m')
print(train_score ,"   ",test_score)