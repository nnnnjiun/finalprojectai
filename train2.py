import os
import cv2
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
path_0 = os.path.join('imgs', '0')    
path_1 = os.path.join('imgs', '1')    
path_2 = os.path.join('imgs', '2')
path_3 = os.path.join('imgs', '3')
# 所有图片的全路径
files = [os.path.join(path_0, zero) for zero in os.listdir(path_0)] + \
        [os.path.join(path_1, one) for one in os.listdir(path_1)]+ \
        [os.path.join(path_2, two) for two in os.listdir(path_2)]+ \
        [os.path.join(path_3, three) for three in os.listdir(path_3)]
X = []
y = [0] * len(os.listdir(path_0)) + [1] * len(os.listdir(path_1)) + [2] * len(os.listdir(path_2)) +[3] * len(os.listdir(path_3))

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
joblib.dump(model, 'long.m')
print(train_score ,"   ",test_score)