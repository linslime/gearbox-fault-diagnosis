from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

y_pred = ['3','2','1','2','2','2','2','2','2','2','0','0','0','0','0','0','0','1','1','1','1','1','1','3','3','3','3','3','3','4','4','4','4','4'] # ['2','2','3','1','4'] # 类似的格式
y_true = ['1','4','2','2','2','2','2','2','2','2','0','0','0','0','0','0','0','1','1','1','1','1','1','3','3','3','3','3','3','4','4','4','4','4'] # 类似的格式
# 对上面进行赋值 # ['0','1','2','3','4'] # 类似的格式
# 对上面进行赋值

C = confusion_matrix(y_true, y_pred, labels=["0",'1','2','3','4']) # 可将'1'等替换成自己的类别，如'cat'。

plt.matshow(C, cmap=plt.cm.Blues) # 根据最下面的图按自己需求更改颜色
# plt.colorbar()

for i in range(len(C)):
    for j in range(len(C)):
        plt.annotate(C[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')

# plt.tick_params(labelsize=15) # 设置左边和上面的label类别如0,1,2,3,4的字体大小。
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.ylabel('预测值')
plt.xlabel('实际值')
# plt.ylabel('True label', fontdict={'family': 'Times New Roman', 'size': 20}) # 设置字体大小。
# plt.xlabel('Predicted label', fontdict={'family': 'Times New Roman', 'size': 20})
plt.show()
