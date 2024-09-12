import numpy as np
from scipy.io import arff
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, \
    silhouette_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, Normalizer

def kmeans_classifier(dataset_name = 'mc1', label_name = 'Defective'):
    # dataset_name = 'kc2'
    # label_name = 'problems'

    data, meta = arff.loadarff(f'dataset/{dataset_name.upper()}.arff')
    df = pd.DataFrame(data)
    print(df.head())

    # 二分类标签
    # df['problems'] = df['problems'].apply(lambda x: 1 if x == b'yes' else 0)
    df[label_name] = df[label_name].apply(lambda x: 1 if x == b'Y' else 0)

    # 分开特征和标签
    # 删除problems列，剩余的为特征
    X = df.drop(label_name, axis=1).values
    y_true = df[label_name].values

    # 对x进行标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # 对特征进行标准化
    print(X_scaled[1])
    # pca = PCA(n_components=16)  # 将特征降维到2维
    # X_pca = pca.fit_transform(X_scaled)

    # # 3. 使用轮廓系数选择最佳簇数
    # best_score = -1
    # best_k = 2  # 初始化簇数
    # for k in range(2, 10):
    #     kmeans = KMeans(n_clusters=k, random_state=42)
    #     kmeans.fit(X_pca)
    #     score = silhouette_score(X_pca, kmeans.labels_)
    #     if score > best_score:
    #         best_score = score
    #         best_k = k
    #
    # print(f"最佳簇数: {best_k}, 轮廓系数: {best_score}")

    # k-means聚类
    kmeans = KMeans(n_clusters=2, max_iter=1000, tol=1e-6)
    kmeans.fit(X_scaled)
    y_pred = kmeans.labels_
    # centroids, y_pred = kmeans_cosine(X_scaled, 2)


    # # 3. 使用高斯混合模型进行聚类
    # gmm = GaussianMixture(n_components=2, random_state=42)
    # gmm.fit(X_pca)
    # y_pred = gmm.predict(X_pca)

    # 调整聚类标签，防止相反
    if np.sum(y_true == y_pred) < np.sum(y_true != y_pred):
        y_pred = 1 - y_pred

    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)

    print(y_true)
    print(y_pred)

    # 精确率
    accuracy = (cm[1][1] + cm[0][0])/(cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

    # 计算ROC曲线和AUC值
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    print(f"ROC AUC: {roc_auc:.2f}")

    # 绘制ROC曲线
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

    return [round(metric, 2) for metric in [roc_auc, accuracy, f1, precision, recall]]


if __name__ == '__main__':
   result_list = kmeans_classifier()
   print(result_list)