import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, accuracy_score, f1_score, recall_score, confusion_matrix, \
    classification_report, roc_curve
from imblearn.over_sampling import SMOTE
from wide_cnn import Wide_CNN
from function import self_define_cnn_kernel_process
import seaborn as sns

if __name__ == '__main__':
    print('Read data and label')
    data = pd.read_csv('data/filtered_after_preprocess_data.csv')
    label = pd.read_csv('data/filtered_label.csv')

    print('Performing data balancing with SMOTE')
    smote = SMOTE(random_state=42)
    X_resampled, Y_resampled = smote.fit_resample(data.values, label.flag.values)

    print('Split Train dataset and Test dataset with ratio 70%')
    for valr in [0.7]:
        print('Train split ratio: %.2f' % valr)

        # 分割训练集和测试集
        X_train_wide, X_test_wide, Y_train, Y_test = train_test_split(X_resampled, Y_resampled,
                                                                      test_size=1 - valr, random_state=42)

        # 重新调整形状以适配深特征和宽特征
        X_train_deep = X_train_wide.reshape(X_train_wide.shape[0], 1, -1, 7).transpose(0, 2, 3, 1)
        X_test_deep = X_test_wide.reshape(X_test_wide.shape[0], 1, -1, 7).transpose(0, 2, 3, 1)

        weeks, days, channel = X_train_deep.shape[1], X_train_deep.shape[2], 1
        wide_len = X_train_wide.shape[1]

        print(X_train_wide.shape, X_train_deep.shape)
        print(X_test_wide.shape, X_test_deep.shape)

        # 预处理数据
        X_train_pre = self_define_cnn_kernel_process(X_train_deep)
        X_test_pre = self_define_cnn_kernel_process(X_test_deep)

        # Run the model for 10 rounds
        for i in range(10):
            print('Round: %d' % i)
            model = Wide_CNN(weeks, days, channel, wide_len)  # 假设 Wide_CNN 是定义好的模型类

            if i == 0:
                print(model.summary())

            # 训练模型
            model.fit([X_train_wide, X_train_pre], Y_train, batch_size=64, epochs=50, verbose=1,
                      validation_data=([X_test_wide, X_test_pre], Y_test))

            # 在每次训练完成后进行评估
            preds = model.predict([X_test_wide, X_test_pre])
            preds_binary = (preds > 0.5).astype(int)

            # 计算 AUC、Precision、Accuracy、F1 Score、Recall
            auc = roc_auc_score(Y_test, preds)
            precision = precision_score(Y_test, preds_binary)
            accuracy = accuracy_score(Y_test, preds_binary)
            f1 = f1_score(Y_test, preds_binary)
            recall = recall_score(Y_test, preds_binary)

            auc = roc_auc_score(Y_test, preds)
            fpr, tpr, _ = roc_curve(Y_test, preds)

            # 绘制 ROC 曲线
            plt.figure(figsize=(6, 4))
            plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})', linewidth=2)
            plt.plot([0, 1], [0, 1], 'k--')  # 对角线
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            plt.show()

            # 计算 MAP@100
            # 加权精确度、召回率、F1 分数
            weighted_precision = precision_score(Y_test, preds_binary, average='weighted')
            weighted_recall = recall_score(Y_test, preds_binary, average='weighted')
            weighted_f1 = f1_score(Y_test, preds_binary, average='weighted')

            print(f"Weighted Precision: {weighted_precision:.4f}")
            print(f"Weighted Recall: {weighted_recall:.4f}")
            print(f"Weighted F1 Score: {weighted_f1:.4f}")

            # 输出 AUC
            print(f"AUC: {auc:.4f}")
            print(f"AUC: {auc:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print(f"Recall: {recall:.4f}")
            conf_matrix = confusion_matrix(Y_test, preds_binary)
            print("Confusion Matrix:")
            print(conf_matrix)

            plt.figure(figsize=(6, 4))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Class 0", "Class 1"],
                        yticklabels=["Class 0", "Class 1"])
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.title("Confusion Matrix")
            plt.show()

            report = classification_report(Y_test, preds_binary, target_names=["Class 0", "Class 1"])
            print("Classification Report:")
            print(report)
