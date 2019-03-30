# -*- "utf-8" -*-

'''
使用 matplotlib.pyplot 绘制 ROC 曲线
并且求出 ROC 曲线与 X 轴构成的面积
'''

import numpy as np
import matplotlib.pyplot as plt
import time
import os

class ROC(object):
    @staticmethod
    def roc(scores, labels, draw=False):
        '''
        scores: ndarray, (1, n_samples), 每个测试样本属于正样本的概率
        labels: ndarray, (1, n_samples), 每个测试样本真正的标签
        draw: bool, 是否绘制 ROC 曲线
        return: float, ROC 曲线下方的面积, Area under the Curve of ROC
        '''
        # 得到坐标点
        points_x = []
        points_y = []
        for threshold in scores:
            predicts = np.where(scores>=threshold, 1, 0)  # 大于等于 threshold 为正样本
            TN = 0  # True negative
            FP = 0  # False positive
            FN = 0  # False negative
            TP = 0  # True positive
            for i in range(scores.shape[0]):
                if predicts[i] == 0 and labels[i] == 0:
                    TN += 1
                elif predicts[i] == 1 and labels[i] == 0:
                    FP += 1
                elif predicts[i] == 0 and labels[i] == 1:
                    FN += 1
                elif predicts[i] == 1 and labels[i] == 1:
                    TP += 1
            TPR = TP/(TP+FN)  # TP Rate, Y-axis
            FPR = FP/(FP+TN)  # FP Rate, X-axis
            points_x.append(FPR)
            points_y.append(TPR)
        
        # 绘图
        if draw:
            plt.figure()
            plt.scatter(points_x, points_y, s=20, c='r', marker='x')  # 散点绘制
            plt.plot(points_x, points_y, color='black', linestyle='--')  # 虚线绘制
            plt.xlabel('False Positive Rate')  # 设置 x 轴名称
            plt.ylabel('True Positive Rate')   # 设置 y 轴名称
            plt.tick_params(top=True,bottom=True,left=True,right=True)  # 四周刻度线全开
            # plt.show()  # 展示图片时程序将阻塞, 建议保存图片后使用图片查看器查看
            if not os.path.exists('./images/'):  # 如果文件夹不存在则创建
                os.mkdir('images')
            t = time.localtime()  # 用当前时间作文件名
            plt.savefig("./images/{}-{}-{}_{}-{}-{}.jpg".format(t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec))  # 保存图片

        # 计算曲线下面积
        indices = sorted(range(len(points_y)), key=points_y.__getitem__)  # 得到 y 值按从小到大排列时的索引序列
        begin = indices[0]
        end = indices[0]
        auc = 0
        for i in range(1, len(indices)):
            if points_y[indices[i]] != points_y[indices[i-1]]:
                end = indices[i-1]
                auc += (points_x[end]-points_x[begin]) * points_y[end]
                begin = indices[i]
        end = indices[-1]
        auc += (points_x[end]-points_x[begin]) * points_y[end]  # 加上最后一段与 x 轴构成的面积

        return auc

if __name__ == '__main__':
    scores = np.array([0.9,0.8,0.7,0.6,0.55,0.54,0.53,0.52,0.51,0.505,0.4,0.39,0.38,0.37,0.36,0.35,0.34,0.33,0.30,0.1])
    labels = np.array([1,1,0,1,1,1,0,0,1,0,1,0,1,0,0,0,1,0,1,0])
    auc = ROC.roc(scores, labels, draw=True)
    print(auc)
