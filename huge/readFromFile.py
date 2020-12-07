#-*-coding:utf-8-*-
# 读取xls，写入二进制文件
import xlrd
import numpy as np
import matplotlib.pyplot as plt

def getDataFromFile(path, sh_num = 0, opath = ".\\odom.bin"):
    wb = xlrd.open_workbook(path)
    sheet = wb.sheets()[sh_num]
    ndim = len(sheet.col(0))
    data = np.zeros((ndim - 1, ndim - 1))
    for i in range(1, ndim):
        col = sheet.row(i)
        data[i - 1, :] = np.array([col[j].value for j in range(1, ndim)])
        data[i - 1, i - 1] = 1e8       # 防止自到达
    # print("Data extracted from %s. ndim = %d"%(path, ndim))
    # print(data)
    data.tofile(opath)

def getCoordinates(path, opath):
    wb = xlrd.open_workbook(path)
    sh = wb.sheets()[0]
    ndim = len(sh.col(0))
    lons = sh.col(1)
    lats = sh.col(2)
    data = np.zeros((ndim - 1, 2))
    for i in range(1, ndim):
        data[i - 1, 0] = lons[i].value / 2
        data[i - 1, 1] = lats[i].value
    plt.scatter(data[:, 0], data[:, 1], c = 'k', s = 8)
    data.tofile(opath)
    return data

def calcCost(adjs, path):
    cost = 0
    for i in range(len(path) - 1):
        cost += adjs[path[i]][path[i + 1]]
    return cost

if __name__ == "__main__":
    plt.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 16
    order = ([12, 7, 42, 43, 44, 52, 100, 99, 102, 101, 103, 9, 3, 4, 2, 0, 1, 5, 8, 105, 104, 97, 98, 95, 96, 94, 89, 93, 74, 70, 90, 91, 92, 76, 69, 68, 66, 67, 72, 73, 71, 75, 27, 28, 19, 17, 18, 24, 25, 20, 21, 26, 22, 23, 10, 11, 30, 29, 106, 41, 40, 34, 33, 35, 37, 36, 38, 
    39, 32, 31, 15, 13, 14, 16, 65, 62, 61, 60, 53, 57, 58, 59, 82, 81, 80, 78, 79, 77, 85, 84, 88, 86, 83, 56, 87, 55, 54, 63, 64, 51, 46, 47, 45, 49, 48, 50, 6, 12])
    adjs = np.fromfile(".\\odom.bin")
    adjs = adjs.reshape((107, 107))
    getDataFromFile(".\\odom.xls", 0)
    data = getCoordinates(".\\coordinate.xlsx", ".\\pos.bin")
    path = data[order]
    print("Final cost:", calcCost(adjs, order), " with length:", len(order))
    plt.title('蚁群算法结果')
    plt.plot(path[:, 0], path[:, 1], c = 'k')
    # for i in range(len(order)):
        # plt.annotate("%d"%(order[i]), xy = (path[i, 0] + 0.01, path[i, 1] + 0.01), xytext = (path[i, 0] + 0.01, path[i, 1] + 0.01))
    plt.scatter(path[:, 0], path[:, 1], c = 'k', s = 7, label = '城市位置')
    plt.grid(axis = 'both')
    plt.xlabel('经度')
    plt.ylabel('纬度')
    plt.legend()
    plt.show()
