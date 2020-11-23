#-*-coding:utf-8-*-
# 读取xls，写入二进制文件
import xlrd
import numpy as np

def getDataFromFile(path, sh_num = 0, opath = ".\\odom.bin"):
    wb = xlrd.open_workbook(path)
    sheet = wb.sheets()[sh_num]
    ndim = len(sheet.col(0))
    data = np.zeros((ndim - 1, ndim - 1))
    for i in range(1, ndim):
        col = sheet.row(i)
        data[i - 1, :] = np.array([col[j].value for j in range(1, ndim)])
        data[i - 1, i - 1] = 1e5       # 防止自到达
    print("Data extracted from %s. ndim = %d"%(path, ndim))
    print(data)
    data.tofile(opath)

if __name__ == "__main__":
    getDataFromFile(".\\odom.xls", 0)
