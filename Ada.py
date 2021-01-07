import csv
from sklearn.ensemble import AdaBoostRegressor


def convert(age, sex, bmi, children, smoker, region):  # 将字符型数据转化为整型，并对类型变量sex, smoker, region进行one-hot编码
    property = []
    property.append(int(age))
    if sex == 'male':
        property.append(1)
    else:
        property.append(0)
    property.append(float(bmi))
    property.append(int(children))
    if smoker == 'yes':
        property.append(1)
    else:
        property.append(0)
    if region == 'southeast':
        property.append(1)
        property.append(0)
        property.append(0)
        property.append(0)
    elif region == 'southwest':
        property.append(0)
        property.append(1)
        property.append(0)
        property.append(0)
    elif region == 'northeast':
        property.append(0)
        property.append(0)
        property.append(1)
        property.append(0)
    else:
        property.append(0)
        property.append(0)
        property.append(0)
        property.append(1)
    return property


def convert_train():     # 读取训练集
    global X_train
    global Y_train
    with open('train1.csv', 'r') as file:  # 读取训练集到X_train, Y_train
        line = csv.reader(file)
        first = 1
        for row in line:
            if not first:
                X_train.append(convert(row[0], row[1], row[2], row[3], row[4], row[5]))
                Y_train.append(float(row[6]))
            first = 0


def convert_test():     # 读取测试集
    global X_test
    with open('test_sample.csv', 'r') as file:  # 读取测试集到X_test
        line = csv.reader(file)
        first = 1
        for row in line:
            if not first:
                X_test.append(convert(row[0], row[1], row[2], row[3], row[4], row[5]))
            first = 0


def print_result():     # 打印结果
    global Y_test
    with open('test_sample.csv', 'r') as file:  # 将结果按照原test_sample.csv的格式写入submission.csv
        line = csv.reader(file)
        with open('submission1.csv', 'w', newline='') as result_file:
            newline = csv.writer(result_file)
            first = 1
            number = 0
            for row in line:
                if not first:
                    temp = row
                    temp[-1] = Y_test[number]
                    number += 1
                    newline.writerow(temp)
                else:
                    newline.writerow(row)
                first = 0


def build_ada(n_estimators, learning_rate, loss):  # 构建Ada
    global X_train
    global Y_train
    model = AdaBoostRegressor(n_estimators=n_estimators, learning_rate=learning_rate, loss=loss, random_state=0)  # 实例化
    model.fit(X_train, Y_train)  # 用训练集数据训练模型
    return model


if __name__ == '__main__':
    X_train = []
    Y_train = []
    X_test = []
    convert_train()  # 读取训练集到X_train,Y_train
    convert_test()     # 读取测试集到X_test
    regression = build_ada(50, 1, 'linear')    # 参数分别为n_estimators, learning_rate, loss,(50,1,'linear')为默认值
    Y_test = regression.predict(X_test)  # 预测测试集数据,结果存至Y_test
    print_result()      # 打印结果到submission.csv
    train_R2 = regression.score(X_train, Y_train)  # 训练集R2
    print('train_R2=', end='')
    print(train_R2)
