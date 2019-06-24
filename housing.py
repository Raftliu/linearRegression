"""
# object function: achieve a model to predict house price precisely
# date : "20190624"
# author : "raftliu"
"""
from __future__ import print_function

import paddle
import paddle.fluid as fluid
import math
import sys
import matplotlib
matplotlib.use('Agg')


# define a batch_size to train
batch_size = 128
num_epochs=10
use_cuda=False
# declaring the program execute place
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
'''executor可以接受传入的program，并根据feed map(输入映射表)和fetch list(结果获取表)
   向program中添加数据输入算子和结果获取算子。使用close()关闭该executor，
   调用run(...)执行program。
'''
exe = fluid.Executor(place)

# this function used to pre-process data such as normalize, shuffle
def preProcess(): #batch_reader and shuffle
    """ pre-process data"""
    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.uci_housing.train(), buf_size=500),
            batch_size=batch_size)

    test_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.uci_housing.train(), buf_size=500),
            batch_size=batch_size)
    return train_reader, test_reader

def network():
    """this function use to build network for train"""

    x = fluid.layers.data(name='x', shape=[13], dtype='float32') #define input data shape and dtype
    y = fluid.layers.data(name='y', shape=[1], dtype='float32') #define input data shape and dtype
    y_ = fluid.layers.fc(input=x, size=1, act=None) #define input and output, that is size.

    return x, y, y_

def backPropagation(y_predict, y):
    """this function use to compute loss and optimize"""
    main_program = fluid.default_main_program() #obtain default or global main function
    startup_program = fluid.default_startup_program() #obtain default or global start program
    cost = fluid.layers.square_error_cost(input=y_predict, label= y)# make label and predict to estimate variance
    avg_loss = fluid.layers.mean(cost) #obtain mean cost

    sgd_optimizer = fluid.optimizer.SGD(learning_rate=10e-6)
    sgd_optimizer.minimize(avg_loss)

    # 克隆main_program得到test_program
    # 有些operator在训练和测试之间的操作是不同的，例如batch_norm，使用参数for_test来区分该程序是用来训练还是用来测试
    # 该api不会删除任何操作符,请在backward和optimization之前使用
    test_program = main_program.clone(for_test=True)

    return avg_loss, main_program, test_program, startup_program

def train_test(executor, program, reader, feeder, fetch_list):
    accumulated = 1*[0]
    count = 0
    for data_test in reader():
        outs = executor.run(program=program, feed=feeder.feed(data_test),fetch_list=fetch_list)
        accumulated = [x_c[0] + x_c[1][0] for x_c in zip(accumulated, outs)] # 累加测试过程中的损失值
        count+=1  # 累加测试集中的样本数量
        return [x_d / count for x_d in accumulated]  # 计算平均损失

def main():
    ##% matplotlib inline
    params_dirname = "fit_a_line.inference.model"
    x, y, y_predict = network()
    feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
    avg_loss, main_program, test_program, startup_program = backPropagation(y_predict, y)
    exe.run(startup_program)
    train_prompt = "train cost"
    test_prompt = "test cost"
    from paddle.utils.plot import Ploter
    plot_prompt = Ploter(train_prompt, test_prompt)
    step = 0

    exe_test = fluid.Executor(place)
    train_reader, test_reader = preProcess()
    for pass_id in range(num_epochs):
        for data_train in train_reader():
            avg_loss_value, = exe.run(main_program,
                                      feed=feeder.feed(data_train),
                                      fetch_list=[avg_loss])
            if step % 10 == 0:  # 每10个批次记录并输出一下训练损失
                plot_prompt.append(train_prompt, step, avg_loss_value[0])
                plot_prompt.plot()
                print("%s, Step %d, Cost %f" %
                      (train_prompt, step, avg_loss_value[0]))
            if step % 100 == 0:  # 每100批次记录并输出一下测试损失
                test_metics = train_test(executor=exe_test,
                                         program=test_program,
                                         reader=test_reader,
                                         fetch_list=[avg_loss.name],
                                         feeder=feeder)
                plot_prompt.append(test_prompt, step, test_metics[0])
                plot_prompt.plot()
                print("%s, Step %d, Cost %f" %
                      (test_prompt, step, test_metics[0]))
                if test_metics[0] < 10.0:  # 如果准确率达到要求，则停止训练
                    break

            step += 1

            if math.isnan(float(avg_loss_value[0])):
                sys.exit("got NaN loss, training failed.")

            # 保存训练参数到之前给定的路径中
            if params_dirname is not None:
                fluid.io.save_inference_model(params_dirname, ['x'], [y_predict], exe)

if __name__ == "__main__":
    main()