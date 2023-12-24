import pickle
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    # 提取保存信息
    model_type_list = ["vgg16_bn", "resnet50", "resnet101"]
    
    model_type = model_type_list[2]

    # 加载数据文件
    if model_type == "vgg16_bn":
        # file = "results/_cache_layer_hits_test2.pkl"
        file = "results/vgg16_bn_ee_test.pkl"
    elif model_type == "resnet50":
        file = "results/resnet50_ee_test.pkl"
    elif model_type == "resnet101":
        file = "results/resnet101_ee_test.pkl"

    with open(file, 'rb') as fi:
        loaded_data = pickle.load(fi)

    exit_layers = loaded_data["exit_layers"]
    avg_time_list = loaded_data["avg_time_list"]


    norm_avg_time_list = [t / avg_time_list[-1] for t in avg_time_list]

    print(loaded_data)

    xs = exit_layers

    # 绘图
    # 创建图像和轴
    fig, ax1 = plt.subplots()
    

    # # 绘制准确率图
    ax1.plot(exit_layers, norm_avg_time_list, color='g', label='norm avg time')
    # ax2.tick_params(axis='y', labelcolor='g')

    # # 绘制全准确率图
    # ratios = [0.877] * len(cache_size_list)
    # ax2.plot(cache_size_list, ratios, color='y', label='no-cache correct ratio')
    # ax2.tick_params(axis='y', labelcolor='g')

    # 显示图例
    ax1.legend(loc='upper left')
    # ax2.legend(loc='upper right')

    # 添加标题和轴标签
    # plt.title('relationship between cache size and inference time & hit and correct ratio')

    # 保存图形
    if model_type == "vgg16_bn":
        # file = "results/_cache_layer_hits_test2.pkl"
        file = "results/vgg16_bn_ee_test.pkl"
    elif model_type == "resnet50":
        file = "results/resnet50_ee_test.pkl"
    elif model_type == "resnet101":
        plt.savefig("figs/resnet101_ee_test.png")
        plt.savefig("figs/resnet101_ee_test.pdf")
    

    # x = list(range(len(draw_cache_sign_list)))
    # x = ["without cache", "1 layer cache", "2 layers cache", "4 layers cache", "8 layers cache", "13 layers cache",]
    
    

    # #  绘图
    # # 创建主图和第一个y轴
    # fig, ax1 = plt.subplots()

    # # 设置柱状图宽度和横坐标位置
    # bar_width = 0.3  # 柱状图宽度
    # x_axis_positions = np.arange(len(xs))  # 横坐标位置

    # # 绘制第一个柱状图（左侧y轴）
    # ax1.bar(x_axis_positions - bar_width/2, hit_ratios, color='b', alpha=0.7, label='hit ratio', width=bar_width, align="center")
    # ax1.set_xlabel('layers')
    # ax1.set_ylabel('hit ratio', color='b')
    # ax1.tick_params(axis='y', labelcolor='b')

    # # 创建第二个y轴
    # ax2 = ax1.twinx()

    # # 绘制第二个柱状图（右侧y轴）
    # ax2.bar(x_axis_positions + bar_width/2, correct_ratios, color='r', alpha=0.7, label='correct ratio', width=bar_width, align="center")
    # ax2.set_ylabel('correct ratio', color='g')
    # ax2.tick_params(axis='y', labelcolor='g')


    # # # 在原始准确率的位置绘制水平的虚线
    # # plt.axhline(y=correct_ratio_list[0], color='r', linestyle='--', linewidth=1)

    # plt.xticks(x_axis_positions, xs)  # 设置x轴刻度的位置

    # # 添加图例
    # ax1.legend(loc='upper left')
    # ax2.legend(loc='upper right')

    # # # 添加标题和轴标签
    # # plt.title('relationship between cache size and inference time & hit and correct ratio')

    # # 保存图形
    # if model_type == "resnet50":
    #     plt.savefig("/home/wyliang/Neurosurgeon/figs/resnet50_layer_hitAndCorrectRatio.png")
    #     plt.savefig("/home/wyliang/Neurosurgeon/figs/resnet50_layer_hitAndCorrectRatio.pdf")
    # elif model_type == "resnet101":
    #     plt.savefig("/home/wyliang/Neurosurgeon/figs/resnet101_layer_hitAndCorrectRatio.png")
    #     plt.savefig("/home/wyliang/Neurosurgeon/figs/resnet101_layer_hitAndCorrectRatio.pdf")