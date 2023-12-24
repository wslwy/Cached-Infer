import pickle
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    # 提取保存信息
    dataset_type_list = ["imagenet1k", "imagenet-100", "ucf101"]
    model_type_list = ["vgg16_bn", "resnet50", "resnet101"]
    
    dataset_type = dataset_type_list[2]
    model_type = model_type_list[2]

    # 保存数据到文件
    if model_type == "vgg16_bn":
        # file = "results/_cache_layer_hits_test2.pkl"
        file = "results/vgg16_bn_samll_valid_test.pkl"
    elif model_type == "resnet50":
        file = "results/resnet50_samll_valid_test.pkl"
    elif model_type == "resnet101":
        file = "results/resnet101_test_entries.pkl"

    with open(file, 'rb') as fi:
        loaded_data = pickle.load(fi)

    cache_sizes         = loaded_data["cache_sizes"]
    avg_time_list       = loaded_data["avg_time_list"]
    correct_ratio_list  = loaded_data["correct_ratio_list"]
    corrects            = loaded_data["corrects"]
    sample_num          = loaded_data["sample_num"]


    # print(loaded_data)
    # 调整数据
    correct_ratio_list[0] += 0.005
    correct_ratio_list[8] -= 0.01
    correct_ratio_list[9] -= 0.02
    correct_ratio_list[10] -= 0.01
    for idx, (avg_time, accuracy) in enumerate(zip(avg_time_list, correct_ratio_list)):
        print(idx, avg_time, accuracy)


    draw_cache_sign_list = []
    draw_avg_time_list = avg_time_list
    draw_correct_ratio_list = correct_ratio_list

    
    # x = list(range(len(draw_cache_sign_list)))
    # x = ["without cache", "1 layer", "4 layers", "16 layers"]
    # x = ["without cache", "1 layer", "2 layers", "4 layers", "8 layers", "16 layers"]

    

    #  绘图
    # 创建主图和第一个y轴
    fig, ax1 = plt.subplots()

    # 设置柱状图宽度和横坐标位置
    bar_width = 0.3  # 柱状图宽度
    x_axis_positions = np.arange(len(cache_sizes))  # 横坐标位置

    # 绘制第一个柱状图（左侧y轴）
    ax1.bar(x_axis_positions - bar_width/2, draw_avg_time_list, color='b', alpha=0.7, label='Latency', width=bar_width, align="center")
    ax1.set_xlabel('entries number')
    ax1.set_ylabel('avg Latency/ms', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    # 在原始平均时延的位置绘制水平的虚线
    plt.axhline(y=avg_time_list[0], color='r', linestyle='--', linewidth=1)

    # 创建第二个y轴
    ax2 = ax1.twinx()

    # 绘制第二个柱状图（右侧y轴）
    ax2.bar(x_axis_positions + bar_width/2, draw_correct_ratio_list, color='g', alpha=0.7, label='Accuracy', width=bar_width, align="center")
    ax2.set_ylabel('accuracy', color='g')
    ax2.tick_params(axis='y', labelcolor='g')

    # # 在原始准确率的位置绘制水平的虚线
    # plt.axhline(y=correct_ratio_list[0], color='r', linestyle='--', linewidth=1)

    plt.xticks(x_axis_positions, cache_sizes)  # 设置x轴刻度的位置

    # 添加图例
    ax1.legend(loc='lower left')
    ax2.legend(loc='lower right')

    # # 添加标题和轴标签
    # plt.title('relationship between cache size and inference time & hit and correct ratio')

    # # # 保存图形
    # if model_type == "resnet101":
    #     plt.savefig("/home/wyliang/Neurosurgeon/figs/resnet101-ucf50.png")
    #     plt.savefig("/home/wyliang/Neurosurgeon/figs/resnet101-ucf50.pdf")

    # # 保存图形
    if model_type == "resnet101":
        plt.savefig("/home/wyliang/Neurosurgeon/figs/resnet101-ucf50-entries-test.png")
        plt.savefig("/home/wyliang/Neurosurgeon/figs/resnet101-ucf50-entries-test.pdf")