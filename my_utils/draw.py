import pickle
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # 提取保存信息
    file = "results/second_valid_small_test.pkl"
    with open(file, 'rb') as fi:
        loaded_data = pickle.load(fi)

    cache_size_list     = loaded_data["cache_size_list"]
    avg_time_list       = loaded_data["avg_time_list"]
    hit_ratio_list      = loaded_data["hit_ratio_list"]
    correct_ratio_list  = loaded_data["correct_ratio_list"]

    print(loaded_data)
    #  绘图
    # 创建图像和轴
    fig, ax1 = plt.subplots()

    # 绘制第一组数据（左y轴）
    ax1.set_xlabel('cache size')
    ax1.set_ylabel('avg time/ms', color='b')
    ax1.plot(cache_size_list, avg_time_list, color='b', label='avg time')
    ax1.tick_params(axis='y', labelcolor='b')

    # 创建第二个y轴
    ax2 = ax1.twinx()

    # 绘制第二组数据（右y轴）
    ax2.set_ylabel('hit and correct ratio', color='r')
    ax2.plot(cache_size_list, hit_ratio_list, color='r', label='hit ratio')
    ax2.tick_params(axis='y', labelcolor='g')

    # 绘制准确率图
    ax2.plot(cache_size_list, correct_ratio_list, color='g', label='correct ratio')
    ax2.tick_params(axis='y', labelcolor='g')

    # 绘制全准确率图
    ratios = [0.877] * len(cache_size_list)
    ax2.plot(cache_size_list, ratios, color='y', label='no-cache correct ratio')
    ax2.tick_params(axis='y', labelcolor='g')

    # 显示图例
    ax1.legend(loc='upper left')
    ax2.legend(loc='lower right')

    # 添加标题和轴标签
    plt.title('relationship between cache size and inference time & hit and correct ratio')

    # 保存图形
    plt.savefig("/home/wyliang/Neurosurgeon/figs/cacheSize_avgTime_hitRatio.png")