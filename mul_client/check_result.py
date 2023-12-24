import os
import pickle

from my_utils.mul_exit import MulExit

if __name__ == "__main__":

    # 生成多出口类
    # mul_exits = MulExit(state="global", model_type="resnet101", dataset="ucf101", exit_layers = [5, 9, 13, 17, 21, 25, 29, 33])
    # mul_exits.display_info()

    file = "mul_client/results/resnet101_4c_woup.pkl"

    with open(file, 'rb') as fi:
        loaded_data = pickle.load(fi)

    print(loaded_data)
    print(f"'avg_time_list': {loaded_data['avg_time_list']}")
    print(f"'correct_ratio_list': {loaded_data['correct_ratio_list']}")

    print(file)
    for bid in range(4):
        print(f"client: {bid}, avg_time: {loaded_data['avg_time_list'][bid]}, accuracy: {loaded_data['correct_ratio_list'][bid]}")