import os
import pickle

from my_utils.mul_exit import MulExit

if __name__ == "__main__":

    # 生成多出口类
    # mul_exits = MulExit(state="global", model_type="resnet101", dataset="ucf101", exit_layers = [5, 9, 13, 17, 21, 25, 29, 33])
    # mul_exits.display_info()

    # file = "mul_client/results/caup_mule_wo_compare.pkl"
    file = "mul_client/results/mul_noniid_resnet101_ucf101.pkl"
    file = "mul_client/results/mul_noniid_resnet101_ucf101_02.pkl"
    
    with open(file, 'rb') as fi:
        loaded_data = pickle.load(fi)

    # print(loaded_data)
    # print(f"'avg_time_list': {loaded_data['avg_time_list']}")
    # print(f"'correct_ratio_list': {loaded_data['correct_ratio_list']}")


    for data in loaded_data:
        # print(data)
        if data["flag"] == "mule":
            print(f"exit layers: {data['exit_layers']}")
            print(f"thresholds: {data['ths']}")
            print(f"avg_times: {data['avg_times']}")
            print(f"correct_ratios: {data['correct_ratios']}")
            print("=======================================")
        elif data["flag"] == "cache":
            print(f"cache_size: {data['cache_size']}")
            print(f"cache_update: {data['cache_update']}")
            print(f"cache_add: {data['cache_add']}")
            print(f"cache_sign_id_list: {data['cache_sign_id_list']}")
            print(f"avg_time_list: {data['avg_time_list']}")
            print(f"correct_ratio_list: {data['correct_ratio_list']}")
            print("=======================================")

    # print(file)
    # for bid in range(4):
    #     print(f"client: {bid}, avg_time: {loaded_data['avg_time_list'][bid]}, accuracy: {loaded_data['correct_ratio_list'][bid]}")