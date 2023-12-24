import os
import pickle

from my_utils.mul_exit import MulExit

if __name__ == "__main__":

    # 生成多出口类
    mul_exits = MulExit(state="global", model_type="resnet101", dataset="ucf101", exit_layers = [5, 9, 13, 17, 21, 25, 29, 33])
    # mul_exits.display_info()

    for bid in mul_exits.exit_layers:
        file = os.path.join("temp", f"{bid}_exit_net_weights.pkl")

        with open(file, 'rb') as fi:
            loaded_data = pickle.load(fi)

        acc = loaded_data["acc"]
        model_weight = loaded_data["weights"]
        print(f"bid: {bid}, acc: {acc}")
        print(model_weight["fc.weight"].shape)