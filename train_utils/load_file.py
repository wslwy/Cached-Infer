import pickle

if __name__ == "__main__":
    file = "results/vgg16_bn_trained_weights.pkl"
    with open(file, 'rb') as fi:
        loaded_data = pickle.load(fi)

    print("acc: {}".format(loaded_data["acc"]))