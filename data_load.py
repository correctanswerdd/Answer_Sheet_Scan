import pickle

dir = "../data/"
with open(dir + "options_three.pkl", "rb") as f:
    opt = pickle.load(f)
    print(opt.shape)

with open(dir + "options_four.pkl", "rb") as f:
    opt = pickle.load(f)
    print(opt.shape)

with open(dir + "options_seven.pkl", "rb") as f:
    opt = pickle.load(f)
    print(opt.shape)

