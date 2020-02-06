import pickle

with open("options_three.pkl", "rb") as f:
    opt = pickle.load(f)
    print(opt.shape)

with open("options_four.pkl", "rb") as f:
    opt = pickle.load(f)
    print(opt.shape)

with open("options_seven.pkl", "rb") as f:
    opt = pickle.load(f)
    print(opt.shape)

