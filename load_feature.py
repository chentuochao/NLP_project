import pickle


save_folder = '../feature/feature_val2017.pkl'
with open(save_folder, 'rb') as f:
    save_json = pickle.load(f)

print(save_json)
