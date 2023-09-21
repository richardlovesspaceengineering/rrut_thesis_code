import pickle

with open("data/MW3_landscape_data.pkl", "rb") as inp:
    landscape = pickle.load(inp)

print(landscape.instance_space)
print(landscape)
