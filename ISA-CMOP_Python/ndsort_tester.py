import pickle

# Loading results from pickle file.
with open("data/MW7_d2_pop_data.pkl", "rb") as inp:
    pops = pickle.load(inp)

pops_global = pops[0]
pops_global[0].write_obj_to_csv("testing/test_obj.csv")
pops_global[0].write_cv_to_csv("testing/test_cv.csv")
pops_global[0].write_rank_to_csv("testing/test_rank.csv")
pops_global[0].write_rank_uncons_to_csv("testing/test_rank_uncons.csv")
