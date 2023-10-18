import pickle
import numpy as np

# Loading results from pickle file.
with open("data/MW7_d2_pop_data.pkl", "rb") as inp:
    pops = pickle.load(inp)


# Global features
pops_global = pops[0]
pop_global = pops_global

# Pareto front.
pop_global.write_pf_to_csv("../../Test/Figures/pf.csv")

# Overall population
# pop_global.write_dec_to_csv("testing/global/dec.csv")
# pop_global.write_obj_to_csv("testing/global/obj.csv")
# pop_global.write_cons_to_csv("testing/global/cons.csv")
# pop_global.write_cv_to_csv("testing/global/cv.csv")
# pop_global.write_rank_to_csv("testing/global/rank_cons.csv")
# pop_global.write_rank_uncons_to_csv("testing/global/rank_uncons.csv")

# # Write bestranked objectives to csv
# obj = pop_global.extract_obj()
# ranks = pop_global.extract_rank()
# bestobj = obj[ranks == 1, :]
# np.savetxt("testing/global/bestobjs.csv", bestobj)


# # Nondominated front
# nondominated_global_cons = pop_global.extract_nondominated(constrained=True)
# pop_global.write_dec_to_csv("testing/global/nd_cons_dec.csv")
# nondominated_global_cons.write_obj_to_csv("testing/global/nd_cons_obj.csv")
# nondominated_global_cons.write_cv_to_csv("testing/global/nd_cons_cv.csv")
# nondominated_global_cons.write_rank_to_csv("testing/global/nd_cons_rank_cons.csv")
# nondominated_global_cons.write_rank_uncons_to_csv(
#     "testing/global/nd_cons_rank_uncons.csv"
# )


# RW features

# Overall population
# pops_rw = pops[1]
# pop_rw = pops_rw[0]
# pop_rw.write_dec_to_csv("testing/rw/dec.csv")
# pop_rw.write_obj_to_csv("testing/rw/obj.csv")
# pop_rw.write_cv_to_csv("testing/rw/cv.csv")
# pop_rw.write_rank_to_csv("testing/rw/rank_cons.csv")
# pop_rw.write_rank_uncons_to_csv("testing/rw/rank_uncons.csv")

# # Nondominated front
# nondominated_rw_cons = pop_rw.extract_nondominated(constrained=True)
# nondominated_rw_cons.write_dec_to_csv("testing/rw/dec.csv")
# nondominated_rw_cons.write_obj_to_csv("testing/rw/nd_cons_obj.csv")
# nondominated_rw_cons.write_cv_to_csv("testing/rw/nd_cons_cv.csv")
# nondominated_rw_cons.write_rank_to_csv("testing/rw/nd_cons_rank_cons.csv")
# nondominated_rw_cons.write_rank_uncons_to_csv("testing/rw/nd_cons_rank_uncons.csv")
