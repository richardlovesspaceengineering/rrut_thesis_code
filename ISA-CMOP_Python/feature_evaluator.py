import pickle
import pandas as pd
from features.FitnessAnalysis import MultipleFitnessAnalysis
from features.RandomWalkAnalysis import MultipleRandomWalkAnalysis
from features.LandscapeAnalysis import LandscapeAnalysis

# with open("data/MW3_landscape_data.pkl", "rb") as inp:
#     landscape = pickle.load(inp)

# Loading results from pickle file.
with open("data/MW7_d2_pop_data.pkl", "rb") as inp:
    pops = pickle.load(inp)

pops_global = pops[0]
pops_rw = pops[1]

# Global features.
global_features = MultipleFitnessAnalysis(pops_global)
global_features.eval_features_for_all_populations()

# Random walk features.
rw_features = MultipleRandomWalkAnalysis(pops_rw)
rw_features.eval_features_for_all_populations()

# Combine all features.
landscape = LandscapeAnalysis(global_features, rw_features)
landscape.extract_feature_arrays()
landscape.aggregate_features(YJ_transform=False)
landscape.extract_features_vector()
landscape.map_features_to_instance_space()

# %%

# Saving results to pickle file.
with open("data/MW3_landscape_data.pkl", "wb") as outp:
    pickle.dump(landscape, outp, -1)

# Apply YJ transformation
# landscape.aggregate_features(True)

aggregated_table = landscape.make_aggregated_feature_table()
unaggregated_global_table = landscape.make_unaggregated_global_feature_table()
unaggregated_rw_table = landscape.make_unaggregated_rw_feature_table()

alsouly_table = landscape.extract_experimental_results()

# landscape.aggregate_features(True)
# aggregated_YJ_table = landscape.make_aggregated_feature_table()


comp_table = pd.concat(
    [
        alsouly_table.loc[:, alsouly_table.columns != "feature_D"],
        aggregated_table,
        # aggregated_YJ_table,
    ]
)

# Reset the index if needed
comp_table.reset_index(drop=True, inplace=True)
comp_table = comp_table.transpose()
print(comp_table)
