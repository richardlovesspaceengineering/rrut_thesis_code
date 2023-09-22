import pickle
import pandas as pd

with open("data/MW3_landscape_data.pkl", "rb") as inp:
    landscape = pickle.load(inp)

aggregated_table = landscape.make_aggregated_feature_table()
unaggregated_global_table = landscape.make_unaggregated_global_feature_table()
unaggregated_rw_table = landscape.make_unaggregated_rw_feature_table()

alsouly_table = landscape.extract_experimental_results()

comp_table = pd.concat(
    [alsouly_table.loc[:, alsouly_table.columns != "feature_D"], aggregated_table]
)

# Reset the index if needed
comp_table.reset_index(drop=True, inplace=True)
print(comp_table)
