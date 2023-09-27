# %%

import pickle
import pandas as pd
from features.FitnessAnalysis import MultipleFitnessAnalysis
from features.RandomWalkAnalysis import MultipleRandomWalkAnalysis
import matplotlib.pyplot as plt
import seaborn as sns
from features.LandscapeAnalysis import LandscapeAnalysis

# with open("data/MW3_landscape_data.pkl", "rb") as inp:
#     landscape = pickle.load(inp)

# Loading results from pickle file.
with open("data/MW7_d2_pop_data.pkl", "rb") as inp:
    pops = pickle.load(inp)


# Global features.
global_features = MultipleFitnessAnalysis(pops)
global_features.eval_features_for_all_populations()

# Random walk features.
rw_features = MultipleRandomWalkAnalysis(pops)
rw_features.eval_features_for_all_populations()

# %%

# Plot objectives
# Create a 5x6 grid of subplots
fig, ax = plt.subplots(5, 6, figsize=(10, 10))
for i in range(5):
    for j in range(6):
        pop = pops[i + j]

        obj = pop.extract_obj()

        # Plot the objectives on the current subplot.
        ax[i, j].scatter(obj[:, 0], obj[:, 1], s=2, color="red")

# Adjust subplot spacing
plt.tight_layout()

# Show the plot
plt.show()

# Plot decisions
# Create a 5x6 grid of subplots
fig, ax = plt.subplots(5, 6, figsize=(10, 10))
for i in range(5):
    for j in range(6):
        pop = pops[i + j]

        var = pop.extract_var()

        # Plot the varision vars on the current subplot.
        ax[i, j].scatter(var[:, 0], var[:, 1], s=2, color="green")

# Adjust subplot spacing
plt.tight_layout()

# Show the plot
plt.show()


# %%
def scatterplot_matrix(data):
    """
    Create a scatterplot matrix for a numpy array.

    Parameters:
    - data: numpy array (n x n) containing the data for the scatterplot matrix.

    Returns:
    - None (displays the scatterplot matrix).
    """
    # Convert the numpy array to a Pandas DataFrame for Seaborn
    import pandas as pd

    df = pd.DataFrame(data)

    # Create the scatterplot matrix using Seaborn
    sns.set(style="ticks")
    sns.pairplot(df, diag_kind="kde", markers="o")

    # Display the plot
    plt.show()


scatterplot_matrix(pops[0].extract_var())


# %%

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

# %%
