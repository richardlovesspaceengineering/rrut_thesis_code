from pathlib import Path
import numpy as np
from os import walk


def data_to_file(self, file_path='./results/',
                 delimiter=',', endtype='.txt'):

    # pop = self.population
    # pop_obj = pop.extract_obj()
    # path = f"./cases/BIOBJ_files/f46_10d_{self.seed}.pf"
    # np.savetxt(path, pop_obj, delimiter=",")
    # return
 
    # Attempt to join output filepath
    try:
        file_path = str(Path(self.output_path).joinpath(self.filename))
    except Exception as e:
        print(e)
    print(file_path)

    # Extract final non-dominated front
    final_pareto = self.opt.extract_obj()

    # Save text file
    np.savetxt(file_path + endtype, final_pareto,
               delimiter=delimiter, fmt='%f')

    # Indicator values (IGD)
    try:
        metric = self.indicator.extract_archive()
        np.savetxt(file_path + '_igd_metric' + endtype, metric,
                   delimiter=delimiter, fmt='%f')
    except Exception as e:
        print(e)

    # Full population, fronts and IGD values
    try:
        self.data_extractor.finalise_output()
    except Exception as e:
        print(e)

