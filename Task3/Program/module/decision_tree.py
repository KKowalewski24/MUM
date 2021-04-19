from typing import Tuple

import numpy as np

from module.LatexGenerator import LatexGenerator

latex_generator: LatexGenerator = LatexGenerator("decision_tree")


def decision_tree_classification(
        data_sets: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        save_latex: bool) -> None:
    # TODO
    pass
