from typing import Dict, Tuple

import pandas as pd

from module.LatexGenerator import LatexGenerator

latex_generator: LatexGenerator = LatexGenerator("svm")


def svm_classification(data_sets: Dict[int, Tuple[pd.DataFrame, pd.DataFrame]],
                       save_latex: bool) -> None:
    # TODO
    pass
