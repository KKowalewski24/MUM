from typing import Dict, List

from model.DividedDataSet import DividedDataSet
from module.LatexGenerator import LatexGenerator

latex_generator: LatexGenerator = LatexGenerator("knn")


def knn_classification(data_sets: Dict[int, List[DividedDataSet]], save_latex: bool) -> None:
    # TODO
    pass
