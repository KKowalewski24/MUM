from typing import List, Tuple
from sklearn import svm
from sklearn import metrics
import numpy as np

from module.LatexGenerator import LatexGenerator

LATEX_RESULTS_DIR = "svm"
latex_generator: LatexGenerator = LatexGenerator("svm")


def svm_classification(data_set: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                       save_latex: bool) -> None:
    accuracy_list: List[List[float]] = []
    svm_classifier = svm.SVC(
        kernel="rbf"
    )
    svm_classifier.fit(data_set[0], data_set[2])
    y_prediction = svm_classifier.predict(data_set[1])
    accuracy = round(metrics.accuracy_score(data_set[3], y_prediction), 4)
    accuracy_list.append([accuracy])
    print("accuracy: " + str(accuracy))







    # if save_latex:
    #     filename_description = "_" + data_set_name + ("_eucl" if is_euclidean_metric else "_manh")
    #     latex_generator.generate_horizontal_table(
    #         ["Accuracy"], list(K_RANGE), accuracy_list,
    #         "knn_table" + filename_description
    #     )
    #     image_filename = filename_description + "-" + datetime.now().strftime("%H%M%S")
    #     latex_generator.generate_chart_image("knn_chart" + image_filename)
    #     plt.savefig(LATEX_RESULTS_DIR + "/knn_chart" + image_filename)
    #     plt.close()
