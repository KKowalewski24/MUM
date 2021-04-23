from typing import List, Tuple
from sklearn import svm
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

from module.LatexGenerator import LatexGenerator

LATEX_RESULTS_DIR = "svm"
latex_generator: LatexGenerator = LatexGenerator("svm")

KERNEL_FUNCTIONS = ("poly", "sigmoid", "rbf")
C_RANGE = [round(x, 1) for x in np.arange(0.1, 2.1, 0.1)] 
GAMMA_VALUES = [float("1e" + str(x)) for x in range(-10, 10)] 

def svm_classification(data_set: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                       save_latex: bool) -> None:
    accuracy_list: List[List[float]] = []

    start = timer()
    for kernel_function in KERNEL_FUNCTIONS :
        print("Kernel function: " + kernel_function)
        for c in C_RANGE :
            svm_classifier = svm.SVC(
                kernel=kernel_function,
                C = c
            )
            svm_classifier.fit(data_set[0], data_set[2])
            y_prediction = svm_classifier.predict(data_set[1])
            accuracy = round(metrics.accuracy_score(data_set[3], y_prediction), 4)
            accuracy_list.append([accuracy])
            print("C value: " + str(c) + "\t" + "accuracy: " + str(accuracy))
        
    
    end = timer()
    print("Time of data collectiong (sec): ", end - start)
    plt.plot(C_RANGE, accuracy_list[0:20], "red")
    plt.plot(C_RANGE, accuracy_list[20:40], "blue")
    plt.plot(C_RANGE, accuracy_list[40:60], "green")

    plt.ylabel("Accuracy")
    plt.xlabel("C value")
    plt.show()





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
