from datetime import datetime
from typing import List

RESULTS_DIR_NAME = "results"


def generate_table(header: List[str], content: List[str], filename: str) -> None:
    if len(header) != len(content):
        raise Exception("Lists must have equal length!!!")

    begin: str = "\\begin{table}[!htbp]\n\centering\n\\begin{tabular}{|c|c|c|c|c|}\n\hline\n"
    table_description = "\n\caption{TODO}\n\label{" + filename.replace("%", "") + "}\n"
    end: str = "\end{tabular}" + table_description + "\end{table}\n"
    result = begin

    for index in range(len(header)):
        result += str(header[index]) + " & " + str(content[index]) + " \\\ \hline\n"

    result += end

    current_time = datetime.now().strftime("%H%M%S")
    with open(RESULTS_DIR_NAME + "/" + filename + current_time + ".txt", "w") as file:
        file.write(result)


def generate_image_figure(image_filename: str) -> None:
    replaced_filename = image_filename.replace("%", "")
    begin = "\\begin{figure}[!htbp]\n\centering\n\includegraphics\n[width=\\textwidth,keepaspectratio]\n"
    middle = "{img/" + replaced_filename + ".png}\n\caption\n[" + replaced_filename + "]\n{" + replaced_filename + "}\n\label{margarine_divorces}"
    end = "\end{figure}\n\FloatBarrier\n"

    current_time = datetime.now().strftime("%H%M%S")
    path = RESULTS_DIR_NAME + "/figure_" + image_filename + current_time + ".txt"
    with open(path, "w") as file:
        file.write(begin + middle + end)
