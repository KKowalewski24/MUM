from datetime import datetime

import pandas as pd

RESULTS_DIR_NAME = "results"


def generate_table(df: pd.DataFrame, filename: str) -> None:
    begin: str = "\\begin{table}[!htbp]\n"
    centering: str = "\centering\n"
    begin_tabular: str = "\\begin{tabular}{|c|c|c|c|c|c|c|}\n"
    back_slashes: str = "\\\\"
    hline: str = "\hline\n"
    end_tabular: str = "\end{tabular}\n"
    end: str = "\end{table}\n"
    result = begin + centering + begin_tabular + hline
    # result += " & " + back_slashes + " " + hline
    for i in range(len(df.values)):
        result += df.index[i] + " & "
        for j in range(len(df.values[i])):
            result += str(round(df.values[i].tolist()[j], 4))
            if j < len(df.values[i]) - 1:
                result += " & "
        result += " " + back_slashes + " " + hline

    result += end_tabular + end

    print(result)


def generate_image_figure(image_filename: str) -> None:
    replaced_filename = image_filename.replace("%", "")
    begin = "\\begin{figure}[!htbp]\n\centering\n\includegraphics\n[width=\\textwidth,keepaspectratio]\n"
    middle = "{img/" + replaced_filename + ".png}\n\caption\n[" + replaced_filename + "]\n{" + replaced_filename + "}\n\label{margarine_divorces}"
    end = "\end{figure}\n\FloatBarrier\n"

    current_time = datetime.now().strftime("%H%M%S")
    path = RESULTS_DIR_NAME + "/figure_" + image_filename + current_time + ".txt"
    with open(path, "w") as file:
        file.write(begin + middle + end)


def save_to_file(self, data: str, filename: str) -> None:
    with open(filename + "-" + datetime.now().strftime("%H%M%S"), "w") as save_file:
        save_file.write(data)
