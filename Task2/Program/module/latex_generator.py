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
    float_barrier: str = "\FloatBarrier\n"

    column_names = ""
    for i in range(len(df.columns)):
        column_names += df.columns[i]
        if i <= len(df.columns[i]) + 1:
            column_names += " & "

    result = begin + centering + begin_tabular + hline
    result += " & " + column_names + " " + back_slashes + " " + hline

    for i in range(len(df.values)):
        result += df.index[i] + " & "
        for j in range(len(df.values[i])):
            result += str(round(df.values[i].tolist()[j], 4))
            if j < len(df.values[i]) - 1:
                result += " & "
        result += " " + back_slashes + " " + hline

    result += end_tabular + end + float_barrier
    save_to_file(result, RESULTS_DIR_NAME + "/table-" + filename)


def generate_image_figure(image_filename: str) -> None:
    replaced_filename = image_filename.replace("%", "")
    result = "\\begin{figure}[!htbp]\n\centering\n\includegraphics\n[width=\\textwidth,keepaspectratio]\n"
    result += "{img/" + replaced_filename + ".png}\n\caption\n[" + replaced_filename + "]\n{" + replaced_filename + "}\n\label{" + replaced_filename + "}\n"
    result += "\end{figure}\n\FloatBarrier\n"

    save_to_file(result, RESULTS_DIR_NAME + "/figure-" + image_filename)


def save_to_file(data: str, filename: str) -> None:
    with open(filename + "-" + datetime.now().strftime("%H%M%S") + ".txt", "w") as txt:
        txt.write(data)
