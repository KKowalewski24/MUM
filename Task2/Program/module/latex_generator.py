from datetime import datetime

import pandas as pd

RESULTS_DIR_NAME = "results"


def generate_table(df: pd.DataFrame, filename: str) -> None:
    begin: str = "\\begin{table}[!htbp]\n"
    centering: str = "\centering\n"
    if 'hypothesis' not in filename:
        begin_tabular: str = "\\begin{tabular}{|c|c|c|c|c|c|c|}\n"
    else:
        begin_tabular: str = "\\begin{tabular}{|c|c|c|}\n"
    back_slashes: str = "\\\\"
    hline: str = "\hline\n"
    end_tabular: str = "\end{tabular}\n"
    caption: str = "\caption{}\n"
    end: str = "\end{table}\n"
    float_barrier: str = "\FloatBarrier\n"


    def get_label(label: str) -> str:
        return "\label{" + label + "}\n"


    column_names = ""
    for i in range(len(df.columns)):
        column_names += df.columns[i]
        if i <= len(df.columns) - 2:
            column_names += " & "

    result = begin + centering + begin_tabular + hline
    result += " & " + column_names + " " + back_slashes + " " + hline

    for i in range(len(df.values)):
        result += str(df.index[i]) + " & "
        for j in range(len(df.values[i])):
            x = df.values[i][j]
            if type(x) is int:
                x = round(x, 4)
            result += str(x)
            if j < len(df.values[i]) - 1:
                result += " & "
        result += " " + back_slashes + " " + hline

    result += caption + get_label(filename) + end_tabular + end + float_barrier
    save_to_file(result, RESULTS_DIR_NAME + "/table-" + filename)


def generate_image_figure(image_filename: str, regression_coef: float,
                          regression_intercept: float) -> None:
    replaced_filename = image_filename.replace("%", "")
    result = "\\begin{figure}[!htbp]\n\centering\n\includegraphics\n[width=\\textwidth,keepaspectratio]\n"
    result += "{img/" + replaced_filename + ".png}\n\caption\n[" + replaced_filename + "]\n"
    result += "{Współczynnik kierunkowy: " + str(regression_coef)
    result += ", Punkt przecięcia: " + str(regression_intercept) + "}\n"
    result += "\label{" + replaced_filename + "}\n"
    result += "\end{figure}\n\FloatBarrier\n"

    save_to_file(result, RESULTS_DIR_NAME + "/figure-" + image_filename)


def save_to_file(data: str, filename: str) -> None:
    with open(filename + "-" + datetime.now().strftime("%H%M%S") + ".txt", "w",
              encoding="utf-8") as txt:
        txt.write(data)
