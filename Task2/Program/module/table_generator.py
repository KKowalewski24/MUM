from datetime import datetime
from typing import List


def generate_table(header: List[str], content: List[str], filename: str) -> None:
    if len(header) != len(content):
        raise Exception("Lists must have equal length and length must be equals 2!!!")

    begin: str = "\\begin{table}[]\n\centering\n\\begin{tabular}{|c|c|c|c|c|}\n\hline\n"
    table_description = "\n\caption{TODO}\n\label{" + filename + "}\n"
    end: str = "\end{tabular}" + table_description + "\end{table}\n"
    result = begin
    # TODO
    # for index in range(len(header)):
    #     result += header[index] + " & " + str(content[index]) + " \\\ \hline\n"

    result += end

    current_time = datetime.now().strftime("%H%M%S")
    with open(filename + current_time + ".txt", "w") as file:
        file.write(result)
