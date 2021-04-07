from typing import List


def generate_table(header: List[str], content: List[str], filename: str) -> None:
    if len(header) != len(content):
        raise Exception("Lists must have equal length and length must be equals 2!!!")

    begin: str = "\\begin{table}[]\n\\begin{tabular}{|c|c|c|c|c|}\n\hline\n"
    end: str = "\end{tabular}\n\end{table}\n"
    result = begin

    for index in range(len(header)):
        result += header[index] + " & " + str(content[index]) + " \\\ \hline\n"

    result += end
    with open(filename + ".txt", "w") as file:
        file.write(result)
