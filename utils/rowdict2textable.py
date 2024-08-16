def generate_latex_table(data_dict, file_name='./class_balance.tex'):
    """
    Generate a LaTeX table from a dictionary and save it to a .tex file.

    Parameters:
    data_dict (dict): Dictionary where keys are column headers and values are lists of column entries.
    file_name (str): The name of the file to save the LaTeX code.
    """

    # Start writing the LaTeX table
    latex_code = "\\begin{table}[h!]\n"
    latex_code += "\\centering\n"
    latex_code += "\\caption{Number of pixels per class.}\n"
    latex_code += "\\begin{tabular}{" + " ".join(["c"] * len(data_dict)) + "}\n"
    latex_code += "\\toprule\n"
    
    # Write the header row
    header = " & ".join(data_dict.keys())
    latex_code += header + " \\\\\n"
    latex_code += "\\midrule\n"
    
    # Write the data row (only one row expected)
    row = " & ".join(str(data_dict[key]) for key in data_dict)
    latex_code += row + " \\\\\n"
    
    latex_code += "\\bottomrule\n"
    latex_code += "\\end{tabular}\n"
    latex_code += "\\label{tab:balance}\n"
    latex_code += "\\end{table}\n"

    # Save to a .tex file
    with open(file_name, 'w') as file:
        file.write(latex_code)

    print(f"LaTeX table has been written to {file_name}")
