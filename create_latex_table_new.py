def create_latex_table(table_data, output_file=None):
    """Create a LaTeX table from data.
    
    Args:
        table_data (dict): Table data dictionary.
        output_file (str): Output file path for LaTeX table.
        
    Returns:
        str: LaTeX table string.
    """
    title = table_data["title"]
    columns = table_data["columns"]
    data = table_data["data"]
    highlight_column = table_data.get("highlight_column", None)
    
    # For the forecasting table with the new data structure, use a special format
    if isinstance(data, dict) and "ETTh1" in data:
        # Create LaTeX table header for the special two-column format
        latex_str = "\\begin{table}[h]\\centering\\caption{%s}\\vspace{0.5em}\\n" % title
        # Create tabular with double columns
        latex_str += "\\begin{tabular}{lccccccc | lccccccc}\\hline\\hline\\n"
        
        # Add column headers (twice for two columns)
        header_line = " & ".join(columns) + " & " + " & ".join(columns)
        latex_str += header_line + " \\\\hline\\n"
        
        # Add data for ETTh1 and ETTm1
        latex_str += "\\textbf{ETTh1:} & & & & & & & & \\textbf{ETTm1:} & & & & & & & \\\\hline\\n"
        
        # Get maximum number of rows between left and right datasets for alignment
        max_rows = max(len(data["ETTh1"]), len(data["ETTm1"]))
        
        # Add data rows for ETTh1 and ETTm1
        for i in range(max_rows):
            # Left part: ETTh1
            left_row = []
            if i < len(data["ETTh1"]):
                left_row_data = data["ETTh1"][i]
                for j, val in enumerate(left_row_data):
                    if j == highlight_column and isinstance(val, float):
                        left_row.append("\\textbf{\\textcolor{red}{%.3f}}" % val)
                    elif isinstance(val, float):
                        left_row.append("%.3f" % val)
                    else:
                        left_row.append(str(val))
            else:
                left_row = [''] * len(columns)
            
            # Right part: ETTm1
            right_row = []
            if i < len(data["ETTm1"]):
                right_row_data = data["ETTm1"][i]
                for j, val in enumerate(right_row_data):
                    if j == highlight_column and isinstance(val, float):
                        right_row.append("\\textbf{\\textcolor{red}{%.3f}}" % val)
                    elif isinstance(val, float):
                        right_row.append("%.3f" % val)
                    else:
                        right_row.append(str(val))
            else:
                right_row = [''] * len(columns)
            
            # Join left and right rows
            full_row = " & ".join(left_row) + " & " + " & ".join(right_row)
            latex_str += full_row + " \\\\hline\\n"
        
        # Add ETTh2 and Electricity
        latex_str += "\\textbf{ETTh2:} & & & & & & & & \\textbf{Electricity:} & & & & & & & \\\\hline\\n"
        
        # Add data rows for ETTh2 and Electricity
        max_rows_2 = max(len(data["ETTh2"]), len(data["Electricity"]))
        for i in range(max_rows_2):
            # Left part: ETTh2
            left_row = []
            if i < len(data["ETTh2"]):
                left_row_data = data["ETTh2"][i]
                for j, val in enumerate(left_row_data):
                    if j == highlight_column and isinstance(val, float):
                        left_row.append("\\textbf{\\textcolor{red}{%.3f}}" % val)
                    elif isinstance(val, float):
                        left_row.append("%.3f" % val)
                    else:
                        left_row.append(str(val))
            else:
                left_row = [''] * len(columns)
            
            # Right part: Electricity
            right_row = []
            if i < len(data["Electricity"]):
                right_row_data = data["Electricity"][i]
                for j, val in enumerate(right_row_data):
                    if j == highlight_column and isinstance(val, float):
                        right_row.append("\\textbf{\\textcolor{red}{%.3f}}" % val)
                    elif isinstance(val, float):
                        right_row.append("%.3f" % val)
                    else:
                        right_row.append(str(val))
            else:
                right_row = [''] * len(columns)
            
            # Join left and right rows
            full_row = " & ".join(left_row) + " & " + " & ".join(right_row)
            latex_str += full_row + " \\\\hline\\n"
        
        # Close LaTeX table
        latex_str += "\\end{tabular}\\end{table}"
    else:
        # Original behavior for other tables
        latex_str = "\\begin{table}[h]\\centering\\caption{%s}\\vspace{0.5em}\\n" % title
        latex_str += "\\begin{tabular}{l%s}\\hline\\hline\\n" % ('c' * (len(columns) - 1))
        
        # Add column headers
        latex_str += " & ".join(columns) + " \\\\hline\\n"
        
        # Add data rows
        for row in data:
            row_str = []
            for i, val in enumerate(row):
                if i == highlight_column and isinstance(val, float):
                    # Highlight best result with red bold
                    row_str.append("\\textbf{\\textcolor{red}{%.2f}}" % val)
                elif isinstance(val, float):
                    row_str.append("%.2f" % val)
                else:
                    row_str.append(str(val))
            latex_str += " & ".join(row_str) + " \\\\hline\\n"
        
        # Close LaTeX table
        latex_str += "\\end{tabular}\\end{table}"
    
    # Save to file if specified
    if output_file:
        # Ensure directory exists
        import os
        dir_path = os.path.dirname(output_file)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        with open(output_file, "w") as f:
            f.write(latex_str)
        print(f"Saved LaTeX table to: {output_file}")
    
    return latex_str
