def print_matrix_journal_style(matrix):
    """Prints a NumPy matrix in a journal-style format."""

    rows, cols = matrix.shape

    # Format each element as a string with consistent spacing
    formatted_rows = []
    for row in matrix:
        formatted_row = [
            "{:.3f}".format(val) if isinstance(val, float) else str(val) for val in row
        ]
        formatted_rows.append(formatted_row)

    # Calculate column widths
    col_widths = [max(len(val) for val in col) for col in zip(*formatted_rows)]

    # Print the matrix
    for row in formatted_rows:
        print(" ".join(val.rjust(width) for val, width in zip(row, col_widths)))
