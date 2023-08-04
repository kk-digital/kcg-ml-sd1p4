
import numpy as np


class UtilHistogram:

    def report_residuals_histogram(self, residuals, type):
        max_residual = max(residuals)
        hist_bins = np.linspace(0, max_residual, 11)

        histogram, bin_edges = np.histogram(residuals, bins=hist_bins)
        total_residues = len(residuals)

        histogram_string = f"\n{type} Residuals Histogram:\n"
        histogram_string += f"{'Range':<13} {'Percentage':<12} {'Histogram'}\n"

        max_digits = 12  # Maximum digits for percentage (including decimal point)
        for i, count in enumerate(histogram):
            bin_start, bin_end = bin_edges[i], bin_edges[i + 1]
            percentage = (count / total_residues) * 100
            asterisks = int(percentage / 2)
            percentage_str = f"{percentage:.2f}%"
            histogram_string += f"{bin_start:.2f} - {bin_end:.2f}   {percentage_str:<{max_digits}} {'*' * asterisks}\n"

        return histogram_string
