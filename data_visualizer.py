import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import logging
from typing import Any

class DataVisualizer:
    """
    A class for visualizing different aspects of a pandas DataFrame,
    including missing values, correlation heatmaps, and data distributions.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the DataVisualizer with a pandas DataFrame.

        Parameters:
            df (pd.DataFrame): The DataFrame to visualize.

        Raises:
            ValueError: If the provided df is not a pandas DataFrame.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame")
        self.df = df.copy()  # Work on a copy to avoid modifying the original data

        # Set up logging
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def visualize_missing_values(self) -> None:
        """
        Plot a heatmap showing the presence of missing values in the DataFrame.
        """
        try:
            plt.figure(figsize=(10, 6))
            sns.heatmap(self.df.isnull(), cbar=False, cmap="viridis")
            plt.title("Missing Values Heatmap")
            plt.tight_layout()
            plt.show()
            self.logger.info("Missing values heatmap displayed successfully.")
        except Exception as e:
            self.logger.exception(f"Failed to visualize missing values: {e}")

    def correlation_heatmap(self) -> None:
        """
        Display a heatmap of correlations between numerical columns in the DataFrame.

        Logs an error if there are not enough numerical columns.
        """
        try:
            numeric_df = self.df.select_dtypes(include=['number'])
            if numeric_df.shape[1] > 1:
                plt.figure(figsize=(12, 10))
                corr_matrix = numeric_df.corr()
                sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
                plt.title("Correlation Heatmap")
                plt.tight_layout()
                plt.show()
                self.logger.info("Correlation heatmap displayed successfully.")
            else:
                self.logger.error("Not enough numerical data for correlation heatmap.")
        except Exception as e:
            self.logger.exception(f"Failed to generate correlation heatmap: {e}")

    def visualize_data_distribution(self) -> None:
        """
        Plot histograms for numerical features to show their distributions.
        """
        try:
            numeric_cols = self.df.select_dtypes(include=['number']).columns
            if numeric_cols.empty:
                self.logger.error("No numerical columns found in the DataFrame.")
                return

            n_cols = 3
            n_rows = (len(numeric_cols) - 1) // n_cols + 1

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
            axes = axes.flatten()

            for i, col in enumerate(numeric_cols):
                sns.histplot(self.df[col], kde=True, color='skyblue', ax=axes[i])
                axes[i].set_title(f"Distribution of {col}")

            # Remove any unused axes
            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])

            plt.tight_layout()
            plt.show()
            self.logger.info("Data distribution plots displayed successfully.")
        except Exception as e:
            self.logger.exception(f"Failed to visualize data distribution: {e}")
