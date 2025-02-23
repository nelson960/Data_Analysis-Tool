import pandas as pd
import logging
from typing import Any
from io import StringIO

class DataReporter:
    """
    A class to generate a comprehensive report for a DataFrame,
    including summary statistics, missing data analysis, and visualizations.
    """

    def __init__(self, df: pd.DataFrame, data_visualizer: Any) -> None:
        """
        Initialize the DataReporter with a DataFrame and a data visualizer instance.
        
        Parameters:
            df (pd.DataFrame): The DataFrame to report on.
            data_visualizer (Any): An instance of a visualizer class with a
                                   visualize_missing_values() method.
        
        Raises:
            ValueError: If the provided df is not a pandas DataFrame.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame")
        self.df = df.copy()  # Work on a copy to avoid modifying original data
        self.data_visualizer = data_visualizer

        # Set up logging
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def generate_summary_statistics(self) -> str:
        """
        Generate summary statistics for the DataFrame.
        
        Returns:
            str: A string containing the number of rows and columns, the first 5 rows,
                 and descriptive statistics of the DataFrame.
        """
        try:
            summary = (
                f"{'='*40}\n"
                f"Number of rows: {self.df.shape[0]}\n"
                f"Number of columns: {self.df.shape[1]}\n\n"
                f"First 5 rows:\n{self.df.head()}\n\n"
                f"{'='*40}\n"
                f"Descriptive Statistics:\n{self.df.describe().round(2)}\n"
                f"{'='*40}"
            )
            self.logger.info("Summary statistics generated successfully.")
            return summary
        except Exception as e:
            self.logger.exception(f"Failed to generate summary statistics: {e}")
            raise

    def analyze_missing_data(self) -> pd.DataFrame:
        """
        Analyze missing data in the DataFrame.
        
        Returns:
            pd.DataFrame: A DataFrame that includes the total count and percentage of missing
                          values for each column with missing data, sorted by the total missing.
        """
        try:
            missing = self.df.isnull().sum()
            missing_percentage = 100 * missing / len(self.df)
            missing_table = pd.DataFrame({
                'Total Missing': missing,
                'Percentage Missing': missing_percentage
            })
            result = missing_table[missing_table['Total Missing'] > 0].sort_values('Total Missing', ascending=False)
            self.logger.info("Missing data analysis completed successfully.")
            return result
        except Exception as e:
            self.logger.exception(f"Failed to analyze missing data: {e}")
            raise

    def generate_report(self) -> None:
        """
        Generate a complete report including summary statistics, missing data analysis,
        visualization of missing values, and DataFrame info.
        """
        try:
            # Generate and log summary statistics
            summary = self.generate_summary_statistics()
            self.logger.info("Summary Statistics:\n" + summary)
            print(summary)

            # Analyze and log missing data
            missing_data = self.analyze_missing_data()
            self.logger.info("Missing Data Analysis:\n" + missing_data.to_string())
            print("\nMissing Data Analysis:")
            print(missing_data)

            # Visualize missing values
            self.logger.info("Visualizing missing values.")
            print("\nVisualizing Missing Values:")
            self.data_visualizer.visualize_missing_values()

            # Capture DataFrame info as a string
            buffer = StringIO()
            self.df.info(buf=buffer)
            info_str = buffer.getvalue()
            self.logger.info("Data Info:\n" + info_str)
            print("\nData Info:")
            print(info_str)
        except Exception as e:
            self.logger.exception(f"Failed to generate report: {e}")
            raise
