import logging
from typing import Union, Any, List, Tuple
import pandas as pd
import re

# Configure logging (customize as needed)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class DataCleaner:
    """
    A class for cleaning and transforming a pandas DataFrame.
    Provides methods to drop columns, drop rows, fill missing values, 
    clean text in columns, and search & replace values.
    """
    def __init__(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            logging.error("Provided data is not a pandas DataFrame.")
            raise ValueError("df must be a pandas DataFrame")
        # Work on a copy of the DataFrame to avoid modifying original data.
        self.df = df.copy()
        self.changes_log: List[str] = []

    def drop_columns(self, columns: Union[str, List[str]]) -> pd.DataFrame:
        """
        Drop specified columns from the DataFrame.
        
        Parameters:
            columns (Union[str, List[str]]): A column name or list of column names to drop.
        
        Returns:
            pd.DataFrame: The modified DataFrame.
            
        Raises:
            TypeError: If 'columns' is not a string or list of strings.
            ValueError: If none of the specified columns exist in the DataFrame.
        """
        if isinstance(columns, str):
            columns_to_drop = [columns]
        elif isinstance(columns, list) and all(isinstance(col, str) for col in columns):
            columns_to_drop = columns
        else:
            logging.error("The 'columns' argument must be a string or a list of strings.")
            raise TypeError("The 'columns' argument must be a string or a list of strings.")

        valid_columns = [col for col in columns_to_drop if col in self.df.columns]
        if valid_columns:
            self.df.drop(columns=valid_columns, inplace=True)
            message = f"Dropped columns: {valid_columns}."
            self.changes_log.append(message)
            logging.info(message)
        else:
            logging.error("None of the specified columns were found in the DataFrame.")
            raise ValueError("None of the specified columns were found in the DataFrame.")

        return self.df

    def drop_rows(self, index_range: Tuple[int, int]) -> pd.DataFrame:
        """
        Drop rows within a given index range (inclusive).
        
        Parameters:
            index_range (Tuple[int, int]): A tuple with start and end indices.
        
        Returns:
            pd.DataFrame: The modified DataFrame.
        
        Raises:
            TypeError: If index_range is not a tuple of two integers.
            ValueError: If indices are out of bounds.
        """
        if not (isinstance(index_range, tuple) and len(index_range) == 2 and 
                all(isinstance(i, int) for i in index_range)):
            logging.error("The 'index_range' must be a tuple of two integers (start, end).")
            raise TypeError("The 'index_range' must be a tuple of two integers (start, end).")
        
        start_idx, end_idx = index_range
        num_rows = self.df.shape[0]
        if 0 <= start_idx <= end_idx < num_rows:
            indices_to_drop = self.df.index[start_idx:end_idx+1]
            self.df.drop(indices_to_drop, inplace=True)
            message = f"Dropped rows from index {start_idx} to {end_idx}."
            self.changes_log.append(message)
            logging.info(message)
        else:
            logging.error("Invalid index range. Indices must be within the DataFrame's index bounds.")
            raise ValueError("Invalid index range. Indices must be within the DataFrame's index bounds.")

        return self.df

    def fill_missing_values(self, fill_value: Union[str, int, float]) -> pd.DataFrame:
        """
        Fill missing values in the DataFrame with a specified fill value.
        
        Parameters:
            fill_value (Union[str, int, float]): The value to use for filling missing values.
        
        Returns:
            pd.DataFrame: The modified DataFrame.
        
        Raises:
            ValueError: If fill_value is None.
        """
        if fill_value is None:
            logging.error("A fill value must be provided.")
            raise ValueError("A fill value must be provided.")
        
        original_missing = self.df.isnull().sum()
        self.df.fillna(fill_value, inplace=True)
        new_missing = self.df.isnull().sum()
        filled_columns = {
            col: (original_missing[col] - new_missing[col])
            for col in self.df.columns if original_missing[col] - new_missing[col] > 0
        }
        for col, count in filled_columns.items():
            message = f"Filled {count} missing values in column '{col}' with {fill_value}."
            self.changes_log.append(message)
            logging.info(message)
        return self.df

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean text by removing URLs, special characters, and converting to lowercase.
        
        Parameters:
            text (str): The text string to clean.
        
        Returns:
            str: The cleaned text.
        """
        if not isinstance(text, str):
            return text
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'[^\w\s,]', '', text)
        text = text.lower()
        return text

    def clean_column(self, column_name: str) -> pd.DataFrame:
        """
        Clean text data in a specific column.
        
        Parameters:
            column_name (str): The name of the column to clean.
        
        Returns:
            pd.DataFrame: The modified DataFrame.
        
        Raises:
            ValueError: If the specified column is not found in the DataFrame.
        """
        if column_name in self.df.columns:
            self.df[column_name] = self.df[column_name].apply(self.clean_text)
            message = f"Cleaned text in column '{column_name}'."
            self.changes_log.append(message)
            logging.info(message)
        else:
            logging.error(f"Column '{column_name}' not found in the DataFrame.")
            raise ValueError(f"Column '{column_name}' not found in the DataFrame.")
        return self.df

    def search_and_replace(self, column: str, search_value: Any, replace_value: Any) -> pd.DataFrame:
        """
        Search for a value in a column and replace it with a new value.
        
        Parameters:
            column (str): The column in which to search and replace.
            search_value (Any): The value to search for.
            replace_value (Any): The value to replace with.
        
        Returns:
            pd.DataFrame: The modified DataFrame.
        
        Raises:
            ValueError: If the specified column is not found in the DataFrame.
        """
        if column not in self.df.columns:
            logging.error(f"Column '{column}' not found in the DataFrame.")
            raise ValueError(f"Column '{column}' not found in the DataFrame.")
        
        original_count = (self.df[column] == search_value).sum()
        self.df[column].replace(search_value, replace_value, inplace=True)
        new_count = (self.df[column] == search_value).sum()
        replace_count = original_count - new_count

        if replace_count > 0:
            message = (f"Replaced {replace_count} occurrence(s) of '{search_value}' "
                       f"with '{replace_value}' in column '{column}'.")
            self.changes_log.append(message)
            logging.info(message)
        else:
            logging.info(f"No occurrences of '{search_value}' found in column '{column}'.")
        
        return self.df

    def show_changes_log(self) -> None:
        """
        Log the changes made to the DataFrame.
        """
        if self.changes_log:
            for log_entry in self.changes_log:
                logging.info(log_entry)
        else:
            logging.info("No changes made yet.")

    def get_clean_data(self) -> pd.DataFrame:
        """
        Retrieve the cleaned DataFrame.
        
        Returns:
            pd.DataFrame: The cleaned DataFrame.
        """
        return self.df
