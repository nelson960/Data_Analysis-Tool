import os
import logging
import pandas as pd
from typing import Optional, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataLoader:
    def __init__(self, file_path: str, file_format: Optional[str] = None, **loader_kwargs):
        """
        Initialize the DataLoader with a file path and optional file format.
        
        Parameters:
            file_path (str): Path to the data file.
            file_format (str, optional): 'csv', 'json', or 'parquet'. If not provided,
                                         the loader will infer the format from the file extension.
            loader_kwargs: Additional keyword arguments passed to the pandas loader.
        """
        self.file_path = file_path
        self.file_format = file_format.lower() if file_format else self._infer_format()
        self.df = None
        self.selected_columns = None
        self._load_data(**loader_kwargs)

    def _infer_format(self) -> str:
        """
        Infer the file format based on the file extension.
        
        Returns:
            str: The inferred file format.
            
        Raises:
            ValueError: If the file extension is unsupported.
        """
        ext = os.path.splitext(self.file_path)[1].lower()
        if ext == '.csv':
            return 'csv'
        elif ext == '.json':
            return 'json'
        elif ext == '.parquet':
            return 'parquet'
        else:
            logging.error(f"Unsupported file extension '{ext}' for file: {self.file_path}")
            raise ValueError(f"Unsupported file extension '{ext}' for file: {self.file_path}")

    def _load_data(self, **loader_kwargs):
        """
        Internal method to load data from the specified file with robust error handling.
        
        Parameters:
            loader_kwargs: Additional keyword arguments for the pandas loader.
            
        Raises:
            FileNotFoundError: If the file does not exist.
            Exception: For any other error during data loading.
        """
        if not os.path.exists(self.file_path):
            logging.error(f"File not found: {self.file_path}")
            raise FileNotFoundError(f"File not found: {self.file_path}")
        
        try:
            if self.file_format == 'csv':
                self.df = pd.read_csv(self.file_path, **loader_kwargs)
            elif self.file_format == 'json':
                self.df = pd.read_json(self.file_path, **loader_kwargs)
            elif self.file_format == 'parquet':
                self.df = pd.read_parquet(self.file_path, **loader_kwargs)
            else:
                # This branch should never be reached because _infer_format handles supported types.
                logging.error(f"Unsupported file format: {self.file_format}")
                raise ValueError(f"Unsupported file format: {self.file_format}")
            
            # Initialize selected_columns as a copy of the full dataframe
            self.selected_columns = self.df.copy()
            logging.info(f"Data loaded successfully from '{self.file_path}' with shape {self.df.shape}.")
        except Exception as e:
            logging.exception(f"Failed to load data from '{self.file_path}'. Error: {e}")
            raise

    def select_columns(self, column_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Select specific columns for analysis. If no columns are provided, all columns are selected.
        
        Parameters:
            column_names (List[str], optional): List of column names to select.
            
        Returns:
            pd.DataFrame: DataFrame containing the selected columns.
        """
        if self.df is None:
            logging.error("Data not loaded. Cannot select columns.")
            raise ValueError("Data not loaded.")
        
        if column_names:
            available_columns = [col for col in column_names if col in self.df.columns]
            if available_columns:
                self.selected_columns = self.df[available_columns].copy()
                logging.info(f"Selected columns: {available_columns}")
            else:
                logging.warning("No matching columns found. Returning an empty DataFrame.")
                self.selected_columns = pd.DataFrame()
        else:
            self.selected_columns = self.df.copy()
            logging.info("No specific columns selected. Using all columns.")
        
        return self.selected_columns

    def get_dataframe(self) -> pd.DataFrame:
        """
        Retrieve the full DataFrame.
        
        Returns:
            pd.DataFrame: The full loaded DataFrame.
        """
        if self.df is None:
            logging.error("Data not loaded.")
            raise ValueError("Data not loaded.")
        return self.df

    def get_selected_columns(self) -> pd.DataFrame:
        """
        Retrieve the DataFrame with selected columns.
        
        Returns:
            pd.DataFrame: The DataFrame containing the selected columns.
        """
        if self.selected_columns is None:
            logging.error("Selected columns not set. Please use select_columns() first.")
            raise ValueError("Selected columns not set.")
        return self.selected_columns

    def reload_data(self, **loader_kwargs):
        """
        Reload data from the file. This is useful if the file has changed or you want to use
        new parameters for loading.
        
        Parameters:
            loader_kwargs: Additional keyword arguments for the pandas loader.
        """
        self._load_data(**loader_kwargs)
        logging.info("Data reloaded successfully.")

# Example usage:
# loader = DataLoader("data.csv")
# selected_df = loader.select_columns(["col1", "col2"])
