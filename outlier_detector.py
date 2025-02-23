import pandas as pd 
import numpy as np
from scipy import stats

class OutlierDetector:
	def __init__(self, df: pd.DataFrame) -> None:
		self.df = df

	def detect_outliers(self, method: str = 'zscore', threshold: float = 3.0 ) ->pd.DataFrame:
		numeric_df = self.df.select_dtypes(include=['number'])
		outliers = pd.DataFrame()

		if method =='zscore':
			zscore = np.abs(stats.zscore(numeric_df))
			outliers = numeric_df[(zscore > threshold).any(axis=1)]
		elif method == 'iqr':
			Q1 = numeric_df.quantile(0.25)
			Q3 = numeric_df.quantile(0.75)
			IQR = Q3 - Q1
			outliers = numeric_df[((numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 +1.5 * IQR))).any(axis=1)]
		else:
			print("Invalid method. Use 'zscore' or 'iqr'. ")
	
		return outliers