import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

class FeatureAnalyzer():
	def __init__(self, df: pd.DataFrame):
		self.df = df
		
	def feature_importance(self, target_column: str) ->None:

		if target_column not in self.df.columns:
			print(f"Target Column '{target_column}' not founded in the dataset.")
			return
		
		if not np.issubdtype(self.df[target_column].dtype, np.number):
			print(f"Error: Target Column '{target_column}' must be numeric.")
			return

		numeric_columns = self.df.select_dtypes(include=[np.number]).columns
		X = self.df[numeric_columns].drop(columns=[target_column], errors='ignore')
		y = self.df[target_column]

		if X.empty:
			print("Error: No numberic features founded for analysis")
			return
		
		try:
			X = pd.get_dummies(X) #One-hot encode categorical variables

			X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.2, random_state=42)

			model = RandomForestRegressor(n_estimators=100, random_state=42)
			model.fit(X_train, y_train)

			feature_importance = pd.DataFrame({
				'feature': X.columns,
				'importance':model.feature_importances_
			}).sort_values('importance', ascending=False)
			
			plt.figure(figsize=(10,8))
			sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
			plt.title(f"Top 10 important features for '{target_column}'")
			plt.tight_layout()
			plt.show()
			
		except Exception as e:
			print(f"An error occured during feature importance analysis: {str(e)}")