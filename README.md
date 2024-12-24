# cost-recsys

This project is aimed at developing a dynamic pricing system for e-commerce platforms. The system includes predictive modeling, demand elasticity analysis, and price optimization under business constraints. 

## Features
- **Demand Prediction**: Utilizes machine learning to predict product demand based on historical data.
- **Elasticity Curve Analysis**: Models the relationship between price changes and demand fluctuations.
- **Price Optimization**: Suggests optimal pricing strategies to maximize revenue or achieve other KPIs.

## Project Components
1. **Data Preprocessing**: 
   - Conducted exploratory data analysis (EDA) using tools like Pandas, Matplotlib, and NumPy.
   - Extracted and engineered features such as weekly sales summaries, elasticity indicators, and gross market value.

2. **Machine Learning Model**:
   - Implemented regression models using the CatBoost framework.
   - Applied RMSE and MAE for loss function evaluation and error interpretation.

3. **Optimization Logic**:
   - Developed methods to adjust pricing dynamically based on predicted demand and business constraints.

4. **Visualization**:
   - Visualized results through graphs to highlight demand-price relationships and optimization outcomes.

## Usage guide
Example datasets at _backend/examples/_  
### Ready API
### Setup for own
Build image
```commandline
docker build -t master-cost-recsys -f Dockerfile .
```
Run container
```commandline
docker run -d -p 12345:12345 --name master-cost-recsys master-cost-recsys
```
### Methods
1. **Train**  
Allows to train a model on your own data  
 - weights_name: string  
 - target_column_name: string  
> For example datasets (backend/examples/) = "_purchase_count_prod_"

 - iterations: int
 - depth: int
 - file: .csv example-like format

2. **Predict**  
Allows to use of trained models and download predictions
 - weights_name: string
 - target_column_name: string

## Tools and Libraries
- **Python**: Main programming language for analytics and development.
- **Pandas**: Data manipulation and cleaning.
- **Matplotlib**: Data visualization.
- **NumPy**: Numerical computations.
- **CatBoost**: Machine learning framework for regression tasks.
- **fastapi**: Web framework for building APIs with Python

## Team Members
- **Korol Maksim Maximovich**: Project Manager, Data Scientist.
- **Shkolin Alexander Yuryevich**: Data Scientist.
- **Nechaev Danila Konstantinovich**: Data Scientist.
- **Gareev Eldar Rustemovich**: Data Scientist.
