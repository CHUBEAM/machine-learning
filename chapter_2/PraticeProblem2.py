from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

initial_values = [10, 100, 1000]
param_grid = [
    {'kernel': ['linear'], 'c':initial_values},
    {'kernel': ['rbf'], 'c':initial_values, 'gamma': initial_values},
]
svr = SVR()
grid_search = GridSearchCV(svr, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)

grid_search.best_params_

randomized_search = RandomizedSearchCV(svr, n_iter=5, scoring='neg_mean_squred_error', return_train_score=True)
