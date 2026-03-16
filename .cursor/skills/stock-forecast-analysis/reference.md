# Stock Forecast Analysis — Reference

## Data Analysis and Manipulation

- Use pandas for data manipulation and analysis
- Prefer method chaining for data transformations
- Use `loc` and `iloc` for explicit data selection
- Utilize `groupby` operations for efficient aggregation

## Visualization

- **matplotlib**: Low-level control and customization
- **seaborn**: Statistical visualizations, aesthetic defaults
- Include proper labels, titles, legends
- Use color-blindness accessible color schemes

## Jupyter Notebook Best Practices

- Structure with clear markdown sections
- Ensure meaningful cell execution order for reproducibility
- Document analysis steps in markdown cells
- Keep code cells focused and modular
- Use `%matplotlib inline` for inline plotting
- **Define functions in `src/` modules; import into notebooks**

## Error Handling and Data Validation

- Run data quality checks at analysis start
- Handle missing data: imputation, removal, or flagging
- Use try-except for error-prone operations (e.g., reading external data)
- Validate data types and ranges

## Performance Optimization

- Vectorized operations in pandas and numpy
- Categorical dtypes for low-cardinality strings
- Consider dask for larger-than-memory datasets
- Profile to find bottlenecks
- Optimize hyperparameter tuning: reduced spaces, fewer iterations, shallow trees

## Machine Learning and Statistical Validation

- 5-fold stratified cross-validation
- GridSearchCV for small parameter spaces; RandomizedSearchCV for large
- Paired t-tests to validate model improvements
- 95% CI for performance estimates
- Statistically validated model selection over raw best performance

## Code Structure (Multi-Model)

For multiple models: create subfolders under `src/forecasting/`, `data/`, and `src/data_preprocess/`.

Reusable functions:
- `src/forecasting/*_utils.py` for forecasting utilities
- `src/data_preprocess/data_preprocess_utils.py` for preprocessing

## Key Conventions

1. Begin analysis with data exploration and summary statistics
2. Create reusable plotting functions for consistent visualizations
3. Document data sources, assumptions, and methodologies
4. Use version control for notebooks and scripts
