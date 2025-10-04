"""
Weekly Data Aggregation for Stock Forecasting

This module aggregates daily stock data to weekly frequency for forecasting.
End-of-week prices are used for price data, while volume is summed for the week.
"""

import pandas as pd
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class WeeklyAggregator:
    """
    Aggregates daily stock data to weekly frequency.
    
    For price data (Open, High, Low, Close): Uses end-of-week values
    For volume data: Sums the weekly volume
    For other features: Applies appropriate aggregation methods
    """
    
    def __init__(
        self,
        price_columns: List[str] = ['Open', 'High', 'Low', 'Close'],
        volume_columns: List[str] = ['Volume'],
        aggregation_methods: Optional[Dict[str, str]] = None,
        week_start: str = 'W-FRI'  # Week ends on Friday
    ):
        """
        Initialize the weekly aggregator.
        
        Args:
            price_columns: Columns to aggregate using end-of-week values
            volume_columns: Columns to sum over the week
            aggregation_methods: Custom aggregation methods for other columns
            week_start: Week frequency string (W-FRI for Friday end)
        """
        self.price_columns = price_columns
        self.volume_columns = volume_columns
        self.aggregation_methods = aggregation_methods if aggregation_methods is not None else {}
        self.week_start = week_start
        
        # Default aggregation methods
        self.default_methods = {
            'mean': ['Adj Close'],
            'last': price_columns,
            'sum': volume_columns,
            'max': ['High'],
            'min': ['Low']
        }
    
    def aggregate(self, data: pd.DataFrame, date_column: str = 'Date') -> pd.DataFrame:
        """
        Aggregate daily data to weekly frequency.
        
        Args:
            data: Daily stock data DataFrame
            date_column: Name of the date column
            
        Returns:
            DataFrame with weekly aggregated data
        """
        logger.info("Starting weekly aggregation...")
        
        # Ensure date column is datetime
        if date_column in data.columns:
            data = data.copy()
            data[date_column] = pd.to_datetime(data[date_column])
            data.set_index(date_column, inplace=True)
        elif not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have a datetime index or date column")
        
        # Create weekly aggregation
        weekly_data = self._create_weekly_aggregation(data)
        
        logger.info(f"Weekly aggregation complete: {len(weekly_data)} weeks from {len(data)} days")
        return weekly_data
    
    def _create_weekly_aggregation(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create weekly aggregation using appropriate methods for each column.
        
        Args:
            data: Daily data with datetime index
            
        Returns:
            Weekly aggregated DataFrame
        """
        weekly_data = {}
        
        # Process each column with appropriate aggregation method
        for column in data.columns:
            # Check if column is numeric
            is_numeric = pd.api.types.is_numeric_dtype(data[column])
            
            if column in self.price_columns:
                # Use last value of the week (end-of-week price)
                weekly_data[column] = data[column].resample(self.week_start).last()
            elif column in self.volume_columns:
                # Sum volume over the week
                weekly_data[column] = data[column].resample(self.week_start).sum()
            elif column in self.aggregation_methods:
                # Use custom aggregation method
                method = self.aggregation_methods[column]
                if method == 'last':
                    weekly_data[column] = data[column].resample(self.week_start).last()
                elif method == 'sum' and is_numeric:
                    weekly_data[column] = data[column].resample(self.week_start).sum()
                elif method == 'mean' and is_numeric:
                    weekly_data[column] = data[column].resample(self.week_start).mean()
                elif method == 'max' and is_numeric:
                    weekly_data[column] = data[column].resample(self.week_start).max()
                elif method == 'min' and is_numeric:
                    weekly_data[column] = data[column].resample(self.week_start).min()
                else:
                    # Default to last for non-numeric or unknown methods
                    weekly_data[column] = data[column].resample(self.week_start).last()
            else:
                # Default aggregation based on column name patterns
                if 'high' in column.lower() or 'max' in column.lower():
                    if is_numeric:
                        weekly_data[column] = data[column].resample(self.week_start).max()
                    else:
                        weekly_data[column] = data[column].resample(self.week_start).last()
                elif 'low' in column.lower() or 'min' in column.lower():
                    if is_numeric:
                        weekly_data[column] = data[column].resample(self.week_start).min()
                    else:
                        weekly_data[column] = data[column].resample(self.week_start).last()
                elif 'volume' in column.lower() or 'vol' in column.lower():
                    if is_numeric:
                        weekly_data[column] = data[column].resample(self.week_start).sum()
                    else:
                        weekly_data[column] = data[column].resample(self.week_start).last()
                else:
                    # Default to last for non-numeric columns, mean for numeric
                    if is_numeric:
                        weekly_data[column] = data[column].resample(self.week_start).mean()
                    else:
                        weekly_data[column] = data[column].resample(self.week_start).last()
        
        # Combine all weekly data
        weekly_df = pd.DataFrame(weekly_data)
        
        # Remove any rows with all NaN values
        weekly_df = weekly_df.dropna(how='all')
        
        # Forward fill any remaining NaN values for price columns
        for col in self.price_columns:
            if col in weekly_df.columns:
                weekly_df[col] = weekly_df[col].ffill()
        
        return weekly_df
    
    def aggregate_multiple_stocks(
        self, 
        stock_data: Dict[str, pd.DataFrame], 
        date_column: str = 'Date'
    ) -> Dict[str, pd.DataFrame]:
        """
        Aggregate multiple stocks to weekly frequency.
        
        Args:
            stock_data: Dictionary mapping stock symbols to DataFrames
            date_column: Name of the date column
            
        Returns:
            Dictionary with weekly aggregated data for each stock
        """
        logger.info(f"Aggregating {len(stock_data)} stocks to weekly frequency...")
        
        weekly_stocks = {}
        for symbol, data in stock_data.items():
            logger.info(f"Processing {symbol}...")
            try:
                weekly_data = self.aggregate(data, date_column)
                weekly_stocks[symbol] = weekly_data
            except Exception as e:
                logger.error(f"Error aggregating {symbol}: {e}")
                continue
        
        logger.info(f"Weekly aggregation complete for {len(weekly_stocks)} stocks")
        return weekly_stocks
    
    def get_aggregation_summary(self, daily_data: pd.DataFrame, weekly_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary statistics of the aggregation process.
        
        Args:
            daily_data: Original daily data
            weekly_data: Aggregated weekly data
            
        Returns:
            Dictionary with aggregation summary
        """
        summary = {
            'daily_records': len(daily_data),
            'weekly_records': len(weekly_data),
            'compression_ratio': len(daily_data) / len(weekly_data),
            'date_range_daily': {
                'start': daily_data.index.min(),
                'end': daily_data.index.max()
            },
            'date_range_weekly': {
                'start': weekly_data.index.min(),
                'end': weekly_data.index.max()
            },
            'columns_processed': list(weekly_data.columns),
            'missing_values': weekly_data.isnull().sum().to_dict()
        }
        
        return summary
