import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from src.utils.database import execute_query
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import minimize
import logging

logger = logging.getLogger('quantum_tracker')

class QuantumPortfolioOptimizer:
    def __init__(self, db_path):
        self.db_path = db_path
        self.train_days = 60
        self.test_days = 30
        
    def get_all_stock_data(self):
        """Get all historical stock data"""
        query = """
            SELECT ticker, price, timestamp
            FROM stock_data
            ORDER BY ticker, timestamp DESC
        """
        data = execute_query(self.db_path, query)
        
        # Convert to pandas DataFrame
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    def prepare_features(self, prices):
        """Create features for prediction models"""
        features = pd.DataFrame()
        
        # Returns
        features['returns_1d'] = prices.pct_change(1)
        features['returns_5d'] = prices.pct_change(5)
        features['returns_10d'] = prices.pct_change(10)
        
        # Volatility
        features['vol_5d'] = prices.pct_change().rolling(5).std()
        features['vol_10d'] = prices.pct_change().rolling(10).std()
        
        # Momentum
        features['momentum_5d'] = prices / prices.shift(5) - 1
        features['momentum_10d'] = prices / prices.shift(10) - 1
        
        # Moving averages
        features['sma_5'] = prices.rolling(5).mean()
        features['sma_10'] = prices.rolling(10).mean()
        features['price_to_sma5'] = prices / features['sma_5']
        
        # RSI
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        
        return features.dropna()
    
    def train_prediction_models(self, ticker_data):
        """Train multiple models for return prediction"""
        
        # Prepare data
        prices = ticker_data.set_index('timestamp')['price'].sort_index()
        features = self.prepare_features(prices)
        
        # Target: next day return
        target = prices.pct_change().shift(-1)
        
        # Align features and target
        valid_idx = features.index.intersection(target.index)
        X = features.loc[valid_idx]
        y = target.loc[valid_idx]
        
        # Remove any remaining NaN
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        if len(X) < 30:
            return None, None
        
        # Split: train on first 60%, test on last 40%
        split_idx = int(len(X) * 0.6)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Train models
        models = {}
        
        # Ridge Regression
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, y_train)
        models['ridge'] = {
            'model': ridge,
            'train_score': ridge.score(X_train, y_train),
            'test_score': ridge.score(X_test, y_test)
        }
        
        # Random Forest
        rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        rf.fit(X_train, y_train)
        models['rf'] = {
            'model': rf,
            'train_score': rf.score(X_train, y_train),
            'test_score': rf.score(X_test, y_test)
        }
        
        # Ensemble prediction (average of models)
        ridge_pred = ridge.predict(X_test)
        rf_pred = rf.predict(X_test)
        ensemble_pred = (ridge_pred + rf_pred) / 2
        
        # Calculate metrics
        actual_returns = y_test.values
        pred_returns = ensemble_pred
        
        # Sharpe ratio of predictions
        pred_sharpe = np.mean(pred_returns) / np.std(pred_returns) * np.sqrt(252) if np.std(pred_returns) > 0 else 0
        
        # Direction accuracy
        direction_accuracy = np.mean((pred_returns > 0) == (actual_returns > 0))
        
        return models, {
            'expected_return': np.mean(pred_returns),
            'predicted_vol': np.std(pred_returns) * np.sqrt(252),
            'sharpe': pred_sharpe,
            'direction_accuracy': direction_accuracy,
            'train_size': len(X_train),
            'test_size': len(X_test)
        }
    
    def optimize_portfolio(self, expected_returns, cov_matrix, tickers):
        """Mean-variance portfolio optimization"""
        
        n_assets = len(expected_returns)
        
        # Objective: minimize variance for target return
        def portfolio_variance(weights):
            return weights.T @ cov_matrix @ weights
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # weights sum to 1
        ]
        
        # Bounds: 0% to 40% per stock (diversification)
        bounds = tuple((0, 0.4) for _ in range(n_assets))
        
        # Initial guess: equal weight
        initial_weights = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(
            portfolio_variance,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            weights = result.x
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_vol = np.sqrt(portfolio_variance(weights))
            sharpe = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
            
            return {
                'weights': weights,
                'expected_return': portfolio_return,
                'volatility': portfolio_vol,
                'sharpe': sharpe
            }
        
        return None
    
    def backtest_strategy(self, ticker_data, weights):
        """Backtest portfolio strategy"""
        
        prices = ticker_data.set_index('timestamp')['price'].sort_index()
        returns = prices.pct_change().dropna()
        
        # Portfolio returns
        portfolio_returns = returns * weights
        cumulative_returns = (1 + portfolio_returns).cumprod()
        
        # Metrics
        total_return = cumulative_returns.iloc[-1] - 1
        sharpe = np.mean(portfolio_returns) / np.std(portfolio_returns) * np.sqrt(252)
        max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min()
        
        return {
            'total_return': total_return,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'final_value': cumulative_returns.iloc[-1]
        }

def analyze_portfolio(db_path):
    """Full quantitative portfolio analysis with ML and optimization"""
    
    optimizer = QuantumPortfolioOptimizer(db_path)
    
    # Get all stock data
    df = optimizer.get_all_stock_data()
    
    if df.empty:
        return {'recommendations': [], 'summary': 'No data available'}
    
    tickers = df['ticker'].unique()
    results = []
    
    logger.info(f"Analyzing {len(tickers)} stocks with ML models...")
    
    # Train models for each stock
    for ticker in tickers:
        ticker_data = df[df['ticker'] == ticker].copy()
        
        if len(ticker_data) < 30:
            continue
        
        models, metrics = optimizer.train_prediction_models(ticker_data)
        
        if metrics:
            current_price = ticker_data.iloc[0]['price']
            
            results.append({
                'ticker': ticker,
                'current_price': current_price,
                'expected_return': metrics['expected_return'] * 252,  # Annualized
                'volatility': metrics['predicted_vol'],
                'sharpe': metrics['sharpe'],
                'direction_accuracy': metrics['direction_accuracy'],
                'train_size': metrics['train_size'],
                'test_size': metrics['test_size']
            })
    
    if not results:
        return {'recommendations': [], 'summary': 'Insufficient data for modeling'}
    
    # Portfolio optimization
    expected_returns = np.array([r['expected_return'] for r in results])
    vols = np.array([r['volatility'] for r in results])
    
    # Simplified covariance (diagonal, can be enhanced with real correlations)
    cov_matrix = np.diag(vols ** 2)
    
    tickers_list = [r['ticker'] for r in results]
    portfolio = optimizer.optimize_portfolio(expected_returns, cov_matrix, tickers_list)
    
    # Combine results
    recommendations = []
    for i, result in enumerate(results):
        weight = portfolio['weights'][i] * 100 if portfolio else 0
        
        # Determine allocation
        if weight > 15:
            allocation = 'OVERWEIGHT'
        elif weight > 10:
            allocation = 'MARKET_WEIGHT'
        elif weight > 5:
            allocation = 'UNDERWEIGHT'
        else:
            allocation = 'AVOID'
        
        recommendations.append({
            'ticker': result['ticker'],
            'current_price': result['current_price'],
            'weight': round(weight, 1),
            'allocation': allocation,
            'expected_return': round(result['expected_return'] * 100, 2),
            'volatility': round(result['volatility'] * 100, 2),
            'sharpe': round(result['sharpe'], 2),
            'direction_accuracy': round(result['direction_accuracy'] * 100, 1),
            'model_metrics': {
                'train_samples': result['train_size'],
                'test_samples': result['test_size']
            }
        })
    
    # Sort by weight
    recommendations.sort(key=lambda x: x['weight'], reverse=True)
    
    # Summary
    if portfolio:
        summary = f"Optimized portfolio across {len(recommendations)} stocks using ML prediction models (Ridge + Random Forest ensemble). Portfolio expected return: {portfolio['expected_return']*100:.2f}%, volatility: {portfolio['volatility']*100:.2f}%, Sharpe: {portfolio['sharpe']:.2f}. Top holdings: {', '.join([r['ticker'] for r in recommendations[:3]])}."
    else:
        summary = f"Analyzed {len(recommendations)} stocks with ML models. Optimization failed."
    
    return {
        'recommendations': recommendations,
        'summary': summary,
        'portfolio_metrics': {
            'expected_return_pct': round(portfolio['expected_return'] * 100, 2) if portfolio else 0,
            'volatility_pct': round(portfolio['volatility'] * 100, 2) if portfolio else 0,
            'sharpe_ratio': round(portfolio['sharpe'], 2) if portfolio else 0
        } if portfolio else None,
        'methodology': 'Mean-variance optimization with ML-based return predictions',
        'analysis_date': datetime.now().isoformat()
    }
