from db_operations import InventoryDB
import pandas as pd
import os
from datetime import datetime, timedelta

class InventoryDBManager:
    def __init__(self, app=None):
        self.db = None
        if app is not None:
            self.init_app(app)
    
    def init_app(self, app):
        """Initialize the database with Flask app context."""
        try:
            db_path = os.path.join(app.instance_path, 'inventory_optimization.db')
            print(f"Initializing database at: {os.path.abspath(db_path)}")
            self.db = InventoryDB(db_path)
            
            # Test the connection
            if not self.db or not self.db.conn:
                raise RuntimeError("Failed to initialize database connection")
                
            print(f"Successfully initialized database at: {os.path.abspath(db_path)}")
            
            # Ensure the instance folder exists
            try:
                os.makedirs(app.instance_path, exist_ok=True)
                print(f"Instance directory: {os.path.abspath(app.instance_path)}")
            except OSError as e:
                print(f"Error creating instance directory: {e}")
                raise
                
            # Register with the app
            app.extensions['db'] = self.db
            
        except Exception as e:
            print(f"Error initializing database: {e}")
            self.db = None
            raise
    
    def save_forecast_results(self, component_name, forecast_df):
        """
        Save forecast results to the database.
        
        Args:
            component_name (str): Name of the component
            forecast_df (pd.DataFrame): DataFrame with 'date' and 'forecast' columns
        """
        if not self.db:
            raise RuntimeError("Database not initialized. Call init_app first.")
            
        try:
            # Create a copy to avoid modifying the original
            forecast_data = forecast_df.copy()
            
            # Ensure date is in the correct format
            if 'date' in forecast_data.columns and not pd.api.types.is_datetime64_any_dtype(forecast_data['date']):
                forecast_data['date'] = pd.to_datetime(forecast_data['date'])
            
            # Prepare data for database insertion
            records = []
            for _, row in forecast_data.iterrows():
                records.append({
                    'date': row['date'],
                    'forecasted_demand': float(row['forecast']),
                    'recommended_stock': float(row['forecast'] * 1.5),  # 1.5x forecast as buffer
                    'component_name': component_name,
                    'created_at': datetime.now()
                })
            
            # Save to database
            if records:
                self.db.conn.executemany('''
                    INSERT INTO forecasts 
                    (date, forecasted_demand, recommended_stock, component_name, created_at)
                    VALUES (:date, :forecasted_demand, :recommended_stock, :component_name, :created_at)
                ''', records)
                self.db.conn.commit()
                print(f"Saved {len(records)} forecast records for {component_name}")
                return True
            return False
            
        except Exception as e:
            print(f"Error saving forecast for {component_name}: {str(e)}")
            self.db.conn.rollback()
            raise
    
    def get_forecast_summary(self, component_name=None):
        """Get forecast summary for the dashboard."""
        if not self.db:
            raise RuntimeError("Database not initialized. Call init_app first.")
            
        cursor = self.db.conn.cursor()
        
        if component_name:
            # Get summary for specific component
            cursor.execute('''
                SELECT 
                    c.component_name,
                    SUM(CASE WHEN f.date <= date('now', '+30 days') THEN f.forecasted_demand ELSE 0 END) as forecast_30d,
                    SUM(CASE WHEN f.date <= date('now', '+60 days') THEN f.forecasted_demand ELSE 0 END) as forecast_60d,
                    SUM(f.forecasted_demand) as forecast_90d,
                    MAX(f.recommended_stock) as recommended_stock
                FROM forecasts f
                JOIN components c ON f.component_id = c.component_id
                WHERE c.component_name = ?
                GROUP BY c.component_name
            ''', (component_name,))
        else:
            # Get summary for all components
            cursor.execute('''
                SELECT 
                    c.component_name,
                    SUM(CASE WHEN f.date <= date('now', '+30 days') THEN f.forecasted_demand ELSE 0 END) as forecast_30d,
                    SUM(CASE WHEN f.date <= date('now', '+60 days') THEN f.forecasted_demand ELSE 0 END) as forecast_60d,
                    SUM(f.forecasted_demand) as forecast_90d,
                    MAX(f.recommended_stock) as recommended_stock
                FROM forecasts f
                JOIN components c ON f.component_id = c.component_id
                GROUP BY c.component_name
            ''')
        
        columns = [desc[0] for desc in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        # Convert to component-based dictionary
        summary = {}
        for row in results:
            component = row.pop('component_name')
            summary[component] = {
                '30d_forecast': row['forecast_30d'],
                '60d_forecast': row['forecast_60d'],
                '90d_forecast': row['forecast_90d'],
                'reorder_qty': row['recommended_stock']
            }
            
        return summary[component_name] if component_name and summary else summary
    
    def get_accuracy_metrics(self):
        """Calculate accuracy metrics from historical data."""
        if not self.db:
            raise RuntimeError("Database not initialized. Call init_app first.")
            
        cursor = self.db.conn.cursor()
        
        cursor.execute('''
            SELECT 
                c.component_name,
                AVG(ABS(i.stock_level - f.forecasted_demand)) as mae,
                AVG(ABS(i.stock_level - f.forecasted_demand) * 100.0 / NULLIF(i.stock_level, 0)) as mape,
                SQRT(AVG(POWER(i.stock_level - f.forecasted_demand, 2))) as rmse,
                COUNT(*) as samples
            FROM inventory i
            JOIN forecast f ON i.component_id = f.component_id AND i.date = f.date
            JOIN components c ON i.component_id = c.component_id
            GROUP BY c.component_name
        ''')
        
        metrics = {}
        for row in cursor.fetchall():
            metrics[row['component_name']] = {
                'mae': round(row['mae'], 2),
                'mape': round(row['mape'], 2) if row['mape'] is not None else None,
                'rmse': round(row['rmse'], 2),
                'samples': row['samples']
            }
            
        return metrics
    
    def close(self):
        """Close the database connection."""
        if self.db:
            self.db.close()

# Initialize the database manager
db_manager = InventoryDBManager()

def init_app(app):
    """Initialize the database with the Flask app."""
    db_manager.init_app(app)
    
    # Ensure tables are created when the app starts
    with app.app_context():
        db_manager.db._create_tables()
    
    # Register teardown function
    @app.teardown_appcontext
    def shutdown_session(exception=None):
        if hasattr(db_manager, 'db') and db_manager.db and db_manager.db.conn:
            try:
                db_manager.db.conn.close()
                db_manager.db.conn = None
            except Exception as e:
                print(f"Error closing database connection: {e}")
    
    return db_manager
