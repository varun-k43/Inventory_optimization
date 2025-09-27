import sqlite3
import pandas as pd
from datetime import datetime
import os

class InventoryDB:
    def __init__(self, db_path='inventory_optimization.db'):
        """Initialize database connection and create tables if they don't exist."""
        self.db_path = db_path
        self.conn = None
        try:
            self._connect()
            self._create_tables()
            print(f"Successfully connected to database at: {os.path.abspath(self.db_path)}")
        except Exception as e:
            print(f"Failed to initialize database: {e}")
            raise

    def _connect(self):
        """Establish database connection."""
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
            # Add check_same_thread=False for SQLite threading
            self.conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=30.0
            )
            self.conn.row_factory = sqlite3.Row  # Enable dictionary-style access
            print(f"Database connection established at: {os.path.abspath(self.db_path)}")
            
            # Test the connection
            cursor = self.conn.cursor()
            cursor.execute('SELECT SQLITE_VERSION()')
            data = cursor.fetchone()
            print(f"SQLite version: {data[0]}")
            
        except Exception as e:
            print(f"Error connecting to database at {self.db_path}: {e}")
            self.conn = None
            raise

    def _create_tables(self):
        """Create database tables if they don't exist."""
        cursor = self.conn.cursor()
        
        # Components table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS components (
            component_id INTEGER PRIMARY KEY AUTOINCREMENT,
            component_name TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Inventory table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS inventory (
            date DATE NOT NULL,
            component_id INTEGER NOT NULL,
            stock_level INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (date, component_id),
            FOREIGN KEY (component_id) REFERENCES components(component_id)
        )
        ''')
        
        # Purchase orders table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS purchase_orders (
            order_id INTEGER PRIMARY KEY AUTOINCREMENT,
            date DATE NOT NULL,
            component_id INTEGER NOT NULL,
            order_quantity INTEGER NOT NULL,
            status TEXT DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (component_id) REFERENCES components(component_id)
        )
        ''')
        
        # Forecasts table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS forecasts (
            forecast_id INTEGER PRIMARY KEY AUTOINCREMENT,
            component_id INTEGER NOT NULL,
            date DATE NOT NULL,
            forecasted_demand REAL NOT NULL,
            recommended_stock REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (component_id) REFERENCES components(component_id),
            UNIQUE(component_id, date)
        )
        ''')
        
        # Create indexes for better query performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_forecasts_component_date ON forecasts(component_id, date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_inventory_component_date ON inventory(component_id, date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_purchase_orders_component_date ON purchase_orders(component_id, date)')
        
        self.conn.commit()

    def get_or_create_component(self, component_name):
        """Get component ID if exists, otherwise create new component."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT component_id FROM components WHERE component_name = ?', (component_name,))
        result = cursor.fetchone()
        
        if result:
            return result[0]
        else:
            cursor.execute('INSERT INTO components (component_name) VALUES (?)', (component_name,))
            self.conn.commit()
            return cursor.lastrowid

    def load_inventory_data(self, filepath):
        """Load inventory data from CSV file."""
        df = pd.read_csv(filepath)
        cursor = self.conn.cursor()
        
        for _, row in df.iterrows():
            component_id = self.get_or_create_component(row['component_name'])
            cursor.execute('''
                INSERT OR REPLACE INTO inventory (date, component_id, stock_level)
                VALUES (?, ?, ?)
            ''', (row['date'], component_id, row['stock_level']))
        
        self.conn.commit()
        return len(df)

    def load_purchase_orders(self, filepath):
        """Load purchase orders from CSV file."""
        try:
            # Read the CSV file
            df = pd.read_csv(filepath)
            
            # Normalize column names (convert to lowercase and strip whitespace)
            df.columns = [col.strip().lower() for col in df.columns]
            
            # Check for required columns
            required_columns = {'component_name', 'date', 'quantity'}
            missing_columns = required_columns - set(df.columns)
            if missing_columns:
                raise ValueError(f"Missing required columns in purchase orders file: {', '.join(missing_columns)}")
            
            cursor = self.conn.cursor()
            records_processed = 0
            
            for _, row in df.iterrows():
                try:
                    component_name = str(row['component_name']).strip()
                    if not component_name:
                        print("Warning: Empty component name in row, skipping")
                        continue
                        
                    component_id = self.get_or_create_component(component_name)
                    
                    # Handle different date formats
                    order_date = pd.to_datetime(row['date']).strftime('%Y-%m-%d')
                    quantity = int(float(row['quantity']))  # Handle potential float values
                    
                    cursor.execute('''
                        INSERT INTO purchase_orders (date, component_id, order_quantity)
                        VALUES (?, ?, ?)
                    ''', (order_date, component_id, quantity))
                    
                    records_processed += 1
                    
                except Exception as e:
                    print(f"Error processing row {_}: {e}")
                    continue
            
            self.conn.commit()
            print(f"Successfully loaded {records_processed} purchase orders")
            return records_processed
            
        except Exception as e:
            self.conn.rollback()
            print(f"Error loading purchase orders: {e}")
            raise

    def save_forecast(self, component_name, forecast_data):
        """
        Save forecast data to the database.
        
        Args:
            component_name (str): Name of the component
            forecast_data (pd.DataFrame): DataFrame containing forecast data with 'date' and 'forecast' columns
        """
        if not hasattr(self, 'conn') or not self.conn:
            print("Database connection not established")
            return False
            
        try:
            cursor = self.conn.cursor()
            
            # Get component ID
            cursor.execute('SELECT component_id FROM components WHERE component_name = ?', (component_name,))
            component_row = cursor.fetchone()
            
            if not component_row:
                print(f"Component '{component_name}' not found in database")
                return False
                
            component_id = component_row[0]
            
            # Delete existing forecasts for this component
            cursor.execute('DELETE FROM forecasts WHERE component_id = ?', (component_id,))
            
            # Insert or update forecast data
            for _, row in forecast_data.iterrows():
                # Convert Timestamp to string in YYYY-MM-DD format
                date_str = row['date'].strftime('%Y-%m-%d')
                forecast = row['forecast']
                recommended_stock = row.get('recommended_stock', int(forecast * 1.2))  # Default to 20% higher than forecast
                
                cursor.execute('''
                    INSERT INTO forecasts 
                    (component_id, date, forecasted_demand, recommended_stock, created_at)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (component_id, date_str, float(forecast), int(recommended_stock)))
            
            self.conn.commit()
            print(f"Saved forecast for {component_name}")
            return True
            
        except Exception as e:
            print(f"Error saving forecast for {component_name}: {e}")
            import traceback
            traceback.print_exc()
            if self.conn:
                self.conn.rollback()
            return False

    def get_component_forecast(self, component_name):
        """Get forecast data for a specific component."""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT f.date, f.forecasted_demand, f.recommended_stock
            FROM forecasts f
            JOIN components c ON f.component_id = c.component_id
            WHERE c.component_name = ?
            ORDER BY f.date
        ''', (component_name,))
        
        return cursor.fetchall()

    def get_current_inventory(self, component_name=None):
        """Get current inventory levels."""
        cursor = self.conn.cursor()
        
        if component_name:
            cursor.execute('''
                SELECT c.component_name, i.date, i.stock_level
                FROM inventory i
                JOIN components c ON i.component_id = c.component_id
                WHERE c.component_name = ?
                ORDER BY i.date DESC
                LIMIT 1
            ''', (component_name,))
        else:
            # Get latest inventory for all components
            cursor.execute('''
                SELECT c.component_name, i.date, i.stock_level
                FROM (
                    SELECT component_id, MAX(date) as max_date
                    FROM inventory
                    GROUP BY component_id
                ) latest
                JOIN inventory i ON i.component_id = latest.component_id AND i.date = latest.max_date
                JOIN components c ON i.component_id = c.component_id
                ORDER BY c.component_name
            ''')
        
        return cursor.fetchall()

    def get_forecast_data(self, component_name):
        """
        Fetch forecast data for a specific component.
        
        Args:
            component_name (str): Name of the component
            
        Returns:
            list: List of dictionaries containing forecast data
        """
        if not hasattr(self, 'conn') or not self.conn:
            print("Database connection not established")
            return []
            
        try:
            cursor = self.conn.cursor()
            
            # Get component ID
            cursor.execute('''
                SELECT f.date, f.forecasted_demand, f.recommended_stock
                FROM forecasts f
                JOIN components c ON f.component_id = c.component_id
                WHERE c.component_name = ?
                ORDER BY f.date
            ''', (component_name,))
            
            forecasts = []
            for row in cursor.fetchall():
                forecasts.append({
                    'date': row[0],
                    'forecast': row[1],
                    'recommended_stock': row[2]
                })
                
            return forecasts
            
        except Exception as e:
            print(f"Error fetching forecast for {component_name}: {e}")
            return []
            
    def get_components_with_forecasts(self):
        """
        Get a list of all components that have forecast data.
        
        Returns:
            list: List of component names
        """
        if not hasattr(self, 'conn') or not self.conn:
            print("Database connection not established")
            return []
            
        try:
            cursor = self.conn.cursor()
            
            cursor.execute('''
                SELECT DISTINCT c.component_name
                FROM components c
                JOIN forecasts f ON c.component_id = f.component_id
                ORDER BY c.component_name
            ''')
            
            return [row[0] for row in cursor.fetchall()]
            
        except Exception as e:
            print(f"Error fetching components with forecasts: {e}")
            return []
            
    def get_forecasts_dataframe(self, component_name):
        """
        Get forecast data as a pandas DataFrame for a specific component.
        
        Args:
            component_name (str): Name of the component
            
        Returns:
            pd.DataFrame: DataFrame containing forecast data
        """
        import pandas as pd
        
        forecasts = self.get_forecast_data(component_name)
        if not forecasts:
            return pd.DataFrame()
            
        df = pd.DataFrame(forecasts)
        df['date'] = pd.to_datetime(df['date'])
        return df

    def get_forecast_summary(self):
        """
        Get summary statistics for all forecasts.
        
        Returns:
            dict: Dictionary with component names as keys and summary statistics as values
        """
        if not hasattr(self, 'conn') or not self.conn:
            print("Database connection not established")
            return {}
            
        try:
            cursor = self.conn.cursor()
            
            # Get forecast summary for each component
            cursor.execute('''
                SELECT 
                    c.component_name,
                    COUNT(f.forecast_id) as forecast_count,
                    MIN(f.date) as first_forecast_date,
                    MAX(f.date) as last_forecast_date,
                    AVG(f.forecasted_demand) as avg_forecast,
                    AVG(f.recommended_stock) as avg_recommended_stock
                FROM forecasts f
                JOIN components c ON f.component_id = c.component_id
                GROUP BY c.component_name
                ORDER BY c.component_name
            ''')
            
            summary = {}
            for row in cursor.fetchall():
                component_name = row[0]
                summary[component_name] = {
                    'forecast_count': row[1],
                    'first_forecast_date': row[2],
                    'last_forecast_date': row[3],
                    'avg_forecast': row[4],
                    'avg_recommended_stock': row[5]
                }
                
            return summary
            
        except Exception as e:
            print(f"Error getting forecast summary: {e}")
            return {}

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()

# Example usage
if __name__ == "__main__":
    # Initialize database
    db = InventoryDB()
    
    try:
        # Load data from CSV files (update paths as needed)
        print("Loading data from CSV files...")
        inventory_count = db.load_inventory_data('inventory_timeseries.csv')
        orders_count = db.load_purchase_orders('purchase_orders.csv')
        
        print(f"Loaded {inventory_count} inventory records")
        print(f"Loaded {orders_count} purchase orders")
        
        # Example: Get current inventory
        print("\nCurrent Inventory:")
        for row in db.get_current_inventory():
            print(f"{row['component_name']}: {row['stock_level']} units (as of {row['date']})")
        
        # Example: Get forecast for a component
        component = "Resistor"
        print(f"\nForecast for {component}:")
        forecasts = db.get_component_forecast(component)
        if forecasts:
            for f in forecasts[:5]:  # Show first 5 forecast entries
                print(f"{f['date']}: {f['forecasted_demand']:.2f} (recommended: {f['recommended_stock']})")
        else:
            print(f"No forecast data available for {component}")
            
    finally:
        db.close()
        print("\nDatabase connection closed.")
