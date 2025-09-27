from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, jsonify, session, send_file
import os
import pandas as pd
from datetime import datetime
from werkzeug.utils import secure_filename
import json
from demand_forecasting import forecast_demand
from db_integration import InventoryDBManager, db_manager
import io

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change this in production

# Initialize database
db_manager = InventoryDBManager()
db_manager.init_app(app)
db = db_manager.db  # This will be the initialized database connection

# Ensure upload and output directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Ensure the forecast plots directory exists in the static folder
forecast_plots_dir = os.path.join('static', 'img', 'forecast_plots')
os.makedirs(forecast_plots_dir, exist_ok=True)

# Update the OUTPUT_FOLDER to point to the static directory for plots
app.config['PLOTS_FOLDER'] = forecast_plots_dir

# Ensure the instance folder exists
try:
    os.makedirs(app.instance_path, exist_ok=True)
    print(f"Instance directory: {os.path.abspath(app.instance_path)}")
except OSError as e:
    print(f"Error creating instance directory: {e}")

# Test database connection
with app.app_context():
    try:
        if not db or not db.conn:
            raise RuntimeError("Failed to initialize database connection")
        print("Database connection test successful")
    except Exception as e:
        print(f"Database connection test failed: {e}")
        raise

# Allowed file extensions
ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'GET':
        return redirect(url_for('index'))
    
    print("\n=== Upload Request ===")
    print(f"Form data: {request.form}")
    print(f"Files in request: {request.files}")
    
    # Debug: Print all form fields and files
    for key, value in request.form.items():
        print(f"Form field - {key}: {value}")
    
    for key, file in request.files.items():
        print(f"File field - {key}: {file.filename if file else 'None'}")
    
    try:
        # Check database connection
        if not db or not db.conn:
            error_msg = 'Database connection not initialized. Please try again.'
            print(error_msg)
            flash(error_msg)
            return jsonify({'error': error_msg}), 500
            
        # Check if the post request has the files
        if 'inventory_file' not in request.files or 'purchase_file' not in request.files or 'starting_file' not in request.files:
            error_msg = 'Missing one or more required files in the request'
            print(error_msg)
            flash(error_msg)
            return redirect(url_for('index'))

        inventory_file = request.files['inventory_file']
        purchase_file = request.files['purchase_file']
        starting_file = request.files['starting_file']
        
        # Check if files are selected
        if inventory_file.filename == '' or purchase_file.filename == '' or starting_file.filename == '':
            error_msg = 'No files selected'
            print(error_msg)
            flash(error_msg)
            return redirect(url_for('index'))
            
        # Check file extensions
        if not (allowed_file(inventory_file.filename) and allowed_file(purchase_file.filename) and allowed_file(starting_file.filename)):
            error_msg = 'Invalid file type. Please upload CSV files only.'
            print(error_msg)
            flash(error_msg)
            return redirect(url_for('index'))
            
        print("\n=== File Information ===")
        print(f"Inventory file: {inventory_file.filename}")
        print(f"Purchase file: {purchase_file.filename}")
        print(f"Starting inventory file: {starting_file.filename}")
        
        # Ensure upload directory exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        # Save uploaded files with secure filenames
        inventory_path = os.path.join(app.config['UPLOAD_FOLDER'], 'inventory_timeseries.csv')
        purchase_path = os.path.join(app.config['UPLOAD_FOLDER'], 'purchase_orders.csv')
        starting_inv_path = os.path.join(app.config['UPLOAD_FOLDER'], 'starting_inventory.csv')
        
        try:
            inventory_file.save(inventory_path)
            purchase_file.save(purchase_path)
            starting_file.save(starting_inv_path)
            print(f"Files saved to: {app.config['UPLOAD_FOLDER']}")
        except Exception as e:
            error_msg = f'Error saving files: {str(e)}'
            print(error_msg)
            flash(error_msg)
            return redirect(url_for('index'))
            
        # Process the files
        try:
            # Load data into database
            print("\n=== Loading Data into Database ===")
            db.load_inventory_data(inventory_path)
            db.load_purchase_orders(purchase_path)
            print("Data loaded successfully")
            
            # Load starting inventory
            starting_inventory = {}
            try:
                df = pd.read_csv(starting_inv_path)
                df.columns = [col.strip().lower() for col in df.columns]
                for _, row in df.iterrows():
                    component_name = str(row['component_name']).strip()
                    starting_inventory[component_name] = int(float(row.get('stock_level', 0)))
                print("Starting inventory loaded")
            except Exception as e:
                print(f"Error loading starting inventory: {e}")
                raise
            
            # Run forecast for each component
            print("\n=== Starting Forecast ===")
            inventory_df = pd.read_csv(inventory_path)
            inventory_df.columns = [col.strip().lower() for col in inventory_df.columns]
            components = inventory_df['component_name'].unique()
            print(f"Found {len(components)} unique components")
            
            # Load purchase orders for forecasting
            po_df = pd.read_csv(purchase_path)
            po_df.columns = [col.strip().lower() for col in po_df.columns]
            
            forecast_results = {}
            for component in components:
                print(f"\nProcessing component: {component}")
                
                # Filter data for current component
                component_po = po_df[po_df['component_name'] == component].copy()
                
                if component_po.empty:
                    print(f"Skipping {component}: No purchase order data")
                    continue
                    
                try:
                    # Ensure the component_po has the required columns with correct names
                    forecast_df = component_po.rename(columns={
                        'date': 'date',
                        'component_name': 'component_name',
                        'quantity': 'order_quantity'  # This should match what forecast_demand expects
                    })
                    
                    # Ensure date is in datetime format with dayfirst=True for DD-MM-YYYY format
                    forecast_df['date'] = pd.to_datetime(forecast_df['date'], dayfirst=True, errors='coerce')
                    
                    # Drop any rows with invalid dates
                    forecast_df = forecast_df.dropna(subset=['date'])
                    
                    if forecast_df.empty:
                        print(f"No valid date data for {component}")
                        continue
                    
                    # Call forecast_demand with proper parameters
                    forecast_result = forecast_demand(
                        df=forecast_df,
                        date_col='date',
                        component_col='component_name',
                        order_col='order_quantity',
                        forecast_days=90,
                        test_days=30
                    )
                    
                    if forecast_result and len(forecast_result) > 0 and not forecast_result[0].future_forecast.empty:
                        print(f"Forecast generated for {component}")
                        
                        # Add recommended_stock column if it doesn't exist
                        forecast_data = forecast_result[0].future_forecast.copy()
                        if 'recommended_stock' not in forecast_data.columns:
                            # Set recommended_stock to be 20% higher than forecast
                            forecast_data['recommended_stock'] = (forecast_data['forecast'] * 1.2).round().astype(int)
                        
                        print(f"Forecast columns: {forecast_data.columns.tolist()}")
                        print(f"Forecast data sample:\n{forecast_data.head()}")
                        
                        # Save forecast to database with the updated dataframe
                        db.save_forecast(component, forecast_data)
                        
                        # Save forecast plot
                        plot_filename = f'demand_forecast_{component}.png'
                        plot_path = os.path.join(app.config['PLOTS_FOLDER'], plot_filename) 
                        
                        # Generate and save plot
                        plot_success = save_forecast_plot(component, forecast_data, plot_path)
                        if plot_success:
                            forecast_results[component] = plot_filename
                            print(f"Plot saved: {plot_path}")
                        else:
                            print(f"Failed to save plot for {component}")
                    else:
                        print(f"No forecast data generated for {component}")
                except Exception as e:
                    print(f"Error forecasting for {component}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            print("\n=== Forecast Complete ===")
            print(f"Processed {len(forecast_results)} components")
            
            # Store results in session
            session['forecast_results'] = forecast_results
            session['starting_inventory'] = starting_inventory
            
            return redirect(url_for('results'))
            
        except Exception as e:
            error_msg = f'Error processing files: {str(e)}'
            print(f"\n=== ERROR ===\n{error_msg}")
            import traceback
            traceback.print_exc()
            flash(error_msg)
            return redirect(url_for('index'))
            
    except Exception as e:
        error_msg = f'An error occurred: {str(e)}'
        print(f"\n=== UNEXPECTED ERROR ===\n{error_msg}")
        import traceback
        traceback.print_exc()
        flash(error_msg)
        return redirect(url_for('index'))

# Helper function to save forecast plots
def save_forecast_plot(component_name, forecast_data, output_path):
    """
    Generate and save a plot of the forecast results.
    
    Args:
        component_name (str): Name of the component
        forecast_data (pd.DataFrame): DataFrame containing forecast data with 'ds' and 'yhat' columns
        output_path (str): Path to save the plot
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Debug: Print column names to understand the structure
        print(f"\n=== Debug: Plotting for {component_name} ===")
        print("Available columns:", forecast_data.columns.tolist())
        print("Data sample:", forecast_data.head() if not forecast_data.empty else 'Empty DataFrame')
        
        # Set the style
        sns.set(style="whitegrid")
        
        # Create figure and axis
        plt.figure(figsize=(12, 6))
        
        # Check if we have date and forecast columns (handle different column naming)
        date_col = 'date' if 'date' in forecast_data.columns else 'ds'
        forecast_col = 'forecast' if 'forecast' in forecast_data.columns else 'yhat'
        
        if date_col not in forecast_data.columns:
            raise ValueError(f"Date column ('ds' or 'date') not found in forecast data")
        if forecast_col not in forecast_data.columns:
            raise ValueError(f"Forecast column ('yhat' or 'forecast') not found in forecast data")
        
        # Plot the forecast
        plt.plot(forecast_data[date_col], forecast_data[forecast_col], 
                label='Forecast', color='#1f77b4')
        
        # Add confidence interval if available
        lower_col = 'yhat_lower' if 'yhat_lower' in forecast_data.columns else None
        upper_col = 'yhat_upper' if 'yhat_upper' in forecast_data.columns else None
        
        if lower_col and upper_col:
            plt.fill_between(
                forecast_data[date_col],
                forecast_data[lower_col],
                forecast_data[upper_col],
                color='#1f77b4',
                alpha=0.2,
                label='Confidence Interval'
            )
        
        # Customize the plot
        plt.title(f'Demand Forecast for {component_name}', fontsize=14, pad=20)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Demand', fontsize=12)
        plt.legend()
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Save the figure
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Successfully saved forecast plot to: {output_path}")
        return True
        
    except Exception as e:
        error_msg = f"Error generating forecast plot for {component_name}: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return False

@app.route('/results')
def results():
    """Display forecast results for all components."""
    try:
        # Get all components with forecasts
        components = db.get_components_with_forecasts()
        
        if not components:
            flash("No forecasts available. Please upload data first.", "warning")
            return redirect(url_for('index'))
        
        # Get forecast data for each component
        forecasts = {}
        for component in components:
            forecasts[component] = db.get_forecast_data(component)
        
        # Get forecast summary
        forecast_summary = db.get_forecast_summary()
        
        # Get current stock levels
        current_stock = {}
        starting_inv_path = os.path.join(app.config['UPLOAD_FOLDER'], 'starting_inventory.csv')
        if os.path.exists(starting_inv_path):
            try:
                stock_df = pd.read_csv(starting_inv_path)
                stock_df.columns = [col.strip().lower() for col in stock_df.columns]
                if 'component_name' in stock_df.columns and 'stock_level' in stock_df.columns:
                    stock_df['stock_level'] = pd.to_numeric(
                        stock_df['stock_level'], 
                        errors='coerce'
                    ).fillna(0).astype(int)
                    current_stock = dict(zip(stock_df['component_name'], stock_df['stock_level']))
            except Exception as e:
                print(f"Error loading current stock: {e}")
        
        # Prepare plot paths for the template
        plot_paths = {
            component: url_for('static', filename=f'img/forecast_plots/demand_forecast_{component}.png')
            for component in components
        }
        
        return render_template(
            'results.html',
            components=components,
            forecasts=forecasts,
            forecast_summary=forecast_summary,
            current_stock=current_stock,
            plot_paths=plot_paths,
            now=datetime.now()
        )
        
    except Exception as e:
        flash(f"Error loading results: {str(e)}", "error")
        print(f"Error in results route: {e}")
        return redirect(url_for('index'))

@app.route('/download/<component>/<format>')
def download_forecast(component, format):
    """
    Download forecast data in the specified format.
    
    Args:
        component (str): Name of the component
        format (str): Format to download (csv or xlsx)
    """
    db = db_manager.db
    
    # Get forecast data as DataFrame
    df = db.get_forecasts_dataframe(component)
    
    if df.empty:
        flash(f"No forecast data found for {component}", "error")
        return redirect(url_for('results'))
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{component}_forecast_{timestamp}"
    
    # Create in-memory file
    output = io.BytesIO()
    
    try:
        if format == 'csv':
            csv_data = df.to_csv(index=False, encoding='utf-8')
            output.write(csv_data.encode('utf-8'))
            mimetype = 'text/csv'
            extension = 'csv'
        elif format == 'xlsx':
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Forecast')
            mimetype = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            extension = 'xlsx'
        else:
            flash("Invalid format specified", "error")
            return redirect(url_for('results'))
        
        # Prepare file for download
        output.seek(0)
        return send_file(
            output,
            mimetype=mimetype,
            as_attachment=True,
            download_name=f"{filename}.{extension}",
            conditional=True
        )
        
    except Exception as e:
        print(f"Error generating {format} file: {e}")
        flash(f"Error generating {format} file: {e}", "error")
        return redirect(url_for('results'))

@app.route('/upload_success')
def upload_success():
    return redirect(url_for('results'))

if __name__ == '__main__':
    app.run(debug=True)
