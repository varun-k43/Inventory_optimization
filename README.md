# Inventory Optimization and Demand Forecasting System

A web-based application for inventory optimization and demand forecasting of electronic components (resistors, capacitors, ICs, etc.). This system helps businesses maintain optimal inventory levels by analyzing historical data and predicting future demand.

## Features

- **Data Upload**: Upload your inventory and order history in CSV format
- **Data Processing**: Automatic handling of missing values and data normalization
- **Demand Forecasting**: Predict future demand using Prophet and scikit-learn models
- **Visual Analytics**: Interactive charts and tables for data exploration
- **Inventory Recommendations**: Get reorder recommendations with buffer calculations
- **Export Results**: Download forecasts and analysis in CSV format

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd inventory-optimization
   ```

2. **Create a virtual environment** (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```
inventory-optimization/
├── app.py                 # Main Flask application
├── demand_forecasting.py  # Core forecasting logic
├── requirements.txt       # Python dependencies
├── uploads/               # Directory for uploaded files
├── output/                # Directory for generated outputs
│   ├── forecast_plots/    # Generated forecast plots
│   └── stock_vs_orders/   # Stock vs orders comparison plots
├── static/                # Static files (CSS, JS, images)
└── templates/             # HTML templates
    ├── base.html          # Base template
    ├── index.html         # Upload form
    └── results.html       # Results dashboard
```

## Usage

1. **Start the application**:

   ```bash
   python app.py
   ```

2. **Access the web interface**:
   Open your browser and go to `http://localhost:5000`

3. **Upload your data files**:

   - **Inventory Timeseries CSV**: Contains historical inventory levels
     - Required columns: `date`, `component_name`, `stock_level`
   - **Purchase Orders CSV**: Contains historical order data
     - Required columns: `date`, `component_name`, `order_quantity`
   - **Starting Inventory CSV**: Contains initial stock levels
     - Required columns: `component_name`, `stock_level`

4. **View and analyze results**:
   - Interactive forecast plots
   - Stock vs orders comparison
   - Forecast accuracy metrics
   - Reorder recommendations

## Data Format

### Inventory Timeseries CSV

```csv
date,component_name,stock_level
2023-01-01,Resistor,1000
2023-01-02,Resistor,950
2023-01-01,Capacitor,500
```

### Purchase Orders CSV

```csv
date,component_name,order_quantity
2023-01-05,Resistor,200
2023-01-10,Capacitor,100
```

### Starting Inventory CSV

```csv
component_name,stock_level
Resistor,1000
Capacitor,500
IC,200
```

## Configuration

You can modify the following settings in `app.py`:

- `UPLOAD_FOLDER`: Directory to store uploaded files (default: 'uploads/')
- `OUTPUT_FOLDER`: Directory to store output files (default: 'output/')
- `MAX_CONTENT_LENGTH`: Maximum file upload size (default: 16MB)

## Deployment

For production deployment, consider using a production WSGI server like Gunicorn:

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For support, please open an issue in the GitHub repository.
