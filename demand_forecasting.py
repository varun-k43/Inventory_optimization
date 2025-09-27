import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Set Agg backend before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg' which doesn't require a display

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Plotting


def load_data(
    inventory_timeseries_path: Path,
    purchase_orders_path: Path,
    starting_inventory_path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    """
    Load the three CSV datasets with proper dtypes and parsed dates.
    Expected columns:
      - inventory_timeseries.csv: date, component_name, stock_level
      - purchase_orders.csv: date, component_name, order_quantity
      - starting_inventory.csv: component_name, stock_level (or initial_stock)
    """
    inv_df = pd.read_csv(inventory_timeseries_path)
    po_df = pd.read_csv(purchase_orders_path)
    start_df = pd.read_csv(starting_inventory_path)

    # Normalize column names
    inv_df.columns = [c.strip().lower() for c in inv_df.columns]
    po_df.columns = [c.strip().lower() for c in po_df.columns]
    start_df.columns = [c.strip().lower() for c in start_df.columns]

    # Detect actual column names present in the CSVs and DO NOT rename
    def _detect_columns(df: pd.DataFrame, kind: str) -> Dict[str, str]:
        cols = {c.strip().lower(): c for c in df.columns}
        mapping: Dict[str, str] = {}

        # date
        for cand in ["date", "order_date", "transaction_date", "day", "dt", "timestamp"]:
            if cand in cols:
                mapping["date_col"] = cols[cand]
                break

        # component identifier
        for cand in ["component_type", "component_name", "component", "item", "item_name", "product", "part", "sku", "sku_id"]:
            if cand in cols:
                mapping["component_col"] = cols[cand]
                break

        if kind == "inventory":
            for cand in ["on_hand", "stock_level", "stock", "quantity", "qty", "stocklevel"]:
                if cand in cols:
                    mapping["stock_col"] = cols[cand]
                    break
        elif kind == "purchase":
            for cand in ["order_quantity", "orders", "demand", "quantity", "qty", "order_qty", "ordered_qty"]:
                if cand in cols:
                    mapping["order_col"] = cols[cand]
                    break
        elif kind == "starting":
            for cand in ["on_hand", "stock_level", "initial_stock", "opening_stock", "starting_stock", "qty", "quantity"]:
                if cand in cols:
                    mapping["stock_col"] = cols[cand]
                    break

        return mapping

    inv_map = _detect_columns(inv_df, "inventory")
    po_map = _detect_columns(po_df, "purchase")
    start_map = _detect_columns(start_df, "starting")

    # Parse dates in-place using detected date columns
    if inv_map.get("date_col"):
        inv_df[inv_map["date_col"]] = pd.to_datetime(inv_df[inv_map["date_col"]], errors="coerce").dt.normalize()
    if po_map.get("date_col"):
        po_df[po_map["date_col"]] = pd.to_datetime(po_df[po_map["date_col"]], errors="coerce").dt.normalize()

    # Build schema used downstream (keep original column names)
    schema = {
        "inv_date": inv_map.get("date_col", "date"),
        "inv_component": inv_map.get("component_col", "component_type"),
        "inv_stock": inv_map.get("stock_col", "on_hand"),
        "po_date": po_map.get("date_col", "date"),
        "po_component": po_map.get("component_col", inv_map.get("component_col", "component_type")),
        "po_order": po_map.get("order_col", "order_quantity"),
        "start_component": start_map.get("component_col", inv_map.get("component_col", "component_type")),
        "start_stock": start_map.get("stock_col", "on_hand"),
    }

    # No renaming of columns; keep originals

    # Validate required columns
    for df, required_cols, name in [
        (inv_df, {schema["inv_date"], schema["inv_component"], schema["inv_stock"]}, "inventory_timeseries"),
        (po_df, {schema["po_date"], schema["po_component"], schema["po_order"]}, "purchase_orders"),
        (start_df, {schema["start_component"], schema["start_stock"]}, "starting_inventory"),
    ]:
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns {missing} in {name} CSV")

    return inv_df, po_df, start_df, schema


def preprocess(
    inv_df: pd.DataFrame,
    po_df: pd.DataFrame,
    start_df: pd.DataFrame,
    schema: Dict[str, str],
) -> pd.DataFrame:
    """
    Merge datasets on date and component_name.
    - Left join the union of dates/components present in either inv_df or po_df
    - Bring in starting inventory as a per-component attribute
    - Fill missing numeric values with zeros where appropriate
    """
    # Aggregate purchase orders in case of duplicates per day
    po_agg = (
        po_df.groupby([schema["po_date"], schema["po_component"]], dropna=False)[schema["po_order"]]
        .sum()
        .reset_index()
    )

    # Aggregate inventory in case of duplicates (use last observed stock per day)
    inv_agg = (
        inv_df.sort_values([schema["inv_component"], schema["inv_date"]])  # ensure chronological order
        .groupby([schema["inv_date"], schema["inv_component"]], dropna=False)[schema["inv_stock"]]
        .last()
        .reset_index()
    )

    # Create a complete index of dates/components present in either source
    key_df = (
        pd.concat([
            inv_agg[[schema["inv_date"], schema["inv_component"]]],
            po_agg[[schema["po_date"], schema["po_component"]]].rename(columns={
                schema["po_date"]: schema["inv_date"],
                schema["po_component"]: schema["inv_component"],
            }),
        ], ignore_index=True)
        .drop_duplicates()
    )

    merged = key_df.merge(inv_agg, on=[schema["inv_date"], schema["inv_component"]], how="left")
    merged = merged.merge(
        po_agg.rename(columns={
            schema["po_date"]: schema["inv_date"],
            schema["po_component"]: schema["inv_component"],
        }),
        on=[schema["inv_date"], schema["inv_component"]],
        how="left",
    )

    # Attach starting inventory per component
    if schema["start_component"] in start_df.columns and schema["start_stock"] in start_df.columns:
        start_comp = start_df[[schema["start_component"], schema["start_stock"]]]
        merged = merged.merge(
            start_comp.rename(columns={
                schema["start_component"]: schema["inv_component"],
                schema["start_stock"]: "__starting_stock__",
            }),
            on=schema["inv_component"],
            how="left",
        )

    # Sort and fill
    merged = merged.sort_values([schema["inv_component"], schema["inv_date"]]).reset_index(drop=True)
    if schema["po_order"] in merged.columns:
        merged[schema["po_order"]] = merged[schema["po_order"]].fillna(0.0)

    # If stock_level missing, forward-fill within component using latest known stock
    merged[schema["inv_stock"]] = (
        merged.groupby(schema["inv_component"])[schema["inv_stock"]].transform("ffill")
    )
    # If still missing (no prior data), use starting stock if available
    if "__starting_stock__" in merged.columns:
        missing_mask = merged[schema["inv_stock"]].isna()
        merged.loc[missing_mask, schema["inv_stock"]] = merged.loc[missing_mask, "__starting_stock__"]

    # Final fallback to zero
    merged[schema["inv_stock"]] = merged[schema["inv_stock"]].fillna(0.0)

    # Return merged with ORIGINAL column names only
    keep_cols = [schema["inv_date"], schema["inv_component"], schema["inv_stock"]]
    if schema["po_order"] in merged.columns:
        keep_cols.append(schema["po_order"])
    merged = merged[keep_cols]
    return merged


def compute_eda(df: pd.DataFrame, component_col: str, stock_col: str, order_col: str) -> pd.DataFrame:
    """
    Compute summary statistics per component: min/max/mean for demand (order_quantity) and stock_level.
    Returns a DataFrame with one row per component.
    """
    agg = df.groupby(component_col).agg(
        **{
            f"min_{order_col}": (order_col, "min"),
            f"max_{order_col}": (order_col, "max"),
            f"mean_{order_col}": (order_col, "mean"),
            f"min_{stock_col}": (stock_col, "min"),
            f"max_{stock_col}": (stock_col, "max"),
            f"mean_{stock_col}": (stock_col, "mean"),
            "observations": (order_col, "size"),
        }
    )
    return agg.reset_index()


def save_eda_outputs(eda_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    eda_csv = output_dir / "eda_summary_by_component.csv"
    eda_df.to_csv(eda_csv, index=False)


def plot_stock_vs_orders(df: pd.DataFrame, output_dir: Path, date_col: str, component_col: str, stock_col: str, order_col: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for component, grp in df.groupby(component_col):
        fig, ax1 = plt.subplots(figsize=(10, 5))

        ax1.set_title(f"Stock vs Orders Over Time - {component}")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Stock Level", color="tab:blue")
        ax1.plot(grp[date_col], grp[stock_col], color="tab:blue", label=stock_col)
        ax1.tick_params(axis="y", labelcolor="tab:blue")

        ax2 = ax1.twinx()
        ax2.set_ylabel("Order Quantity", color="tab:red")
        ax2.bar(grp[date_col], grp[order_col], color="tab:red", alpha=0.3, label=order_col)
        ax2.tick_params(axis="y", labelcolor="tab:red")

        fig.tight_layout()
        out_path = output_dir / f"stock_vs_orders_{safe_filename(str(component))}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)


def safe_filename(name: str) -> str:
    return (
        name.replace(" ", "_")
        .replace("/", "-")
        .replace("\\", "-")
        .replace(":", "-")
        .replace("|", "-")
    )


@dataclass
class ForecastResult:
    component_name: str  # stores the component identifier value
    model_name: str
    history: pd.DataFrame  # columns: date, y (demand)
    test_actual_vs_pred: pd.DataFrame  # columns: date, actual, predicted
    future_forecast: pd.DataFrame  # columns: date, forecast


def _prepare_demand_series(df: pd.DataFrame, component_name: str, date_col: str, order_col: str, component_col: str) -> pd.DataFrame:
    comp = df[df[component_col] == component_name].copy()
    comp = (
        comp.groupby(date_col, as_index=False)[order_col].sum().sort_values(date_col)
    )
    comp = comp.rename(columns={order_col: "y", date_col: "ds"})
    # Ensure daily continuity for Prophet; fill missing days with 0 demand
    if not comp.empty:
        full_idx = pd.date_range(comp["ds"].min(), comp["ds"].max(), freq="D")
        comp = comp.set_index("ds").reindex(full_idx).fillna(0.0).rename_axis("ds").reset_index()
    return comp


def _train_prophet(train_df: pd.DataFrame):
    try:
        from prophet import Prophet  # type: ignore
    except Exception:  # pragma: no cover
        return None, "prophet_unavailable"

    m = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False)
    m.fit(train_df)
    return m, "Prophet"


def _train_linear_regression(train_df: pd.DataFrame):
    from sklearn.linear_model import LinearRegression

    X = _time_features(train_df["ds"])  # shape (n, k)
    y = train_df["y"].values
    model = LinearRegression()
    model.fit(X, y)
    return model, "LinearRegression"


def _time_features(ds: pd.Series) -> np.ndarray:
    epoch = pd.Timestamp("1970-01-01")
    t = (pd.to_datetime(ds) - epoch).dt.days.values.reshape(-1, 1)
    # Add simple seasonal terms (weekly/yearly) using sin/cos
    week = 2 * np.pi * (t / 7.0)
    year = 2 * np.pi * (t / 365.25)
    features = np.concatenate([
        t,
        np.sin(week), np.cos(week),
        np.sin(year), np.cos(year),
    ], axis=1)
    return features


def forecast_demand(
    df: pd.DataFrame,
    date_col: str,
    component_col: str,
    order_col: str,
    forecast_days: int = 90,
    test_days: int = 30,
) -> List[ForecastResult]:
    results: List[ForecastResult] = []
    for component in sorted(df[component_col].dropna().unique()):
        series = _prepare_demand_series(df, component, date_col, order_col, component_col)
        if series.empty or series["y"].sum() == 0 and series["y"].nunique() <= 1:
            # Not enough signal to model
            empty_hist = series.rename(columns={"ds": "date"})[["date", "y"]]
            results.append(
                ForecastResult(
                    component_name=component,
                    model_name="InsufficientData",
                    history=empty_hist,
                    test_actual_vs_pred=pd.DataFrame(columns=["date", "actual", "predicted"]),
                    future_forecast=pd.DataFrame(columns=["date", "forecast"]),
                )
            )
            continue

        # Train/test split
        series = series.sort_values("ds").reset_index(drop=True)
        if len(series) <= test_days + 7:  # ensure some train data
            test_days = max(1, min(7, len(series) // 4))

        train = series.iloc[:-test_days].copy()
        test = series.iloc[-test_days:].copy()

        # Try Prophet
        model, model_name = _train_prophet(train)
        if model is None:
            # Fallback to simple Linear Regression on time features
            lr_model, model_name = _train_linear_regression(train)
            # Predict on test
            X_test = _time_features(test["ds"]) 
            test_pred = lr_model.predict(X_test)
            test_eval = pd.DataFrame({
                "date": test["ds"].values,
                "actual": test["y"].values,
                "predicted": np.maximum(0.0, test_pred),
            })
            # Forecast future
            future_dates = pd.date_range(series["ds"].max() + pd.Timedelta(days=1), periods=forecast_days, freq="D")
            X_future = _time_features(future_dates.to_series())
            future_pred = lr_model.predict(X_future)
            future_df = pd.DataFrame({"date": future_dates, "forecast": np.maximum(0.0, future_pred)})
        else:
            # Prophet path
            # Predict on test
            test_forecast = model.predict(test.rename(columns={"ds": "ds"})[["ds"]])
            test_eval = pd.DataFrame({
                "date": test["ds"].values,
                "actual": test["y"].values,
                "predicted": np.maximum(0.0, test_forecast["yhat"].values),
            })
            # Forecast future
            future = model.make_future_dataframe(periods=forecast_days, freq="D", include_history=False)
            future_forecast = model.predict(future)
            future_df = pd.DataFrame({
                "date": future_forecast["ds"].values,
                "forecast": np.maximum(0.0, future_forecast["yhat"].values),
            })

        hist_df = series.rename(columns={"ds": "date"})[["date", "y"]]
        results.append(
            ForecastResult(
                component_name=component,
                model_name=model_name,
                history=hist_df,
                test_actual_vs_pred=test_eval,
                future_forecast=future_df,
            )
        )
    return results


def save_forecast_outputs(results: List[ForecastResult], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save per-component evaluation and future forecast
    eval_rows = []
    future_rows = []
    for r in results:
        eval_df = r.test_actual_vs_pred.copy()
        eval_df.insert(0, "component_name", r.component_name)
        eval_df.insert(1, "model", r.model_name)
        eval_rows.append(eval_df)

        fut_df = r.future_forecast.copy()
        fut_df.insert(0, "component_name", r.component_name)
        fut_df.insert(1, "model", r.model_name)
        future_rows.append(fut_df)

    if eval_rows:
        all_eval = pd.concat(eval_rows, ignore_index=True)
        all_eval.to_csv(output_dir / "actual_vs_pred_test.csv", index=False)
    else:
        pd.DataFrame(columns=["component_name", "model", "date", "actual", "predicted"]).to_csv(
            output_dir / "actual_vs_pred_test.csv", index=False
        )

    if future_rows:
        all_future = pd.concat(future_rows, ignore_index=True)
        all_future.to_csv(output_dir / "future_forecast_90d.csv", index=False)
    else:
        pd.DataFrame(columns=["component_name", "model", "date", "forecast"]).to_csv(
            output_dir / "future_forecast_90d.csv", index=False
        )


def plot_forecasts(results: List[ForecastResult], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_title(f"Demand History and Forecast - {r.component_name} ({r.model_name})")
        ax.set_xlabel("Date")
        ax.set_ylabel("Demand (orders)")

        if not r.history.empty:
            ax.plot(r.history["date"], r.history["y"], label="Historical Demand", color="tab:blue")
        if not r.test_actual_vs_pred.empty:
            ax.plot(r.test_actual_vs_pred["date"], r.test_actual_vs_pred["actual"], label="Test Actual", color="tab:green")
            ax.plot(r.test_actual_vs_pred["date"], r.test_actual_vs_pred["predicted"], label="Test Predicted", color="tab:orange")
        if not r.future_forecast.empty:
            ax.plot(r.future_forecast["date"], r.future_forecast["forecast"], label="Future Forecast", color="tab:red", linestyle="--")

        ax.legend()
        fig.tight_layout()
        out_path = output_dir / f"demand_forecast_{safe_filename(r.component_name)}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)


def run_pipeline(
    inventory_timeseries_csv: Path,
    purchase_orders_csv: Path,
    starting_inventory_csv: Path,
    output_dir: Path,
    forecast_days: int = 90,
    test_days: int = 30,
) -> None:
    inv_df, po_df, start_df, schema = load_data(inventory_timeseries_csv, purchase_orders_csv, starting_inventory_csv)
    merged = preprocess(inv_df, po_df, start_df, schema)

    # EDA
    eda_df = compute_eda(merged, component_col=schema["inv_component"], stock_col=schema["inv_stock"], order_col=schema["po_order"])
    save_eda_outputs(eda_df, output_dir)
    plot_stock_vs_orders(merged, output_dir / "stock_vs_orders", date_col=schema["inv_date"], component_col=schema["inv_component"], stock_col=schema["inv_stock"], order_col=schema["po_order"])

    # Forecasting
    results = forecast_demand(merged, date_col=schema["inv_date"], component_col=schema["inv_component"], order_col=schema["po_order"], forecast_days=forecast_days, test_days=test_days)
    save_forecast_outputs(results, output_dir)
    plot_forecasts(results, output_dir / "forecast_plots")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inventory demand EDA and forecasting")
    parser.add_argument("--inventory_timeseries", required=True, help="Path to inventory_timeseries.csv")
    parser.add_argument("--purchase_orders", required=True, help="Path to purchase_orders.csv")
    parser.add_argument("--starting_inventory", required=True, help="Path to starting_inventory.csv")
    parser.add_argument("--output_dir", required=False, default="output", help="Directory to write outputs")
    parser.add_argument("--forecast_days", type=int, default=90, help="Number of days to forecast")
    parser.add_argument("--test_days", type=int, default=30, help="Number of days for test evaluation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(
        inventory_timeseries_csv=Path(args.inventory_timeseries),
        purchase_orders_csv=Path(args.purchase_orders),
        starting_inventory_csv=Path(args.starting_inventory),
        output_dir=Path(args.output_dir),
        forecast_days=args.forecast_days,
        test_days=args.test_days,
    )


if __name__ == "__main__":
    main()
