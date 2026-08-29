from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "model_data.csv"
)


# These are the ONLY columns required by forecaster.py
REQUIRED_COLUMNS = [
    "Store",
    "Date",
    "Sales",
    "Open",
    "Promo",
    "SchoolHoliday",
    "StateHoliday",
]


def load_store_history(store_id: int) -> pd.DataFrame:
    """
    Loads historical data for only one requested store.

    The full model_data.csv is never kept in RAM.
    It is read in chunks and only matching rows are retained.
    """

    matching_chunks = []

    for chunk in pd.read_csv(
        MODEL_DATA_PATH,
        usecols=REQUIRED_COLUMNS,
        chunksize=100_000
    ):

        store_rows = chunk[
            chunk["Store"] == store_id
        ]

        if not store_rows.empty:
            matching_chunks.append(store_rows)

    if not matching_chunks:
        return pd.DataFrame(
            columns=REQUIRED_COLUMNS
        )

    return pd.concat(
        matching_chunks,
        ignore_index=True
    )


def get_dataset_metadata():
    """
    Reads only lightweight metadata from model_data.csv.

    The full dataframe is never retained in memory.
    """

    min_date = None
    max_date = None
    stores = set()

    # For global median calculation
    sales_values = []

    for chunk in pd.read_csv(
        MODEL_DATA_PATH,
        usecols=[
            "Store",
            "Date",
            "Open",
            "Sales"
        ],
        chunksize=100_000
    ):

        chunk["Date"] = pd.to_datetime(
            chunk["Date"]
        )

        chunk_min = chunk["Date"].min()
        chunk_max = chunk["Date"].max()

        if (
            min_date is None
            or chunk_min < min_date
        ):
            min_date = chunk_min

        if (
            max_date is None
            or chunk_max > max_date
        ):
            max_date = chunk_max

        stores.update(
            chunk["Store"].unique()
        )

        open_sales = chunk.loc[
            chunk["Open"] == 1,
            "Sales"
        ].dropna()

        if not open_sales.empty:
            sales_values.append(
                open_sales
            )

    if sales_values:

        all_open_sales = pd.concat(
            sales_values,
            ignore_index=True
        )

        global_median_sales = float(
            all_open_sales.median()
        )

    else:

        global_median_sales = 5430.0

    return {
        "date_min": min_date,
        "date_max": max_date,
        "total_stores": len(stores),
        "global_median_sales": global_median_sales
    }