import polars as pl
import plotly.express as px
import plotly.graph_objects as go
import numpy as np # numpy might still be useful for certain operations, but try to use polars expressions

# --- Player retention (player counts from steam)
def load_players_data_pl(file_path="../data/players.csv"):
    """
    Loads player data from a CSV file with specific columns using Polars.

    Args:
        file_path (str): The path to the CSV file.
                         Defaults to "../data/players.csv".

    Returns:
        polars.DataFrame: A DataFrame containing the loaded data,
                          or None if an error occurs.
    """
    try:
        # Use pl.read_csv for Polars DataFrames
        # Specify separator directly
        df = pl.read_csv(
            file_path,
            separator=';',
            has_header=True,
            dtypes={'DateTime': pl.Utf8, 'Players': pl.Int64, 'Average Players': pl.Float64} # Define dtypes for efficiency
        )

        # Check if the required columns exist
        required_columns = ["DateTime", "Players", "Average Players"]
        if not all(col in df.columns for col in required_columns):
            missing_columns = [col for col in required_columns if col not in df.columns]
            print(f"Error: Missing required columns in the CSV file: {missing_columns}")
            return None

        # Select only the required columns and cast DateTime
        df = df.select(
            pl.col("DateTime").str.strptime(pl.Datetime, strict=False).alias("DateTime"), # Convert to datetime
            pl.col("Players"),
            pl.col("Average Players")
        )
        return df
    except pl.ComputeError as e: # Catch Polars-specific errors for file operations
        print(f"Error reading file '{file_path}': {e}")
        return None
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

# --- League Information ---
def load_league_info_pl():
    """
    Loads league information from a CSV file using Polars.

    Returns:
        polars.DataFrame: A DataFrame containing the loaded league data.
    """
    try:
        df = pl.read_csv(
            '../data/LeagueData.csv',
            has_header=True,
            # Read as Utf8 first to prevent Polars from trying to infer before we specify the format
            dtypes={'Release Date': pl.Utf8, 'End Date': pl.Utf8} 
        )
        
        date_time_format = "%Y-%m-%d %I:%M:%S %p" 
        
        # Convert date columns to Datetime using the specified format
        df = df.with_columns([
            pl.col('Release Date').str.strptime(pl.Datetime, format=date_time_format, strict=False).alias('Release Date'),
            pl.col('End Date').str.strptime(pl.Datetime, format=date_time_format, strict=False).alias('End Date'),
            pl.col('League').str.replace_all(' league', '').alias('league')
        ])
        
        return df
    except pl.ComputeError as e:
        print(f"Error loading LeagueData.csv: {e}")
        return pl.DataFrame()
    except FileNotFoundError:
        print(f"Error: The file '../data/LeagueData.csv' was not found.")
        return pl.DataFrame()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return pl.DataFrame()

# --- Currency Information ---
def load_data_pl():
    """
    Loads currency data from multiple CSV files using Polars.

    Returns:
        polars.DataFrame: A combined DataFrame containing the loaded currency data.
    """
    league_files = {
        "Necropolis": "Necropolis.currency.csv",
        "Affliction": "Affliction.currency.csv",
        "Ancestor": "Ancestor.currency.csv",
        "Crucible": "Crucible.currency.csv",
        "Sanctum": "Sanctum.currency.csv",
        "Kalandra": "Kalandra.currency.csv",
        # "Expedition": "Expedition.currency.csv",
        # "Scourge": "Scourge.currency.csv",
        # "Sentinel": "Sentinel.currency.csv",
        # "Ultimatum": "Ultimatum.currency.csv",
    }
    all_data = []

    for league_name, file_name in league_files.items():
        full_path = f"../data/{file_name}"
        try:
            df = pl.read_csv(
                full_path,
                separator=';', # <--- ADDED: Specify semicolon separator
                has_header=True,
                infer_schema_length=10000, 
            )

            if df.is_empty():
                print(f"File '{file_name}' is empty.")
                continue

            if 'league' not in df.columns:
                df = df.with_columns(pl.lit(league_name).alias('league'))

            all_data.append(df)

        except pl.ComputeError as e:
            print(f"Error reading file '{full_path}': {e}")
            continue
        except FileNotFoundError:
            print(f"Error: File '{full_path}' not found.")
            continue
        except Exception as e:
            print(f"Error loading {file_name}: {str(e)}")
            continue

    if not all_data:
        print("No data could be loaded from any files.")
        return pl.DataFrame()

    try:
        combined_df = pl.concat(all_data, how="vertical")
    except Exception as e:
        print(f"Error combining dataframes: {str(e)}")
        return pl.DataFrame()

    try:
        # Define the date format for 'date' column based on example: YYYY-MM-DD
        date_format_currency = "%Y-%m-%d"

        # Explicitly cast to Utf8 *before* strptime to ensure consistency
        # Then attempt to parse into pl.Date
        combined_df = combined_df.with_columns([
            pl.col('Date')
              .cast(pl.Utf8) # Ensure it's a string type first
              .str.strptime(pl.Date, format=date_format_currency, strict=False)
              .alias('date_parsed')
        ])
        
        # Check for nulls introduced by failed parsing and drop them
        # (strict=False converts unparseable dates to null)
        initial_rows = combined_df.height
        combined_df = combined_df.drop_nulls(subset=['date_parsed'])
        if combined_df.height < initial_rows:
            print(f"Warning: Dropped {initial_rows - combined_df.height} rows in currency data due to failed date parsing in 'date' column.")
            
        # Rename the parsed date column back to 'date' and drop the temporary 'date_parsed'
        # Note: 'date_string' alias from previous attempt is removed as it's not strictly necessary with chaining
        combined_df = combined_df.with_columns(pl.col('date_parsed').alias('Date')).drop('date_parsed')

        numeric_columns = ['Value']
        for col in numeric_columns:
            if col in combined_df.columns:
                combined_df = combined_df.with_columns(
                    pl.col(col).cast(pl.Float64, strict=False).alias(col)
                )

        string_columns = ['Get', 'Pay', 'League', 'Confidence']
        for col in string_columns:
            if col in combined_df.columns:
                combined_df = combined_df.with_columns(
                    pl.col(col).cast(pl.Utf8).alias(col)
                )
        
        required_cols = ['Date', 'Value', 'Confidence', 'Get', 'Pay', 'League']
        existing_required_cols = [col for col in required_cols if col in combined_df.columns]
        
        if existing_required_cols:
            combined_df = combined_df.drop_nulls(subset=existing_required_cols)
        
        return combined_df
    
    except pl.ComputeError as ce:
        # This will now print the *actual* Polars compute error, which is crucial for debugging.
        print(f"Polars Compute Error during data cleaning in 'load_data_pl': {ce}")
        print("This often indicates a fundamental issue with data types or formats in a column.")
        print("Please check the 'date' column in your currency CSVs for non-standard entries or unexpected characters.")
        return pl.DataFrame()
    except Exception as e:
        # For any other unexpected errors
        print(f"An unexpected error occurred during data cleaning in 'load_data_pl': {type(e).__name__}: {str(e)}")
        return pl.DataFrame()