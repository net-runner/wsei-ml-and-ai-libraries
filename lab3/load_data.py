import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# --- League Information ---
def load_league_info():
    df = pd.read_csv('../data/LeagueData.csv')
    df['Release Date'] = pd.to_datetime(df['Release Date'])
    df['End Date'] = pd.to_datetime(df['End Date'])
    df['league'] = df['League'].str.replace(' league', '', regex=False)
    df.set_index('league', inplace=True)
    return df

# --- Currency Information ---
def load_data():
    league_files = {
        "Ancestor": "Ancestor.currency.csv",
        "Crucible": "Crucible.currency.csv",
        "Affliction": "Affliction.currency.csv",
        "Necropolis": "Necropolis.currency.csv",
        "Kalandra": "Kalandra.currency.csv",
        "Sanctum": "Sanctum.currency.csv",
    }
    all_data = []
    files_found = True
    
    for league_name, file_name in league_files.items():
        try:
            df = pd.read_csv(f"../data/{league_name}.currency.csv", sep=None, engine='python', header=0)
            
            if df.empty:
                st.sidebar.warning(f"File '{file_name}' is empty.")
                continue
            
            df.columns = df.columns.str.lower()
            
            if 'league' not in df.columns:
                df['league'] = league_name
            
            all_data.append(df)
            
        except FileNotFoundError:
            print(f"Error: File '{file_name}' not found.")
            files_found = False
        except pd.errors.EmptyDataError:
            print(f"Error: File '{file_name}' is empty or corrupted.")
            files_found = False
        except Exception as e:
            print(f"Error loading {file_name}: {str(e)}")
            files_found = False
    
    if not all_data:
        print("No data could be loaded from any files.")
        return pd.DataFrame()
    
    try:
        combined_df = pd.concat(all_data, ignore_index=True)
    except Exception as e:
        print(f"Error combining dataframes: {str(e)}")
        return pd.DataFrame()
    
    try:
        combined_df['date'] = pd.to_datetime(combined_df['date'], errors='coerce')
        
        combined_df.dropna(subset=['date'], inplace=True)
        
        numeric_columns = ['value']
        for col in numeric_columns:
            if col in combined_df.columns:
                combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
        
        string_columns = ['get', 'pay', 'league', 'confidence']
        for col in string_columns:
            if col in combined_df.columns:
                combined_df[col] = combined_df[col].astype(str)
        
        required_cols = ['date', 'value', 'confidence', 'get', 'pay', 'league']
        existing_required_cols = [col for col in required_cols if col in combined_df.columns]
        
        if existing_required_cols:
            combined_df.dropna(subset=existing_required_cols, inplace=True)
        
        return combined_df
        
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return pd.DataFrame()