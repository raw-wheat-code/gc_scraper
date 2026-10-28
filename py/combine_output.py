import os
import glob
import sys
import pandas as pd

def combine_and_map_stats():
    # 2. Determine directories based on your structure
    # script_dir will be: GC Scraper/py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # base_dir moves one level up to the root: GC Scraper
    base_dir = os.path.dirname(script_dir)
    
    # 3. Target files inside the /output directory relative to the root
    output_dir = os.path.join(base_dir, 'output')
    search_path = os.path.join(output_dir, 'season_stats_*.csv')
    csv_files = glob.glob(search_path)
    
    if not csv_files:
        print(f"Error: No matching stats files found in: {output_dir}")
        return
        
    print(f"Found {len(csv_files)} files in output directory to process.")
    
    # 4. Reference team_map.csv in the root directory
    team_map_path = os.path.join(base_dir, 'config/team_map.csv')
    if not os.path.exists(team_map_path):
        print(f"Error: Mapping file not found at {team_map_path}")
        return
        
    try:
        team_map_df = pd.read_csv(team_map_path)
        # Convert mapping columns into a lookup dictionary
        team_lookup = dict(zip(team_map_df['team_id'], team_map_df['team_name']))
    except Exception as e:
        print(f"Failed to read team mapping file: {e}")
        return

    # 5. Process and combine stats files
    df_list = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            df_list.append(df)
        except Exception as e:
            print(f"Error reading file {os.path.basename(file)}: {e}")
            
    combined_df = pd.concat(df_list, ignore_index=True)
    
    # Apply lookup mapping to 'Team' and 'Opp.' columns
    # If an ID isn't found in team_map.csv, fallback to retaining the original ID
    for col in ['Team', 'Opp.']:
        if col in combined_df.columns:
            combined_df[col] = combined_df[col].map(team_lookup).fillna(combined_df[col])
            
    # Check for any remaining raw IDs to alert the user
    unmapped_teams = [t for t in combined_df['Team'].unique() if str(t).startswith('Team_')]
    if unmapped_teams:
        print(f"Warning: {len(unmapped_teams)} Team IDs were not found in team_map.csv and remained unmapped.")

    # 6. Save output to the root directory (GC Scraper/combined_season_stats.csv)
    output_path = os.path.join(base_dir, 'combined_season_stats.csv')
    combined_df.to_csv(output_path, index=False)
    print(f"Successfully compiled data into: {output_path}")

if __name__ == '__main__':
    combine_and_map_stats()