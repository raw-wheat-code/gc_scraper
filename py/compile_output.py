import pandas as pd
import numpy as np
import os
import glob

# Set the directory where your raw data lives and where you want the compiled files to go
DATA_DIR = r"E:\GC Scraper\output"

# Standard innings per game for your league's ERA calculation (Softball is typically 7)
INNINGS_PER_GAME = 7

# Map raw position values from the boxscore to the nine position columns we track.
POSITION_MAP = {
    'p': 'P',
    'pitcher': 'P',
    'w': 'P',          # Winning Pitcher
    'l': 'P',          # Losing Pitcher
    's': 'P',          # Save
    'sv': 'P',         # Save
    'c': 'C',
    'catcher': 'C',
    '1b': '1B',
    'firstbaseman': '1B',
    'firstbase': '1B',
    '2b': '2B',
    'secondbaseman': '2B',
    'secondbase': '2B',
    '3b': '3B',
    'thirdbaseman': '3B',
    'thirdbase': '3B',
    'ss': 'SS',
    'shortstop': 'SS',
    'lf': 'LF',
    'leftfielder': 'LF',
    'leftfield': 'LF',
    'cf': 'CF',
    'centerfielder': 'CF',
    'centerfield': 'CF',
    'centrefielder': 'CF',
    'centrefield': 'CF',
    'rf': 'RF',
    'rightfielder': 'RF',
    'rightfield': 'RF'
}

POSITION_COLUMNS = ['P', 'C', '1B', '2B', '3B', 'SS', 'LF', 'CF', 'RF']

def normalize_position(pos_value):
    if pd.isna(pos_value):
        return None
    pos = str(pos_value).strip().lower()
    if not pos:
        return None
    pos = pos.replace('.', '').replace(' ', '')

    for sep in ['/', ',', '|', ' or ', ' & ']:
        if sep in pos:
            pos = pos.split(sep)[0]
            break

    return POSITION_MAP.get(pos)


def find_latest_file(keyword):
    """Searches the DATA_DIR for a CSV containing the keyword and returns the most recent one."""
    search_pattern = os.path.join(DATA_DIR, f"*{keyword}*.csv")
    matching_files = glob.glob(search_pattern)
    
    if not matching_files:
        raise FileNotFoundError(f"Could not find any file containing '{keyword}' in {DATA_DIR}")
    
    matching_files.sort(key=os.path.getmtime, reverse=True)
    selected_file = matching_files[0]
    print(f"Found {keyword} file: {os.path.basename(selected_file)}")
    return selected_file

def compile_data():
    print("Starting GC Scraper Compilation...")

    # ==========================================
    # PART 1: BOX SCORE AGGREGATION (HITTING)
    # ==========================================
    try:
        boxscore_file = find_latest_file("boxscore")
        
        df_box = pd.read_csv(boxscore_file)
        df_box = df_box.dropna(subset=['Player']) 
        
        hit_cols = ['AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'BB', 'SO', 'HBP', 'SF']
        
        for col in hit_cols:
            if col in df_box.columns:
                df_box[col] = pd.to_numeric(df_box[col], errors='coerce').fillna(0)
        
        box_agg = df_box.groupby('Player').agg(
            G=('Game', 'nunique'),
            AB=('AB', 'sum'),
            R=('R', 'sum'),
            H=('H', 'sum'),
            Double=('2B', 'sum'),
            Triple=('3B', 'sum'),
            HR=('HR', 'sum'),
            RBI=('RBI', 'sum'),
            BB=('BB', 'sum'),
            SO=('SO', 'sum'),
            HBP=('HBP', 'sum'),
            SF=('SF', 'sum') if 'SF' in df_box.columns else ('AB', lambda x: 0)
        ).reset_index()

        # Calculate Base Stats
        box_agg['1B'] = box_agg['H'] - box_agg['Double'] - box_agg['Triple'] - box_agg['HR']
        box_agg['PA'] = box_agg['AB'] + box_agg['BB'] + box_agg['HBP'] + box_agg['SF']

        # Calculate Rate Stats (using np.where to prevent Division by Zero errors)
        box_agg['AVG'] = np.where(box_agg['AB'] > 0, box_agg['H'] / box_agg['AB'], 0.0)
        box_agg['OBP'] = np.where(box_agg['PA'] > 0, (box_agg['H'] + box_agg['BB'] + box_agg['HBP']) / box_agg['PA'], 0.0)
        total_bases = box_agg['1B'] + (2 * box_agg['Double']) + (3 * box_agg['Triple']) + (4 * box_agg['HR'])
        box_agg['SLG'] = np.where(box_agg['AB'] > 0, total_bases / box_agg['AB'], 0.0)
        box_agg['OPS'] = box_agg['OBP'] + box_agg['SLG']

        # Round the rate stats
        for stat in ['AVG', 'OBP', 'SLG', 'OPS']:
            box_agg[stat] = box_agg[stat].round(3)

        # Set final column order
        final_cols = ['Player', 'G', 'PA', 'AB', 'R', 'H', '1B', 'Double', 'Triple', 'HR', 'RBI', 'BB', 'SO', 'HBP', 'AVG', 'OBP', 'SLG', 'OPS']
        box_agg = box_agg[[c for c in final_cols if c in box_agg.columns]]
        box_agg = box_agg.rename(columns={'Double': '2B', 'Triple': '3B', 'Player': 'Player Name'})
        
        box_out_path = os.path.join(DATA_DIR, 'compiled_hitting_box_scores.csv')
        box_agg.to_csv(box_out_path, index=False)
        print(f"✅ Hitting box scores compiled -> {box_out_path}")

    except Exception as e:
        print(f"❌ Error compiling hitting box scores: {e}")

    # ==========================================
    # PART 2: PITCHING AGGREGATION
    # ==========================================
    try:
        # We use the same boxscore file dynamically found in Part 1
        pitch_cols = ['IP', 'H (P)', 'R (P)', 'ER', 'BB (P)', 'SO (P)', 'W', 'WP', 'HBP (P)', 'TP', 'TS']
        
        # Ensure all columns exist and convert to numeric
        for col in pitch_cols:
            if col in df_box.columns:
                df_box[col] = pd.to_numeric(df_box[col], errors='coerce').fillna(0)
                
        # Group by Player and sum the pitching stats
        pitch_agg = df_box.groupby('Player')[pitch_cols].sum().reset_index()
        
        # Filter out players who didn't pitch (Total Pitches == 0)
        pitch_agg = pitch_agg[pitch_agg['TP'] > 0].copy()

        # Calculate Advanced Pitching Stats
        # ERA = (Earned Runs * Innings Per Game) / Innings Pitched
        pitch_agg['ERA'] = np.where(pitch_agg['IP'] > 0, (pitch_agg['ER'] * INNINGS_PER_GAME) / pitch_agg['IP'], 0.0)
        
        # WHIP = (Walks + Hits) / Innings Pitched
        pitch_agg['WHIP'] = np.where(pitch_agg['IP'] > 0, (pitch_agg['BB (P)'] + pitch_agg['H (P)']) / pitch_agg['IP'], 0.0)
        
        # Strike % = Total Strikes / Total Pitches
        pitch_agg['Strike %'] = np.where(pitch_agg['TP'] > 0, pitch_agg['TS'] / pitch_agg['TP'], 0.0)

        # Round the rate stats
        for stat in ['IP', 'ERA', 'WHIP', 'Strike %']:
            pitch_agg[stat] = pitch_agg[stat].round(3)

        pitch_agg = pitch_agg.rename(columns={'Player': 'Player Name'})
        
        pitch_out_path = os.path.join(DATA_DIR, 'compiled_pitching_stats.csv')
        pitch_agg.to_csv(pitch_out_path, index=False)
        print(f"✅ Pitching stats compiled -> {pitch_out_path}")

    except Exception as e:
        print(f"❌ Error compiling pitching stats: {e}")

    # ==========================================
    # PART 3: POSITIONAL BREAKDOWN PERCENTAGES
    # ==========================================
    try:
        if 'Pos.' not in df_box.columns:
            raise KeyError("Missing 'Pos.' column in boxscore source data.")

        pos_df = df_box.dropna(subset=['Player', 'Pos.']).copy()
        pos_df['Normalized Pos'] = pos_df['Pos.'].apply(normalize_position)
        pos_df = pos_df.dropna(subset=['Normalized Pos']).copy()
        pos_df['Player'] = pos_df['Player'].astype(str)

        position_counts = pos_df.groupby(['Player', 'Normalized Pos']).size().reset_index(name='Count')
        position_pivot = position_counts.pivot(index='Player', columns='Normalized Pos', values='Count').fillna(0).astype(int)

        for pos in POSITION_COLUMNS:
            if pos not in position_pivot.columns:
                position_pivot[pos] = 0

        position_pivot = position_pivot[POSITION_COLUMNS]
        position_totals = position_pivot.sum(axis=0).replace({0: 1})
        position_percent = position_pivot.divide(position_totals, axis=1).fillna(0).round(3)

        compiled_positions = position_percent.reset_index().rename(columns={'Player': 'Player Name'})
        positions_out_path = os.path.join(DATA_DIR, 'compiled_positions.csv')
        compiled_positions.to_csv(positions_out_path, index=False)
        print(f"✅ Positional breakdown compiled -> {positions_out_path}")

    except Exception as e:
        print(f"❌ Error compiling positional breakdowns: {e}")

    # ==========================================
    # PART 4: SCOUTING METRICS PERCENTAGES
    # =========================================
    try:
        scouting_file = find_latest_file("scouting")
        
        df_scout = pd.read_csv(scouting_file)
        df_scout = df_scout.dropna(subset=['Batter'])

        def get_metric_percentages(df, col_name):
            valid_data = df.dropna(subset=[col_name])
            counts = valid_data.groupby(['Batter', col_name]).size().reset_index(name='Count')
            totals = valid_data.groupby('Batter').size().reset_index(name='Total')
            merged = pd.merge(counts, totals, on='Batter')
            merged['Value'] = (merged['Count'] / merged['Total']).round(3)
            merged = merged.rename(columns={'Batter': 'Player Name', col_name: 'Metric'})
            merged['Category'] = col_name 
            return merged[['Player Name', 'Category', 'Metric', 'Value']]

        outcomes_df = get_metric_percentages(df_scout, 'Outcome')
        contact_df = get_metric_percentages(df_scout, 'Contact')
        location_df = get_metric_percentages(df_scout, 'Location')

        compiled_scouting = pd.concat([outcomes_df, contact_df, location_df], ignore_index=True)
        compiled_scouting = compiled_scouting.sort_values(by=['Player Name', 'Category', 'Value'], ascending=[True, True, False])

        scout_out_path = os.path.join(DATA_DIR, 'compiled_scouting_metrics.csv')
        compiled_scouting.to_csv(scout_out_path, index=False)
        print(f"✅ Scouting metrics compiled -> {scout_out_path}")

    except Exception as e:
        print(f"❌ Error compiling scouting metrics: {e}")

if __name__ == "__main__":
    compile_data()