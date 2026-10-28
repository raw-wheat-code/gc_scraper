import pandas as pd
import matplotlib.pyplot as plt
import os

# Configuration
input_csv = r"E:\GC Scraper\output\compiled_scouting_metrics.csv"
output_dir = r"E:\GC Scraper\charts"

# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def generate_all_spray_charts(csv_file, output_directory):
    # Load data
    df = pd.read_csv(csv_file)
    
    # Define coordinates
    field_map = {
        'Pitcher': (0, 15),
        'Catcher': (0, -2),
        'First Baseman': (15, 15),
        'Second Baseman': (12, 25),
        'Third Baseman': (-15, 15),
        'Shortstop': (-12, 25),
        'Left Fielder': (-25, 45),
        'Center Fielder': (0, 55),
        'Right Fielder': (25, 45)
    }

    # Get list of unique players
    players = df['Player Name'].unique()
    
    for player in players:
        player_data = df[(df['Player Name'] == player) & (df['Category'] == 'Location')].copy()
        
        # Skip if no location data
        if player_data.empty:
            continue
            
        player_data['coords'] = player_data['Metric'].map(field_map)
        player_data = player_data.dropna(subset=['coords'])
        player_data['x'] = player_data['coords'].apply(lambda x: x[0])
        player_data['y'] = player_data['coords'].apply(lambda x: x[1])
        
        # Setup plot
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Draw field
        ax.plot([-30, 0, 30], [50, 0, 50], color='green', linestyle='-', linewidth=2)
        ax.plot([-25, 25], [45, 45], color='green', linestyle='--', linewidth=1)
        
        # Scaling logic for exaggerated circle sizes
        min_val = player_data['Value'].min()
        max_val = player_data['Value'].max()
        
        if max_val == min_val:
            sizes = 2000
        else:
            sizes = 200 + ((player_data['Value'] - min_val) / (max_val - min_val)) * (8000 - 200)
            
        # Plot with red color
        ax.scatter(player_data['x'], player_data['y'], s=sizes, alpha=0.6, c='red', edgecolors='black')
        
        # Labels
        for i, row in player_data.iterrows():
            ax.text(row['x'], row['y'] + 2, row['Metric'], ha='center', fontsize=9)
        
        plt.title(f"Spray Chart: {player}")
        plt.axis('off')
        plt.xlim(-40, 40)
        plt.ylim(-5, 65)
        
        # Save as team_player_name.jpg
        filename = os.path.join(output_directory, f"team_{player.replace(' ', '_')}.jpg")
        plt.savefig(filename, format='jpg', dpi=300)
        plt.close(fig)
        print(f"Generated: {filename}")

# Run
generate_all_spray_charts(input_csv, output_dir)