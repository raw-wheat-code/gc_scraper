import csv
import json
import re
import sys
import time
import requests
import os

# ---------------------------------------------------------
# RELIABLE ABSOLUTE DIRECTORY RESOLUTION
# ---------------------------------------------------------
script_path = os.path.realpath(__file__)
script_dir = os.path.dirname(script_path)

base_dir = os.path.dirname(script_dir)
config_dir = os.path.join(base_dir, 'config')
output_dir = os.path.join(base_dir, 'output')

os.makedirs(output_dir, exist_ok=True)

TOKEN_PATH = os.path.join(config_dir, "token.txt")
MAPPING_FILE = os.path.join(config_dir, "team_map.csv")

with open(TOKEN_PATH, "r", encoding="utf-8") as f:
    GC_TOKEN = f.read().strip()

HEADERS = {
    "gc-app-name": "web",
    "gc-device-id": "db36d9905c9edb74fb84986ea5bbfacd",
    "gc-token": GC_TOKEN,
    "origin": "https://web.gc.com",
    "referer": "https://web.gc.com/",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36"
}

BOXSCORE_COLUMNS = [
    "Game", "Inc.", "Team", "Player", "Pos.", "#", "PositionStart", "Opp.",
    "AB", "R", "H", "RBI", "BB", "SO", "2B", "3B", "HR", "TB", "HBP", "SF", 
    "SB", "CS", "E", "AdjIP", "IP", "H (P)", "R (P)", "ER", "BB (P)", "SO (P)", 
    "W", "WP", "HBP (P)", "TP", "TS"
]

SCOUTING_COLUMNS = ["Game", "Team", "Batter", "Outcome", "Contact", "Location"]

POSITIONS_REGEX = r"(pitcher|catcher|first baseman|second baseman|third baseman|shortstop|left fielder|center fielder|right fielder)"
CONTACT_REGEX = r"(hard ground ball|ground ball|fly ball|line drive|pop fly|bunt)"

def extract_team_hash(url):
    match = re.search(r"/teams/([^/]+)", url)
    if match:
        return match.group(1)
    print("❌ Error: Could not find team hash parameter in the URL.")
    sys.exit(1)

def load_team_mappings(file_path):
    mappings = {}
    if os.path.exists(file_path):
        try:
            with open(file_path, mode="r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row.get("team_id") and row.get("team_name"):
                        mappings[row["team_id"].strip()] = row["team_name"].strip()
        except Exception as e:
            print(f"⚠️ Warning: Failed to parse team maps: {e}")
    return mappings

def ensure_team_in_map(target_id, mapping_file, mapping_db):
    if target_id not in mapping_db:
        print(f"\n⚠️ Target team ID '{target_id}' not found in team_map.csv.")
        new_name = input("Please enter the team name for this ID: ").strip()
        
        if not new_name:
            print("❌ Team name cannot be blank. Exiting.")
            sys.exit(1)
            
        file_exists = os.path.isfile(mapping_file)
        with open(mapping_file, mode="a", newline="", encoding="utf-8") as f:
            fieldnames = ["team_id", "team_name"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow({"team_id": target_id, "team_name": new_name})
            
        mapping_db[target_id] = new_name
        print(f"✅ Saved '{new_name}' to team_map.csv.\n")
        
    return mapping_db

def fetch_schedule_games(team_hash):
    url = f"https://api.team-manager.gc.com/public/teams/{team_hash}/games"
    print(f"📡 Querying team manager timeline service for: {team_hash}...")
    response = requests.get(url, headers={"Accept": "application/json"})
    if response.status_code != 200:
        print(f"❌ Failed to fetch timeline. Status code: {response.status_code}")
        sys.exit(1)
    return response.json()

def fetch_api_data(game_id, endpoint_type, accept_header):
    url = f"https://api.team-manager.gc.com/game-stream-processing/{game_id}/{endpoint_type}"
    local_headers = HEADERS.copy()
    local_headers["accept"] = accept_header
    response = requests.get(url, headers=local_headers)
    return response.json() if response.status_code == 200 else None

def process_boxscore(data, game_id, mapping_db, target_team_id):
    team_ids = list(data.keys())
    if target_team_id not in team_ids:
        return []

    # Identify the opponent ID dynamically
    o_id_list = [tid for tid in team_ids if tid != target_team_id]
    o_id = o_id_list[0] if o_id_list else "Unknown"

    team_names = {
        target_team_id: mapping_db.get(target_team_id, "Unknown Team"),
        o_id: mapping_db.get(o_id, f"Team_{o_id}")
    }

    processed_rows = []
    
    # We only process the target_team_id data, dropping the opponent
    current_team_data = data[target_team_id]
    
    roster = {
        p["id"]: {
            "name": f"{p.get('first_name', '')} {p.get('last_name', '')}".strip(),
            "jersey": p.get("number", "")
        }
        for p in current_team_data.get("players", [])
    }

    player_stats_map = {}

    def create_entry(p_id):
        if p_id not in player_stats_map:
            p_meta = roster.get(p_id, {"name": "Unknown Player", "jersey": ""})
            player_stats_map[p_id] = {col: "" for col in BOXSCORE_COLUMNS}
            player_stats_map[p_id].update({
                "Game": game_id, 
                "Team": team_names[target_team_id], 
                "Opp.": team_names[o_id],
                "Player": p_meta["name"], 
                "#": p_meta["jersey"]
            })
        return player_stats_map[p_id]

    for group in current_team_data.get("groups", []):
        category = group.get("category")
        
        for entry in group.get("stats", []):
            p_row = create_entry(entry.get("player_id"))
            metrics = entry.get("stats", {})
            
            if entry.get("player_text"):
                p_row["Pos."] = entry["player_text"].replace("(", "").replace(")", "")

            if category == "lineup":
                p_row.update({k: metrics.get(k, "") for k in ["AB", "R", "H", "RBI", "BB", "SO"]})
            elif category == "pitching":
                p_row["IP"] = round(metrics.get("IP", 0.0), 2)
                p_row.update({f"{k} (P)" if k in ["H", "BB", "SO"] else k: metrics.get(k, "") for k in ["H", "R", "ER", "BB", "SO"]})

        for extra in group.get("extra", []):
            stat_name = extra.get("stat_name")
            for item in extra.get("stats", []):
                p_row = create_entry(item.get("player_id"))
                val = item.get("value", "")
                if stat_name in ["2B", "3B", "HR", "TB", "SB", "CS", "WP", "SF"]:
                    p_row[stat_name] = val
                elif stat_name == "#P":
                    p_row["TP"] = val
                elif stat_name == "TS":
                    p_row["TS"] = val

    processed_rows.extend(player_stats_map.values())
    return processed_rows

def process_plays(data, game_id, mapping_db, target_team_id):
    if not data or "plays" not in data:
        return []

    player_map = {}
    for team_id, players in data.get("team_players", {}).items():
        team_name = mapping_db.get(team_id, f"Team_{team_id}")
        for p in players:
            player_map[p["id"]] = {
                "name": f"{p.get('first_name', '')} {p.get('last_name', '')}".strip(),
                "team_name": team_name,
                "team_id": team_id
            }

    scouting_rows = []
    uuid_pattern = re.compile(r"\$\{([a-f0-9\-]{36})\}")

    for play in data.get("plays", []):
        final_details = play.get("final_details", [])
        if not final_details:
            continue
            
        found_uuids = uuid_pattern.findall(final_details[0].get("template", ""))
        if not found_uuids:
            continue

        batter_info = player_map.get(found_uuids[0], {"name": "Unknown Player", "team_name": "Unknown Team", "team_id": "Unknown"})
        
        # Discard the play if the batter does not belong to the target team
        if batter_info["team_id"] != target_team_id:
            continue
            
        main_template = final_details[0].get("template", "").lower()
        base_outcome = play.get("name_template", {}).get("template", "Unknown")
        
        if base_outcome == "Strikeout":
            if "looking" in main_template: outcome = "Strikeout Looking"
            elif "swinging" in main_template: outcome = "Strikeout Swinging"
            else: outcome = "Strikeout"
        else:
            outcome = base_outcome

        contact_type = ""
        if contact_match := re.search(CONTACT_REGEX, main_template):
            contact_type = contact_match.group(1).title()
        else:
            if outcome == "Fly Out": contact_type = "Fly Ball"
            elif outcome == "Ground Out": contact_type = "Ground Ball"
            elif outcome == "Line Out": contact_type = "Line Drive"
            elif outcome == "Pop Out": contact_type = "Pop Fly"
            elif outcome == "Bunt Out": contact_type = "Bunt"

        location = ""
        if not any(x in outcome.lower() for x in ["walk", "strikeout", "hit by pitch"]):
            if pos_match := re.search(POSITIONS_REGEX, main_template):
                location = pos_match.group(1).title()

        scouting_rows.append({
            "Game": game_id, "Team": batter_info["team_name"], "Batter": batter_info["name"],
            "Outcome": outcome, "Contact": contact_type, "Location": location
        })

    return scouting_rows

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_gc_scraper.py [SCHEDULE_URL]")
        sys.exit(1)
        
    target_team_hash = extract_team_hash(sys.argv[1])
    mapping_db = load_team_mappings(MAPPING_FILE)
    
    # Check map and prompt if missing
    mapping_db = ensure_team_in_map(target_team_hash, MAPPING_FILE, mapping_db)
    
    games_data = fetch_schedule_games(target_team_hash)
    
    completed_games = [g for g in games_data if g.get("game_status") == "completed"]
    print(f"📊 Identified {len(completed_games)} completed games to harvest tracking metrics from.")
    
    master_boxscore_rows = []
    master_scouting_rows = []
    processed_count = 0
    
    for index, game_item in enumerate(completed_games, start=1):
        game_id = game_item.get("id")
        if not game_id:
            continue
            
        print(f"🔄 [{index}/{len(completed_games)}] Harvesting game: {game_id}...")
        
        # Pull Boxscore API Data Frame
        raw_boxscore = fetch_api_data(game_id, "boxscore", "application/vnd.gc.com.event_box_score+json; version=0.0.0")
        if raw_boxscore:
            master_boxscore_rows.extend(process_boxscore(raw_boxscore, game_id, mapping_db, target_team_hash))
            
        # Pull Plays API Data Frame
        raw_plays = fetch_api_data(game_id, "plays", "application/vnd.gc.com.event_plays+json; version=0.0.0")
        if raw_plays:
            master_scouting_rows.extend(process_plays(raw_plays, game_id, mapping_db, target_team_hash))
            
        processed_count += 1
        time.sleep(0.3)

    if master_boxscore_rows:
        boxscore_file = os.path.join(output_dir, f"season_boxscores_{target_team_hash}.csv")
        with open(boxscore_file, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=BOXSCORE_COLUMNS)
            writer.writeheader()
            writer.writerows(master_boxscore_rows)
        print(f"📊 Boxscores exported: {boxscore_file}")

    if master_scouting_rows:
        scouting_file = os.path.join(output_dir, f"season_scouting_{target_team_hash}.csv")
        with open(scouting_file, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=SCOUTING_COLUMNS)
            writer.writeheader()
            writer.writerows(master_scouting_rows)
        print(f"🥎 Spray tracking metrics exported: {scouting_file}")

    print(f"\n🎉 Finished execution loop. Parsed isolated target team data across {processed_count} games.")

if __name__ == "__main__":
    main()