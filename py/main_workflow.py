import os
import re
import sys
import subprocess
import csv

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

def save_team_mapping(file_path, team_id, team_name):
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode="a", newline="", encoding="utf-8") as f:
        fieldnames = ["team_id", "team_name"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({"team_id": team_id, "team_name": team_name})

def clear_output_dir(output_dir):
    if not os.path.isdir(output_dir):
        return
    for name in os.listdir(output_dir):
        path = os.path.join(output_dir, name)
        if os.path.isfile(path) or os.path.islink(path):
            try:
                os.remove(path)
            except Exception as exc:
                print(f"⚠️ Warning: Could not remove file {path}: {exc}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python main_workflow.py [GAMECHANGER_URL] [OPTIONAL: run_token_extractor (true/false)]")
        sys.exit(1)

    url = sys.argv[1]
    team_hash = extract_team_hash(url)

    # Determine whether to run the token extractor (defaults to True if not provided)
    run_token = True
    if len(sys.argv) >= 3:
        token_arg = sys.argv[2].lower()
        if token_arg in ["false", "f", "0", "skip", "no"]:
            run_token = False

    # ---------------------------------------------------------
    # RELIABLE ABSOLUTE DIRECTORY RESOLUTION
    # ---------------------------------------------------------
    script_dir = os.path.dirname(os.path.realpath(__file__))
    
    # If main_workflow is executed from inside a subdirectory like 'py', step up to project root
    if os.path.basename(script_dir) == 'py':
        base_dir = os.path.dirname(script_dir)
    else:
        base_dir = script_dir
        
    config_dir = os.path.join(base_dir, 'config')
    scripts_dir = os.path.join(base_dir, 'py')
    output_dir = os.path.join(base_dir, 'output')
    
    # Ensure config folder is safely created before operations
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    clear_output_dir(output_dir)
    
    mapping_file = os.path.join(config_dir, "team_map.csv")
    mapping_db = load_team_mappings(mapping_file)

    # Single source of truth lookup
    if team_hash not in mapping_db:
        print(f"\n⚠️ Target team ID '{team_hash}' not found in team_map.csv.")
        team_name = input("Please enter the human-readable team name for this ID: ").strip()
        if not team_name:
            print("❌ Team name cannot be blank. Exiting.")
            sys.exit(1)
        save_team_mapping(mapping_file, team_hash, team_name)
        mapping_db[team_hash] = team_name
        print(f"✅ Saved '{team_name}' to team_map.csv.\n")
    else:
        team_name = mapping_db[team_hash]

    print(f"🚀 Starting execution pipeline for team: {team_name} ({team_hash})...")

    # Step 1: Run Token Extractor (Conditional)
    if run_token:
        print("\n🔑 Step 1: Extracting authentication token...")
        subprocess.run(["python", os.path.join(scripts_dir, "get_token.py")], check=True)
    else:
        print("\n⏭️ Step 1: Skipping token extraction (using existing token)...")

    # Step 2: Run Scraper (Updated to 2.1.1)
    print("\n📡 Step 2: Scraping GameChanger API...")
    subprocess.run(["python", os.path.join(scripts_dir, "gc_scraper_2_1_1.py"), url], check=True)

    # Step 3: Run Compiler
    print("\n📊 Step 3: Compiling season statistics...")
    subprocess.run(["python", os.path.join(scripts_dir, "compile_output.py"), team_hash], check=True)

    # Step 4: Generate Spray Charts
    print("\n🥎 Step 4: Generating player spray charts...")
    subprocess.run(["python", os.path.join(scripts_dir, "generate_spray_charts.py"), team_hash], check=True)

    # Step 5: Generate Final PDF Report
    print("\n📄 Step 5: Generating visual PDF reports...")
    subprocess.run(["python", os.path.join(scripts_dir, "generate_report.py"), team_name, team_hash], check=True)

    print(f"\n🎉 Pipeline complete! Visual reports generated for {team_name}.")

if __name__ == "__main__":
    main()