import os
import sys

def main():
    # Define absolute path to your config file
    config_dir = r"E:\GC Scraper\config"
    token_file = os.path.join(config_dir, "token.txt")
    
    # Check if a token was provided as an argument
    if len(sys.argv) < 2:
        print("❌ Error: No token provided.")
        print('Usage: python update_token.py "YOUR_NEW_TOKEN_HERE"')
        sys.exit(1)
        
    # Extract token and strip out any accidental wrapper quotes or spaces
    new_token = sys.argv[1].strip().strip('"').strip("'")
    
    # Optional: Strip out the word 'Bearer ' if you accidentally copied it with the prefix
    if new_token.startswith("Bearer "):
        new_token = new_token.replace("Bearer ", "", 1)
        
    try:
        # Ensure the config folder exists
        os.makedirs(config_dir, exist_ok=True)
        
        # Overwrite token.txt with the fresh token
        with open(token_file, "w", encoding="utf-8") as f:
            f.write(new_token)
            
        print("✅ Success! Gc-Token cleanly updated at:")
        print(f"   {token_file}")
        
    except Exception as e:
        print(f"❌ Execution error writing token: {e}")

if __name__ == "__main__":
    main()