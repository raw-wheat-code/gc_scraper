import json
import os
import sys
import time
from playwright.sync_api import sync_playwright

# Configuration Paths & Credentials
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
CONFIG_DIR = os.path.join(BASE_DIR, "config")
TOKEN_FILE = os.path.join(CONFIG_DIR, "token.txt")
CREDENTIALS_FILE = os.path.join(CONFIG_DIR, "credentials.json")
TARGET_URL = "https://web.gc.com/login"


def load_credentials():
    email = os.environ.get("GC_EMAIL", "").strip()
    password = os.environ.get("GC_PASSWORD", "").strip()

    if (not email or not password) and os.path.exists(CREDENTIALS_FILE):
        try:
            with open(CREDENTIALS_FILE, "r", encoding="utf-8") as f:
                creds = json.load(f)
                email = email or creds.get("email", "").strip()
                password = password or creds.get("password", "").strip()
        except Exception as e:
            print(f"[❌] Error reading {CREDENTIALS_FILE}: {e}")

    return email, password


def intercept_token():
    if not os.path.exists(CONFIG_DIR):
        os.makedirs(CONFIG_DIR)

    gc_email, gc_password = load_credentials()

    if (
        not gc_email
        or not gc_password
        or gc_email == "your_email@example.com"
        or gc_password == "your_password_here"
    ):
        print(
            f"[❌] Error: Please set GC_EMAIL/GC_PASSWORD or update {CREDENTIALS_FILE} first."
        )
        return

    print("[*] Launching browser window...")
    
    with sync_playwright() as p:
        # Keep headless=False to handle potential AWS WAF challenges visibly
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()

        captured_token = [None]

        # Network interceptor event listener
        def handle_request(request):
            headers = request.headers
            if "gc-token" in headers:
                token = headers["gc-token"]
                if token and token != captured_token[0]:
                    captured_token[0] = token
                    print("[+] Gc-Token successfully intercepted!")

        page.on("request", handle_request)

        print(f"[*] Navigating to {TARGET_URL}")
        page.goto(TARGET_URL, wait_until="domcontentloaded", timeout=60000)
        
        # 1. Wait for the email input field and type the email
        print("[*] Waiting for email form element...")
        page.wait_for_selector("input[type='email']")
        print("[*] Typing email...")
        page.fill("input[type='email']", gc_email)
        time.sleep(0.5) 

        # 2. Handle the "Continue" Button Step (Right after email input)
        print("[*] Scanning for 'Continue' prompt step...")
        continue_btn = page.locator("button:has-text('Continue')")
        try:
            # Wait up to 5 seconds to see if the continue button renders/becomes clickable
            continue_btn.wait_for(state="visible", timeout=5000)
            print("[+] 'Continue' button detected. Clicking to advance...")
            continue_btn.click()
            time.sleep(0.5)  # Brief pause for the password field to transition into view
        except Exception:
            print("[*] No 'Continue' prompt appeared; proceeding straight to password phase...")

        # 3. Wait for the password input field and type the password
        print("[*] Waiting for password form element...")
        page.wait_for_selector("input[type='password']")
        print("[*] Typing password...")
        page.fill("input[type='password']", gc_password)
        time.sleep(0.5)

        # 4. Submit the Login form
        print("[*] Submitting login request...")
        if page.locator("button:has-text('Sign In')").count() > 0:
            page.click("button:has-text('Sign In')")
        elif page.locator("button[type='submit']").count() > 0:
            page.click("button[type='submit']")
        else:
            page.press("input[type='password']", "Enter")

        print("[!] Monitoring authentication flow. (Solve any security puzzles if prompted)...")

        # Wait loop until token is extracted out of background network requests
        while captured_token[0] is None:
            time.sleep(1)
            if browser.connections == []:
                print("[-] Browser window closed prematurely.")
                return

        # Write out the token file to config folder
        try:
            with open(TOKEN_FILE, "w", encoding="utf-8") as f:
                f.write(captured_token[0])
            print(f"[✅] Token cleanly updated at: {TOKEN_FILE}")
        except Exception as e:
            print(f"[❌] Directory write failure: {e}")

        # Clean closure
        context.close()
        browser.close()

if __name__ == "__main__":
    try:
        intercept_token()
    except KeyboardInterrupt:
        print("\n[-] Automation stopped.")
        sys.exit(0)
