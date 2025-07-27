from playwright.async_api import Page, BrowserContext
import pathlib

STATE_FILE = pathlib.Path("auth.json")
LOGIN_URL = "https://web.gc.com/login"
DASHBOARD_URL_PATTERN = "**/home"
DEBUG = True


async def ensure_logged_in(context: BrowserContext):
    if STATE_FILE.exists():
        if DEBUG:
            print("✅ Using saved session.")
        return

    page: Page = await context.new_page()
    print("🔓 Launching browser for manual login...")
    await page.goto(LOGIN_URL)

    # Wait for login, then save state
    await page.wait_for_url(DASHBOARD_URL_PATTERN, timeout=180000)
    print("✅ Login detected! Saving session...")
    await context.storage_state(path=str(STATE_FILE))
    await page.close()
