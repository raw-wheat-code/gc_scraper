# gc_scraper.py

import asyncio
from modules.login_manager import LoginManager
from modules.schedule_scraper import ScheduleScraper

# ----- CONFIGURATION -----
TEAM_URL = "https://web.gc.com/teams/QTG5YfweKvGm/2024-fall-carmel-gold-softball-13u/schedule"
STATE_FILE = "auth.json"
OUTPUT_FILE = "schedule.json"


async def main():
    # Initialize login manager and authenticate if needed
    login = LoginManager(state_file=STATE_FILE)
    context = await login.get_context()

    # Run the schedule scraper
    scraper = ScheduleScraper(team_url=TEAM_URL, context=context)
    schedule = await scraper.scrape_schedule()

    # Save results
    scraper.save_schedule(schedule, OUTPUT_FILE)

    await context.close()


if __name__ == "__main__":
    asyncio.run(main())
