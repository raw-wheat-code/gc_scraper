from typing import List
from playwright.async_api import Page

SCHEDULE_SELECTOR = "a.ScheduleListByMonth__event"


async def collect_schedule(page: Page, team_url: str, debug: bool = False) -> List[dict]:
    await page.goto(team_url, timeout=60000)
    await page.wait_for_selector(SCHEDULE_SELECTOR, timeout=10000)
    anchors = await page.query_selector_all(SCHEDULE_SELECTOR)
    schedule = []
    for a in anchors:
        href = await a.get_attribute("href")
        date = await a.inner_text()
        if href and "/schedule/" in href:
            schedule.append({"date": date.strip(), "link": href.split("?")[0]})
    if debug:
        print(f"[DEBUG] Found {len(schedule)} schedule entries")
    return schedule
