#!/usr/bin/env python3
import asyncio, json, pathlib, re, pytesseract, sys
import pandas as pd
from typing import List
from urllib.parse import urlparse
from playwright.async_api import async_playwright
from PIL import Image

# ----- configure here -----
TEAM_URL   = "https://web.gc.com/teams/QTG5YfweKvGm/2024-fall-carmel-gold-softball-13u/schedule"
EMAIL      = "your_email_here"
PASSWORD   = "your_password_here"
STATE_FILE = pathlib.Path("auth.json")  # Saved login state
OUT_FILE   = pathlib.Path("schedule.json")
DUMP_FILE   = pathlib.Path("dump.txt")
DEBUG      = True if "--debug" in sys.argv else False
LIMIT_OCR_FILES = 5 if "--test" in sys.argv else None
PARSE_EXTRAS = True if "--extras" in sys.argv else False

# ----- constants ----------
SCHEDULE_SELECTOR = "a.ScheduleListByMonth__event"
BOX_SCORE_SELECTOR = "[data-testid='box-score-table']"
LOGIN_URL = "https://web.gc.com/login"
DASHBOARD_URL_PATTERN = "**/home"

# ----- directory setup -----
def extract_team_slug(url: str) -> str:
    return url.strip("/").split("/")[-2]

def ensure_team_folder(slug: str) -> pathlib.Path:
    folder = pathlib.Path(slug)
    folder.mkdir(exist_ok=True)
    return folder

# ----- OCR helpers -----
def load_bad_characters(path: str = "bad_characters.txt") -> List[str]:
    try:
        with open(path, encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print("⚠️ 'bad_characters.txt' not found. Proceeding with no replacements.")
        return []

def correct_ocr_numbers(parts: List[str]) -> List[str]:
    return [p.replace('o', '0').replace('O', '0') if re.fullmatch(r'[oO0-9.]+', p) else p for p in parts]

def sanitize_line(line: str, bad_chars: List[str]) -> str:
    for ch in bad_chars:
        line = line.replace(ch, " ")
    return line

def try_fix_batting_stats(stats: List[str]) -> List[str] | None:
    if len(stats) == 6:
        return stats  # Already good

    if len(stats) != 5:
        return None  # Can't fix anything unless we're off by exactly 1

    for i, val in enumerate(stats):
        # 1. Leading 0 or 00
        if val.startswith("0") and len(val) > 1:
            fixed = stats[:i] + list(val) + stats[i+1:]
            if len(fixed) == 6:
                return fixed

        # 2. Ends in 0 (like 30 → 3 0)
        if len(val) == 2 and val.endswith("0"):
            fixed = stats[:i] + [val[0], val[1]] + stats[i+1:]
            if len(fixed) == 6:
                return fixed

        # 3. Greater than 25 and not ending in 0 (e.g., 92 → 9 2)
        if len(val) == 2 and not val.endswith("0") and val.isdigit() and int(val) > 25:
            fixed = stats[:i] + [val[0], val[1]] + stats[i+1:]
            if len(fixed) == 6:
                return fixed
            
    return None  # Couldn't fix

def try_fix_pitching_stats(parts: List[str]) -> List[str] | None:
    # Step 1: Remove problematic characters
    cleaned = [re.sub(r'[().WLwl]', '', p) for p in parts]
    cleaned = [p for p in cleaned if p]  # Remove empty entries

    # Step 2: Re-run OCR number corrections
    cleaned = correct_ocr_numbers(cleaned)

    # Step 3: If we have 6 items (IP + 5 stats), return
    if len(cleaned) == 6:
        return cleaned

    # Step 4: If we have 5 stats (missing RawIP), try repairing
    if len(cleaned) == 5:
        fixed = try_fix_batting_stats(cleaned)  # Reuse same splitting rules
        if fixed and len(fixed) == 6:
            return fixed

    # Step 5: If we have 6 items but IP is malformed (e.g., 218), try special fix
    if len(cleaned) == 6 and re.fullmatch(r'\d{3}', cleaned[0]):
        raw = cleaned[0]
        fixed_ip = f"{raw[0]}.{raw[1]}"
        cleaned[0] = fixed_ip
        return cleaned
    
    # If we have 6 but malformed IP (e.g., 218)
    if len(cleaned) == 6 and re.fullmatch(r'\d{3}', cleaned[0]):
        cleaned[0] = f"{cleaned[0][0]}.{cleaned[0][1]}"
        return cleaned

    return None

def parse_batting_line(line, game_id, team_side, stat_labels):
    if line.startswith("TEAM"):
        return []

    # Pre-clean line (fix common OCR letter-number swaps)
    def fix_numbers_in_line(text):
        return re.sub(r'(?<=\s)[oO](?=\s|\d)', '0', text)

    line = fix_numbers_in_line(line)

    tokens = line.split()
    if len(tokens) < 2:
        return []

    player = " ".join(tokens[:-6]) if len(tokens) >= 7 else " ".join(tokens[:-5])
    stats = tokens[-6:] if len(tokens) >= 7 else tokens[-5:]

    stats = correct_ocr_numbers(stats)

    if len(stats) != 6:
        fixed = try_fix_batting_stats(stats)
        if fixed:
            stats = fixed
        else:
            return []

    return [{
        "Game": game_id,
        "Player": player.strip(),
        "Stat": label,
        "Value": val,
        "TeamSide": team_side,
        "Section": "Batting"
    } for label, val in zip(stat_labels, stats)]

def parse_pitching_line(line, game_id, team_side, stat_labels):
    if line.startswith("TEAM"):
        return []

    # Tokenize and clean
    parts = re.split(r'\s+', line)
    original_parts = parts.copy()

    # Step 1: Clean garbage chars from potential junk tokens
    cleaned = [re.sub(r'[().WLwl]', '', p) for p in parts]
    cleaned = [p for p in cleaned if p]  # remove empty strings
    cleaned = correct_ocr_numbers(cleaned)

    # Step 2: If we have exactly 6 stats, assume it's [IP, H, R, ER, BB, SO]
    if len(cleaned) >= 6:
        possible_stats = cleaned[-6:]
        possible_name = ' '.join(cleaned[:-6])
    else:
        # Try fallback repair
        fixed = try_fix_pitching_stats(cleaned)
        if fixed and len(fixed) == 6:
            possible_stats = fixed
            possible_name = ' '.join(cleaned[:-6])  # fallback fallback
        else:
            return []

    try:
        raw_ip = possible_stats[0]
        # Fix common edge case like '218' -> '2.1'
        if re.fullmatch(r'\d{3}', raw_ip):
            raw_ip = f"{raw_ip[0]}.{raw_ip[1]}"

        if "." in raw_ip:
            full, outs = raw_ip.split(".")
            if not outs.isdigit() or int(outs) > 2:
                return []  # Invalid IP format
            ip_val = int(full) + (int(outs) / 3)
        else:
            ip_val = float(raw_ip)

        result = [{
            "Game": game_id,
            "Player": possible_name.strip(),
            "Stat": "IP",
            "Value": round(ip_val, 3),
            "TeamSide": team_side,
            "Section": "Pitching"
        }, {
            "Game": game_id,
            "Player": possible_name.strip(),
            "Stat": "RawIP",
            "Value": raw_ip,
            "TeamSide": team_side,
            "Section": "Pitching"
        }]

        for label, val in zip(stat_labels[1:], possible_stats[1:]):
            result.append({
                "Game": game_id,
                "Player": possible_name.strip(),
                "Stat": label,
                "Value": val,
                "TeamSide": team_side,
                "Section": "Pitching"
            })

        return result

    except Exception as e:
        return []

def parse_batting_extras_line(line, game_id, team_side):
    tags = {"2B", "3B", "HR", "SB", "TB"}
    results = []

    for tag in tags:
        if not line.startswith(f"{tag}:"):
            continue

        try:
            remainder = line[len(tag) + 1:].strip()
            tokens = remainder.split()
            i = 0

            while i < len(tokens) - 1:
                name1 = tokens[i]
                name2 = tokens[i + 1]

                if i + 2 < len(tokens) and tokens[i + 2].isdigit():
                    count = int(tokens[i + 2])
                    i += 3
                else:
                    count = 1
                    i += 2

                player = f"{name1} {name2}"
                results.append({
                    "Game": game_id,
                    "Player": player,
                    "Stat": tag,
                    "Value": count,
                    "TeamSide": team_side,
                    "Section": "Batting"
                })

        except Exception as e:
            print(f"[WARN] Failed to parse batting extras line: {line} ({e})")

    return results


def parse_pitching_extras_line(line, game_id, team_side):
    tags = {"WP", "PB"}
    results = []

    for tag in tags:
        if not line.startswith(f"{tag}:"):
            continue

        try:
            remainder = line[len(tag) + 1:].strip()
            tokens = remainder.split()
            i = 0

            while i < len(tokens) - 1:
                name1 = tokens[i]
                name2 = tokens[i + 1]

                if i + 2 < len(tokens) and tokens[i + 2].isdigit():
                    count = int(tokens[i + 2])
                    i += 3
                else:
                    count = 1
                    i += 2

                player = f"{name1} {name2}"
                results.append({
                    "Game": game_id,
                    "Player": player,
                    "Stat": tag,
                    "Value": count,
                    "TeamSide": team_side,
                    "Section": "Pitching"
                })

        except Exception as e:
            print(f"[WARN] Failed to parse pitching extras line: {line} ({e})")

    return results

async def ensure_logged_in(context):
    if STATE_FILE.exists():
        if DEBUG: print("✅ Using saved session.")
        return
    page = await context.new_page()
    print("🔓 Launching browser for manual login...")
    await page.goto(LOGIN_URL)
    await page.wait_for_url(DASHBOARD_URL_PATTERN, timeout=180000)
    print("✅ Login detected! Saving session...")
    await context.storage_state(path=str(STATE_FILE))
    await page.close()

async def collect_schedule(page, team_url: str) -> List[dict]:
    await page.goto(team_url, timeout=60000)
    await page.wait_for_selector(SCHEDULE_SELECTOR, timeout=10000)
    anchors = await page.query_selector_all(SCHEDULE_SELECTOR)
    schedule = []
    for a in anchors:
        href = await a.get_attribute("href")
        date = await a.inner_text()
        if href and "/schedule/" in href:
            game_id = href.split("/schedule/")[-1].split("?")[0]
            schedule.append({
                "date": date.strip().replace("/", "-"),
                "link": href.split("?")[0],
                "game_id": game_id
            })
    if DEBUG:
        print(f"[DEBUG] Found {len(schedule)} schedule entries")
    return schedule

async def screenshot_box_score(page, url: str, save_path: pathlib.Path):
    print(f"[INFO] Navigating to: {url}")
    await page.goto(url, timeout=60000)
    try:
        await page.wait_for_selector(BOX_SCORE_SELECTOR, timeout=15000)
        element = await page.query_selector(BOX_SCORE_SELECTOR)
        box = await element.bounding_box()
        if not box:
            print("⚠️  Could not locate box score.")
            return
        await page.set_viewport_size({
            "width": int(box["width"]) + 40,
            "height": int(box["height"]) + 40
        })
        await element.screenshot(path=save_path)
        print(f"📸 Screenshot saved: {save_path}")
    except Exception as e:
        print(f"❌ Error screenshotting {url}: {e}")

def split_screenshots_vertically(directory: pathlib.Path):
    split_dir = directory / "split"
    split_dir.mkdir(exist_ok=True)

    for file in directory.glob("*.png"):
        image = Image.open(file)
        width, height = image.size
        middle = width // 2

        left = image.crop((0, 0, middle, height))
        right = image.crop((middle, 0, width, height))

        left.save(split_dir / f"{file.stem}_left.png")
        right.save(split_dir / f"{file.stem}_right.png")

    print(f"✅ All screenshots split and saved to: {split_dir}")

def ocr_and_export_to_csv(split_dir: pathlib.Path, output_csv: pathlib.Path, max_files: int = None):
    rows = []
    discarded = []

    extra_stat_tags = ["2B", "3B", "SB", "TB", "HR", "WP", "PB"]
    stat_labels_batting = ["AB", "R", "H", "RBI", "BB", "SO"]
    stat_labels_pitching = ["IP", "H", "R", "ER", "BB", "SO"]

    image_files = sorted(split_dir.glob("*.png"))
    if max_files is not None:
        image_files = image_files[:max_files]

    for img_file in image_files:
        img = Image.open(img_file).convert("L")
        text = pytesseract.image_to_string(img)

        team_side = "left" if "left" in img_file.stem else "right"
        game_id = img_file.stem.replace("_left", "").replace("_right", "")
        section = None

        bad_chars = load_bad_characters()

        for line in text.splitlines():
            line = sanitize_line(line.strip(), bad_chars)
            if not line:
                continue

            if "LINEUP" in line:
                section = "Batting"
                continue
            if "PITCHING" in line:
                section = "Pitching"
                continue

            try:
                parsed = []
                
                if PARSE_EXTRAS:
                    if section == "Batting":
                        parsed += parse_batting_extras_line(line, game_id, team_side)
                    elif section == "Pitching":
                        parsed += parse_pitching_extras_line(line, game_id, team_side)

                # If not extras, try parsing based on current section
                if not parsed and section == "Batting":
                    parsed += parse_batting_line(line, game_id, team_side, stat_labels_batting)
                elif not parsed and section == "Pitching":
                    parsed += parse_pitching_line(line, game_id, team_side, stat_labels_pitching)

                if parsed:
                    rows.extend(parsed)
                else:
                    discarded.append((img_file.name, f"⚠️ Unparsed line: {line}"))

            except Exception as e:
                discarded.append((img_file.name, f"⚠️ Error parsing: {line} ({e})"))

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(output_csv, index=False)
        print(f"✅ Cleaned OCR results written to: {output_csv}")
    else:
        print("⚠️ No valid stat lines parsed.")

    if discarded:
        with open(split_dir / "discarded_lines.txt", "w", encoding="utf-8") as f:
            for filename, bad_line in discarded:
                f.write(f"{filename}: {bad_line}\n")
        print(f"🗑️ Discarded {len(discarded)} lines. See 'discarded_lines.txt'.")

async def scrape_all():
    team_slug = extract_team_slug(TEAM_URL)
    screenshot_dir = ensure_team_folder(team_slug)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False, args=["--start-maximized"])
        context = await browser.new_context(storage_state=str(STATE_FILE) if STATE_FILE.exists() else None, viewport=None)
        await ensure_logged_in(context)
        page = await context.new_page()

        schedule = await collect_schedule(page, TEAM_URL)
        if DEBUG: 
            OUT_FILE.write_text(json.dumps(schedule, indent=2))
            print(f"✅ Saved schedule to '{OUT_FILE.name}'")

        # 🔽 Limit the schedule processed
        games_to_process = schedule[:LIMIT_OCR_FILES] if LIMIT_OCR_FILES else schedule

        for game in games_to_process:
            game_url = "https://web.gc.com" + game["link"]
            filename = f"{game['game_id']}.png"
            save_path = screenshot_dir / filename
            await screenshot_box_score(page, game_url, save_path)

        await browser.close()

    split_screenshots_vertically(screenshot_dir)
    ocr_and_export_to_csv(screenshot_dir / "split", screenshot_dir / "box_scores.csv", max_files=LIMIT_OCR_FILES)
    print (f"{PARSE_EXTRAS} {DEBUG}")


if __name__ == "__main__":
    asyncio.run(scrape_all())
