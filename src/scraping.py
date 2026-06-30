from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

DEFAULT_EVENTS_URL = "http://ufcstats.com/statistics/events/completed?page=all"
DEFAULT_COLUMNS = [
    "Event",
    "Date",
    "Location",
    "WL",
    "Fighter_A",
    "Fighter_B",
    "Fighter_A_KD",
    "Fighter_B_KD",
    "Fighter_A_STR",
    "Fighter_B_STR",
    "Fighter_A_TD",
    "Fighter_B_TD",
    "Fighter_A_SUB",
    "Fighter_B_SUB",
    "Victory_Result",
    "Victory_Method",
    "Round",
    "Time",
    "Weight_Class",
    "Title",
    "Fight_Bonus",
    "Perf_Bonus",
    "Sub_Bonus",
    "KO_Bonus",
]


@dataclass
class ScrapeConfig:
    events_url: str = DEFAULT_EVENTS_URL
    start_year: int = 2013
    end_year: int = datetime.now().year
    timeout_seconds: int = 30
    retries: int = 4
    delay_seconds: float = 0.4
    backoff_factor: float = 1.5
    include_incomplete: bool = False
    output_format: str = "csv"
    output_path: Path | None = None

    def resolved_output_path(self) -> Path:
        if self.output_path is not None:
            return self.output_path
        project_root = Path(__file__).resolve().parents[1]
        extension = ".parquet" if self.output_format == "parquet" else ".csv"
        return project_root / "data" / "raw" / f"ufc_event_data{extension}"


def clean_text(value: str) -> str:
    return " ".join((value or "").replace("\xa0", " ").split()).strip()


def parse_event_year(date_text: str) -> int | None:
    normalized = clean_text(date_text)
    if not normalized:
        return None
    try:
        return datetime.strptime(normalized, "%B %d, %Y").year
    except ValueError:
        match = re.search(r"(19|20)\d{2}", normalized)
        return int(match.group(0)) if match else None


def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            )
        }
    )
    return session


CHALLENGE_MARKER = "Checking your browser"


def looks_like_challenge(html: str) -> bool:
    return CHALLENGE_MARKER in html and "/__c" in html


def solve_challenge(
    session: requests.Session,
    url: str,
    html: str,
    timeout_seconds: int,
    logger: logging.Logger,
) -> None:
    """Solve the ufcstats.com JS proof-of-work interstitial and post the answer.

    The page mines a nonce ``n`` such that ``sha256(f"{nonce}:{n}")`` starts with
    a number of hex zeros, then POSTs it to ``/__c`` to obtain a clearance cookie
    on the session. We replicate that here so a subsequent request returns the
    real content.
    """
    nonce_match = re.search(r'nonce="([0-9a-fA-F]+)"', html)
    target_match = re.search(r"target=new Array\((\d+)\+1\)", html)
    if not nonce_match or not target_match:
        raise RuntimeError("unrecognized browser-check challenge format")

    nonce = nonce_match.group(1)
    zeros = int(target_match.group(1))
    prefix = "0" * zeros

    answer = 0
    while hashlib.sha256(f"{nonce}:{answer}".encode()).hexdigest()[:zeros] != prefix:
        answer += 1

    logger.info("solved browser-check challenge (zeros=%s, n=%s)", zeros, answer)
    base = re.match(r"(https?://[^/]+)", url).group(1)
    session.post(
        base + "/__c",
        data={"nonce": nonce, "n": answer},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=timeout_seconds,
    )


def fetch_page(
    session: requests.Session,
    url: str,
    timeout_seconds: int,
    retries: int,
    backoff_factor: float,
    logger: logging.Logger,
) -> str:
    transient_statuses = {429, 500, 502, 503, 504}
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            response = session.get(url, timeout=timeout_seconds)
            if response.status_code == 200:
                if looks_like_challenge(response.text):
                    solve_challenge(session, url, response.text, timeout_seconds, logger)
                    continue
                return response.text
            if response.status_code in transient_statuses:
                sleep_seconds = backoff_factor**attempt + random.uniform(0.0, 0.3)
                logger.warning(
                    "transient status %s for %s on attempt %s/%s; sleeping %.2fs",
                    response.status_code,
                    url,
                    attempt,
                    retries,
                    sleep_seconds,
                )
                time.sleep(sleep_seconds)
                continue
            response.raise_for_status()
        except requests.RequestException as exc:
            last_error = exc
            sleep_seconds = backoff_factor**attempt + random.uniform(0.0, 0.3)
            logger.warning(
                "request error for %s on attempt %s/%s: %s; sleeping %.2fs",
                url,
                attempt,
                retries,
                exc,
                sleep_seconds,
            )
            time.sleep(sleep_seconds)

    raise RuntimeError(f"failed to fetch {url}") from last_error


def fetch_soup(
    session: requests.Session,
    url: str,
    timeout_seconds: int,
    retries: int,
    backoff_factor: float,
    logger: logging.Logger,
) -> BeautifulSoup:
    html = fetch_page(
        session=session,
        url=url,
        timeout_seconds=timeout_seconds,
        retries=retries,
        backoff_factor=backoff_factor,
        logger=logger,
    )
    return BeautifulSoup(html, "html.parser")


def collect_event_links(events_soup: BeautifulSoup, start_year: int, end_year: int) -> list[str]:
    urls: list[str] = []
    for row in events_soup.select("tr.b-statistics__table-row"):
        link = row.select_one("a[href*='event-details']")
        if link is None:
            continue

        cells = row.find_all("td")
        date_text = clean_text(cells[1].get_text(" ", strip=True)) if len(cells) > 1 else ""
        year = parse_event_year(date_text)
        if year is not None and not (start_year <= year <= end_year):
            continue

        href = clean_text(link.get("href") or "")
        if href:
            urls.append(href)

    return urls


def extract_pair(cell: Any, default: str = "") -> tuple[str, str]:
    values = [
        clean_text(p.get_text(" ", strip=True))
        for p in cell.find_all("p")
        if clean_text(p.get_text(" ", strip=True))
    ]
    if not values:
        text = clean_text(cell.get_text(" ", strip=True))
        return (text, default) if text else (default, default)
    if len(values) == 1:
        return values[0], default
    return values[0], values[1]


def first_p_text(cell: Any) -> str:
    paragraph = cell.find("p")
    if paragraph is not None:
        return clean_text(paragraph.get_text(" ", strip=True))
    return clean_text(cell.get_text(" ", strip=True))


def bonus_flags(weight_cell: Any) -> dict[str, int]:
    flags = {
        "Title": 0,
        "Fight_Bonus": 0,
        "Perf_Bonus": 0,
        "Sub_Bonus": 0,
        "KO_Bonus": 0,
    }
    for img in weight_cell.find_all("img"):
        src = img.get("src") or ""
        filename = Path(urlparse(src).path).name.lower()
        if "belt" in filename:
            flags["Title"] = 1
        elif "fight" in filename:
            flags["Fight_Bonus"] = 1
        elif "perf" in filename:
            flags["Perf_Bonus"] = 1
        elif "sub" in filename:
            flags["Sub_Bonus"] = 1
        elif "ko" in filename:
            flags["KO_Bonus"] = 1
    return flags


def parse_event_metadata(event_soup: BeautifulSoup) -> dict[str, str]:
    title_tag = event_soup.find("span", class_="b-content__title-highlight")
    event_title = clean_text(title_tag.get_text(" ", strip=True)) if title_tag else ""

    event_info: dict[str, str] = {}
    for info_item in event_soup.find_all("li", class_="b-list__box-list-item"):
        label = info_item.find("i")
        if label is None:
            continue
        label_text = clean_text(label.get_text(" ", strip=True)).rstrip(":")
        full_text = clean_text(info_item.get_text(" ", strip=True))
        value = clean_text(full_text.replace(clean_text(label.get_text(" ", strip=True)), "", 1))
        event_info[label_text] = value

    return {
        "Event": event_title,
        "Date": event_info.get("Date", ""),
        "Location": event_info.get("Location", ""),
    }


def parse_fight_table(event_soup: BeautifulSoup, include_incomplete: bool = False) -> list[dict[str, Any]]:
    metadata = parse_event_metadata(event_soup)
    if not metadata.get("Event"):
        return []

    selectors = (
        "tr.b-fight-details__table-row.b-fight-details__table-row__hover.js-fight-details-click,"
        "tr.b-fight-details__table-row.js-fight-details-click"
    )

    records: list[dict[str, Any]] = []
    for row in event_soup.select(selectors):
        cells = row.find_all("td")
        if len(cells) < 10:
            continue

        wl = clean_text(cells[0].get_text(" ", strip=True)).upper()
        fighter_a, fighter_b = extract_pair(cells[1])
        fighter_a_kd, fighter_b_kd = extract_pair(cells[2])
        fighter_a_str, fighter_b_str = extract_pair(cells[3])
        fighter_a_td, fighter_b_td = extract_pair(cells[4])
        fighter_a_sub, fighter_b_sub = extract_pair(cells[5])
        victory_result, victory_method = extract_pair(cells[7])
        round_text = first_p_text(cells[8])
        time_text = first_p_text(cells[9])

        if not include_incomplete:
            has_outcome = any((wl, victory_result, victory_method, round_text, time_text))
            if not has_outcome:
                continue
            if "VIEW MATCHUP" in fighter_a_td.upper():
                continue

        records.append(
            {
                "Event": metadata["Event"],
                "Date": metadata["Date"],
                "Location": metadata["Location"],
                "WL": wl,
                "Fighter_A": fighter_a,
                "Fighter_B": fighter_b,
                "Fighter_A_KD": fighter_a_kd,
                "Fighter_B_KD": fighter_b_kd,
                "Fighter_A_STR": fighter_a_str,
                "Fighter_B_STR": fighter_b_str,
                "Fighter_A_TD": fighter_a_td,
                "Fighter_B_TD": fighter_b_td,
                "Fighter_A_SUB": fighter_a_sub,
                "Fighter_B_SUB": fighter_b_sub,
                "Victory_Result": victory_result,
                "Victory_Method": victory_method,
                "Round": round_text,
                "Time": time_text,
                "Weight_Class": first_p_text(cells[6]),
                **bonus_flags(cells[6]),
            }
        )

    return records


def clean_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()

    for row in records:
        key = (
            clean_text(str(row.get("Event", ""))),
            clean_text(str(row.get("Date", ""))),
            clean_text(str(row.get("Fighter_A", ""))),
            clean_text(str(row.get("Fighter_B", ""))),
            clean_text(str(row.get("Weight_Class", ""))),
            clean_text(str(row.get("Round", ""))),
            clean_text(str(row.get("Time", ""))),
        )
        if key in seen:
            continue

        cleaned_row: dict[str, Any] = {}
        for column in DEFAULT_COLUMNS:
            value = row.get(column, "")
            if isinstance(value, str):
                cleaned_row[column] = clean_text(value)
            elif value is None:
                cleaned_row[column] = ""
            else:
                cleaned_row[column] = value

        deduped.append(cleaned_row)
        seen.add(key)

    return deduped


def save_raw(records: list[dict[str, Any]], output_path: Path, output_format: str = "csv") -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_format == "parquet":
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("parquet output requires pandas and pyarrow") from exc

        dataframe = pd.DataFrame(records, columns=DEFAULT_COLUMNS)
        dataframe.to_parquet(output_path, index=False)
        return

    with output_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=DEFAULT_COLUMNS)
        writer.writeheader()
        writer.writerows(records)


def run_scrape(config: ScrapeConfig, logger: logging.Logger) -> tuple[Path, int]:
    session = build_session()
    logger.info("loading event index: %s", config.events_url)

    events_soup = fetch_soup(
        session=session,
        url=config.events_url,
        timeout_seconds=config.timeout_seconds,
        retries=config.retries,
        backoff_factor=config.backoff_factor,
        logger=logger,
    )
    event_urls = collect_event_links(events_soup, config.start_year, config.end_year)
    logger.info("found %s event pages in year range %s-%s", len(event_urls), config.start_year, config.end_year)

    records: list[dict[str, Any]] = []
    for index, event_url in enumerate(event_urls, start=1):
        try:
            event_soup = fetch_soup(
                session=session,
                url=event_url,
                timeout_seconds=config.timeout_seconds,
                retries=config.retries,
                backoff_factor=config.backoff_factor,
                logger=logger,
            )
            metadata = parse_event_metadata(event_soup)
            event_year = parse_event_year(metadata.get("Date", ""))
            if event_year is not None and not (config.start_year <= event_year <= config.end_year):
                continue

            records.extend(parse_fight_table(event_soup, include_incomplete=config.include_incomplete))
        except Exception as exc:  # noqa: BLE001
            logger.warning("skipping event %s due to error: %s", event_url, exc)
            continue

        if config.delay_seconds > 0:
            time.sleep(config.delay_seconds)

        if index % 25 == 0:
            logger.info("processed %s/%s events | rows=%s", index, len(event_urls), len(records))

    cleaned_records = clean_records(records)
    output_path = config.resolved_output_path()
    save_raw(cleaned_records, output_path, output_format=config.output_format)

    logger.info("saved %s records to %s", len(cleaned_records), output_path)
    return output_path, len(cleaned_records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="scrape UFC fight-level results from ufcstats.com")
    parser.add_argument("--start-year", type=int, default=2013, help="first year to include")
    parser.add_argument("--end-year", type=int, default=datetime.now().year, help="last year to include")
    parser.add_argument("--output-format", choices=["csv", "parquet"], default="csv")
    parser.add_argument("--output-path", type=Path, default=None, help="optional custom output path")
    parser.add_argument("--include-incomplete", action="store_true", help="include rows with missing outcomes")
    parser.add_argument("--delay-seconds", type=float, default=0.4, help="sleep between event requests")
    parser.add_argument("--retries", type=int, default=4, help="max request attempts per page")
    parser.add_argument("--timeout-seconds", type=int, default=30, help="http timeout in seconds")
    parser.add_argument("--backoff-factor", type=float, default=1.5, help="exponential backoff base")
    parser.add_argument("--verbose", action="store_true", help="enable debug logs")
    return parser.parse_args()


def configure_logging(verbose: bool) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("ufc.scraping")


def main() -> int:
    args = parse_args()
    logger = configure_logging(verbose=args.verbose)

    config = ScrapeConfig(
        start_year=args.start_year,
        end_year=args.end_year,
        retries=args.retries,
        timeout_seconds=args.timeout_seconds,
        delay_seconds=args.delay_seconds,
        backoff_factor=args.backoff_factor,
        include_incomplete=args.include_incomplete,
        output_format=args.output_format,
        output_path=args.output_path,
    )

    try:
        output_path, row_count = run_scrape(config=config, logger=logger)
    except Exception as exc:  # noqa: BLE001
        logger.exception("scrape failed: %s", exc)
        return 1

    logger.info("done | rows=%s | output=%s", row_count, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
