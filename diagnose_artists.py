import sys
import time
import pandas as pd
import requests
from urllib.parse import quote

artists = pd.read_csv("data/artists.csv")

headers = {
    "User-Agent": "ArtStyleClassifier/1.0 (diagnostic script)"
}


def try_known_url(artist_name, known_url):
    if not isinstance(known_url, str) or "/wiki/" not in known_url:
        return "no_wikipedia_url_in_csv", None

    known_title = known_url.split("/wiki/")[-1]

    for attempt in range(2):
        try:
            summary_url = (
                "https://en.wikipedia.org/api/rest_v1/page/summary/"
                + quote(known_title, safe="")
            )
            r = requests.get(summary_url, headers=headers, timeout=10)

            if r.status_code == 429:
                if attempt == 0:
                    time.sleep(1.5)
                    continue
                return "rate_limited_after_retry", None
            if r.status_code == 200:
                summary = r.json()
                if summary.get("type") == "disambiguation":
                    return "known_url_is_disambiguation", None
                has_extract = bool(summary.get("extract"))
                has_image = bool(summary.get("originalimage") or summary.get("thumbnail"))
                return "ok", (has_extract, has_image)
            return f"known_url_http_{r.status_code}", None
        except requests.RequestException as e:
            return f"known_url_request_error:{e.__class__.__name__}", None
        except ValueError:
            return "known_url_bad_json", None


def try_search_fallback(artist_name):
    """Mirrors the old search-based STEP 1-5 fallback in pages/Artists.py"""
    api_url = "https://en.wikipedia.org/w/api.php"
    try:
        search_params = {
            "action": "query", "list": "search", "srsearch": artist_name,
            "format": "json", "srlimit": 5,
        }
        r = requests.get(api_url, params=search_params, headers=headers, timeout=10)
        if r.status_code == 429:
            return "rate_limited", None
        r.raise_for_status()
        results = r.json().get("query", {}).get("search", [])
        if not results:
            return "search_zero_results", None

        wikipedia_title = results[0].get("title")
        if not wikipedia_title:
            return "search_no_title", None

        encoded_title = quote(wikipedia_title.replace(" ", "_"), safe="")
        summary_url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + encoded_title
        r2 = requests.get(summary_url, headers=headers, timeout=10)

        if r2.status_code == 200:
            summary = r2.json()
            has_extract = bool(summary.get("extract"))
            has_image = bool(summary.get("originalimage") or summary.get("thumbnail"))
            return f"ok_via_search (matched: {wikipedia_title!r})", (has_extract, has_image)

        return f"search_summary_http_{r2.status_code} (top hit was {wikipedia_title!r})", None

    except requests.RequestException as e:
        return f"search_request_error:{e.__class__.__name__}", None
    except ValueError:
        return "search_bad_json", None


def main():
    total = len(artists)
    fully_broken = []
    step0_failed_but_fallback_worked = []
    step0_ok = []

    for i, row in artists.iterrows():
        name = row["name"]
        known_url = row.get("wikipedia")

        reason0, data0 = try_known_url(name, known_url)

        if reason0 == "ok":
            step0_ok.append((name, data0))
            print(f"[OK - direct]  {name:25s}  extract={data0[0]}  image={data0[1]}")
            continue

        # STEP 0 failed - try the fallback, same as the app does
        reason1, data1 = try_search_fallback(name)

        if reason1.startswith("ok_via_search"):
            step0_failed_but_fallback_worked.append((name, reason0, reason1))
            print(f"[OK - fallback] {name:25s}  (direct failed: {reason0})  {reason1}")
        else:
            fully_broken.append((name, reason0, reason1))
            print(f"[BROKEN]       {name:25s}  direct={reason0}  fallback={reason1}")

    print("\n" + "=" * 70)
    print(f"Total artists checked: {total}")
    print(f"Worked via direct known-URL lookup: {len(step0_ok)}")
    print(f"Worked via search fallback only:    {len(step0_failed_but_fallback_worked)}")
    print(f"Fully broken (both paths failed):   {len(fully_broken)}")
    print("=" * 70)

    if fully_broken:
        print("\nFULLY BROKEN ARTISTS (these will show 'No information found'):")
        for name, r0, r1 in fully_broken:
            print(f"  - {name}: direct={r0} | fallback={r1}")

    if step0_failed_but_fallback_worked:
        print("\nDirect lookup failed but fallback saved them (slower, but working):")
        for name, r0, r1 in step0_failed_but_fallback_worked:
            print(f"  - {name}: direct={r0}")


if __name__ == "__main__":
    main()
