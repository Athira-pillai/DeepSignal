"""Strict publication-time and ticker-relevance checks for news features."""

from bisect import bisect_left
from datetime import date, datetime, time, timedelta, timezone
from hashlib import sha256
import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from zoneinfo import ZoneInfo


NEWS_ALIGNMENT_VERSION = "strict-published-at-point-in-time-v1"
MARKET_TIMEZONE = ZoneInfo("America/New_York")
TRACKING_PARAMETERS = {"fbclid", "gclid", "mc_cid", "mc_eid"}


def parse_publication_time(value):
    """Return timezone-aware UTC publication time; reject ambiguous/missing values."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        if isinstance(value, (int, float)):
            epoch = float(value)
            if abs(epoch) > 1e12:
                epoch /= 1000.0
            parsed = datetime.fromtimestamp(epoch, tz=timezone.utc)
        elif isinstance(value, datetime):
            parsed = value
        elif isinstance(value, str):
            raw = value.strip()
            if not raw or re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
                return None
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        else:
            return None
    except (ValueError, TypeError, OverflowError, OSError):
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(timezone.utc)


def canonical_article_reference(url, *, title, source, published_at):
    """Return a stable URL reference or a content-derived opaque ID."""
    if url:
        try:
            parts = urlsplit(str(url).strip())
            if parts.scheme.lower() in {"http", "https"} and parts.netloc:
                query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
                         if not k.lower().startswith("utm_") and k.lower() not in TRACKING_PARAMETERS]
                path = parts.path.rstrip("/") or "/"
                return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path,
                                   urlencode(query, doseq=True), ""))
        except ValueError:
            pass
    stable = "|".join((str(source or "").strip().casefold(),
                       str(published_at or "").strip(),
                       " ".join(str(title or "").casefold().split())))
    return "sha256:" + sha256(stable.encode("utf-8")).hexdigest()


def _normalized_term(value):
    return " ".join(str(value or "").casefold().split()).strip()


def _article_relevant(article, symbol, relevance_terms):
    ticker = _normalized_term(re.sub(r"\.(NS|BO)$", "", str(symbol), flags=re.IGNORECASE))
    related = article.get("related_symbols") or article.get("tickers") or []
    if isinstance(related, str):
        related = re.split(r"[,;\s]+", related)
    if any(_normalized_term(value) == ticker for value in related):
        return True
    matched_query = _normalized_term(article.get("matched_query"))
    allowed = {_normalized_term(term) for term in relevance_terms if _normalized_term(term)}
    if matched_query and matched_query in allowed:
        return True
    title = str(article.get("title") or "")
    return bool(ticker and re.search(rf"(?<![A-Za-z0-9]){re.escape(ticker)}(?![A-Za-z0-9])", title, re.IGNORECASE))


def filter_articles_as_of(articles, cutoff, *, symbol, relevance_terms=()):
    """Drop un-timestamped, post-cutoff, irrelevant and duplicate articles."""
    as_of = parse_publication_time(cutoff)
    if as_of is None:
        raise ValueError("cutoff must be a timezone-aware timestamp")
    accepted, seen = [], set()
    counts = {key: 0 for key in (
        "received", "accepted", "missing_or_ambiguous_timestamp", "after_cutoff",
        "irrelevant", "duplicate",
    )}
    for article in articles or []:
        counts["received"] += 1
        if not isinstance(article, dict):
            counts["missing_or_ambiguous_timestamp"] += 1
            continue
        published = parse_publication_time(article.get("published_at") or article.get("publishedAt"))
        if published is None:
            counts["missing_or_ambiguous_timestamp"] += 1
            continue
        if published > as_of:
            counts["after_cutoff"] += 1
            continue
        if not _article_relevant(article, symbol, relevance_terms):
            counts["irrelevant"] += 1
            continue
        reference = canonical_article_reference(
            article.get("url") or article.get("article_url"), title=article.get("title"),
            source=article.get("source"), published_at=published.isoformat(),
        )
        if reference in seen:
            counts["duplicate"] += 1
            continue
        seen.add(reference)
        accepted.append({
            **article,
            "published_at": published.isoformat().replace("+00:00", "Z"),
            "article_id": reference,
            "article_url": article.get("url") or article.get("article_url"),
            "alignment_status": "included_at_or_before_cutoff",
        })
    counts["accepted"] = len(accepted)
    return accepted, {"version": NEWS_ALIGNMENT_VERSION,
                      "cutoff": as_of.isoformat().replace("+00:00", "Z"), **counts}


def session_close_cutoff(session_date):
    """Return the 16:00 America/New_York session cutoff in UTC."""
    if isinstance(session_date, datetime):
        session_date = session_date.date()
    elif not isinstance(session_date, date):
        session_date = date.fromisoformat(str(session_date)[:10])
    return datetime.combine(session_date, time(16, 0), tzinfo=MARKET_TIMEZONE).astimezone(timezone.utc)


def align_articles_to_sessions(sessions, articles, *, symbol, relevance_terms=()):
    """Assign each eligible article once to its first session-close bucket."""
    ordered_sessions = sorted({str(value)[:10] for value in sessions})
    cutoffs = [session_close_cutoff(value) for value in ordered_sessions]
    buckets = {value: [] for value in ordered_sessions}
    counts = {key: 0 for key in (
        "received", "eligible", "missing_or_ambiguous_timestamp", "after_last_cutoff",
        "before_window", "irrelevant", "duplicate",
    )}
    seen = set()
    for article in articles or []:
        counts["received"] += 1
        if not isinstance(article, dict):
            counts["missing_or_ambiguous_timestamp"] += 1
            continue
        published = parse_publication_time(article.get("published_at") or article.get("publishedAt"))
        if published is None:
            counts["missing_or_ambiguous_timestamp"] += 1
            continue
        if not cutoffs or published > cutoffs[-1]:
            counts["after_last_cutoff"] += 1
            continue
        if published < cutoffs[0] - timedelta(days=1):
            counts["before_window"] += 1
            continue
        if not _article_relevant(article, symbol, relevance_terms):
            counts["irrelevant"] += 1
            continue
        reference = canonical_article_reference(
            article.get("url") or article.get("article_url"), title=article.get("title"),
            source=article.get("source"), published_at=published.isoformat(),
        )
        if reference in seen:
            counts["duplicate"] += 1
            continue
        seen.add(reference)
        index = bisect_left(cutoffs, published)
        if index >= len(ordered_sessions):
            counts["after_last_cutoff"] += 1
            continue
        buckets[ordered_sessions[index]].append({
            "article_id": reference,
            "published_at": published.isoformat().replace("+00:00", "Z"),
            "source": article.get("source"),
            "article_url": article.get("url") or article.get("article_url"),
            "title": str(article.get("title") or ""),
        })
        counts["eligible"] += 1
    rows = [{
        "session_date": session,
        "cutoff": cutoff.isoformat().replace("+00:00", "Z"),
        "article_count": len(buckets[session]),
        "articles": buckets[session],
    } for session, cutoff in zip(ordered_sessions, cutoffs)]
    missing = [row["session_date"] for row in rows if row["article_count"] == 0]
    return {
        "status": "ok" if ordered_sessions else "unavailable",
        "symbol": symbol,
        "version": NEWS_ALIGNMENT_VERSION,
        "timezone": "America/New_York",
        "sessions": rows,
        "coverage": {
            "session_count": len(rows),
            "covered_sessions": len(rows) - len(missing),
            "coverage_fraction": ((len(rows) - len(missing)) / len(rows)) if rows else 0.0,
            "missing_dates": missing,
            **counts,
        },
    }
