# `/data_pipeline` — Zepto Data & AI Platform

Scrapes book catalog data from [books.toscrape.com](https://books.toscrape.com/), a public scraping-practice site, cleans and enriches it with a fixed-rate currency conversion, and loads it into a normalized SQLite database for querying with both SQL and pandas.

## Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

Required packages: `requests`, `beautifulsoup4`, `pandas`, `numpy`. `sqlite3` is part of Python's standard library and needs no separate install.

## How to Run

Run the pipeline script/notebook top to bottom, in order, in a single execution (no partial re-runs — see the Design Decisions note on notebook state below):

```bash
python data_pipeline.ipynb
```

This will:
1. Scrape all books across 4 categories (Travel, Mystery, Historical Fiction, Sequential Art), following pagination on each category page
2. Clean and type-convert the scraped fields
3. Convert GBP prices to INR using the fixed project rate
4. Build a normalized SQLite database (`books_database.db`) with two linked tables
5. Run and print 5 required SQL queries
6. Verify the JOIN query's results against an equivalent `pandas.merge` computation

**Output:** `books_database.db` (the SQLite database file) is created in the working directory, along with printed output for every query and the verification check.

## Data Source

**books.toscrape.com** — a public site built specifically for scraping practice. No login, no API key, no paid tier required.

Final dataset: **144 books across 4 categories**, comfortably exceeding the required minimum of 60 books across at least 3 categories.

## Currency Conversion

**Fixed rate used: 1 GBP = 105.50 INR.**

This is a fixed, project-defined constant for this assignment, not a live or historical market rate — it requires no external API call and no date reference. `price_inr` is computed directly as `price_gbp × 105.50`.

## Design Decisions

### Why SQLite
SQLite is a lightweight, file-based relational database engine that ships built into Python, requires zero server configuration, integrates directly with pandas (`to_sql`/`read_sql`), and produces a single portable database file — ideal for a project of this scale. At higher scale with concurrent multi-user writes, a client-server database (PostgreSQL/MySQL) would be the more appropriate choice, since SQLite is not built for high-concurrency production workloads.

### Handling Encoding Issues
The site's raw HTML responses were being decoded with an incorrect default encoding, producing garbled characters in text fields (e.g., `Â£` instead of `£`, `Noahâs` instead of `Noah's`). This was fixed by explicitly setting `response.encoding = "utf-8"` immediately after each `requests.get()` call, before parsing with BeautifulSoup.

### Drop vs. Impute Strategy
Two categories of fields required different handling when parsing failed for a row:

- **`title` and `category` (dropped if missing/invalid):** these are identity fields with no meaningful "typical value" to substitute — there is no sensible average or central-tendency estimate for a missing book title, so a row missing either field is dropped entirely rather than guessed at.
- **`price_gbp` and `rating` (median-imputed if missing):** these are measurement fields with a genuine central tendency, so a missing value can be reasonably estimated from the rest of the dataset using the median.

**Why median, not mean, for both fields:**
- For `price_gbp`, median is more robust to outliers (a handful of very expensive books won't skew the imputed value as much as they would with a mean).
- For `rating`, an ordinal 1–5 scale, taking the mean can produce a non-discrete value (e.g., 3.4), which isn't a valid rating. Since 1–5 ratings are ordered, the median of a discrete dataset returns an exact integer, preserving a valid value on the original scale.

### Normalized Two-Table Schema
Rather than storing category as a plain text column directly inside the `books` table, the schema separates `categories` (with `category_id`, `category_name`) and `books` (referencing `category_id` as a foreign key). This avoids the update anomaly that comes with denormalized data: if a category name needed to change (e.g., renaming "Mystery" to "Mystery Fiction"), a denormalized design would require updating every single book row carrying that category text, whereas the normalized design requires updating exactly one row in the `categories` table.

### Fixed Rate vs. Live Currency API
A fixed exchange rate was used instead of a live currency-conversion API for several reasons:
- **Reproducibility:** the pipeline produces identical output on every run, which simplifies debugging and grading.
- **Reliability:** third-party HTTP APIs introduce failure modes outside this project's control — downtime, rate limits, network timeouts, or breaking API changes — none of which should block a data pipeline's core, required output.
- **Scope fit:** this project's brief explicitly defines the fixed rate as the graded baseline; a live rate would be appropriate in a context where currency accuracy at the moment of query is itself a business requirement (e.g., a real financial trading system), which isn't the case here.

### Notebook/Script Execution Order
This pipeline should always be run as a single top-to-bottom execution. Because `raw_df` and `df_clean` are built sequentially from earlier cells/steps, partial re-runs (e.g., re-running only the cleaning step after changing the scraping step) can leave stale data in memory, producing inconsistent output between stages. A full restart-and-rerun was used to verify the final submitted output is internally consistent.

## Database Schema

```sql
CREATE TABLE categories(
    category_id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT NOT NULL
);

CREATE TABLE books(
    book_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    price_gbp REAL NOT NULL,
    price_inr REAL NOT NULL,
    rating INTEGER NOT NULL,
    in_stock INTEGER NOT NULL,
    category_id INTEGER NOT NULL,
    FOREIGN KEY (category_id) REFERENCES categories (category_id)
);
```

## SQL Queries Included

| # | Query | Clauses demonstrated |
|---|-------|----------------------|
| 1 | High-rated books | `SELECT`, `WHERE`, `LIMIT` |
| 2 | Most expensive books in INR | `ORDER BY`, `LIMIT` |
| 3 | Distinct ratings available | `DISTINCT` |
| 4 | Books in a price range with specific ratings | `BETWEEN`, `IN` |
| 5 | Top 5-star books with category names | `JOIN` |

Query 5's result is independently verified by reproducing the same JOIN using `pandas.merge()` on in-memory DataFrames (no SQL) — both approaches were confirmed to produce identical output.

## Git Workflow

Work on this module was developed on a feature branch (`feature/data-pipeline-scraper`), committed to across multiple commits, and merged back into `main` — satisfying the project's required Git branching workflow.