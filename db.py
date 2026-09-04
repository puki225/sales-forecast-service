"""
Postgres access for the forecasting job. Talks to the same database as
react-finance-dashboard/server - reuses its vat_divisor()/v_sku_revenue view so the
"actual" revenue this job trains on is identical, penny for penny, to what the
Sales Forecast tab's history line already shows. Never redefines that logic here.
"""
import os
import psycopg2
import psycopg2.extras
import pandas as pd


def get_connection():
    dsn = os.environ["DATABASE_URL"]
    return psycopg2.connect(dsn)


def fetch_daily_revenue(conn, min_date):
    """Per-SKU, per-day net revenue (GBP, VAT-exclusive), refunds subtracted - same basis
    as the dashboard's /api/sales-forecast history line (and Sales Summary/Product
    Breakdown's "net revenue" everywhere else in the app). v_sku_revenue.net_revenue on
    its own is pre-refund and would train this model on a number that doesn't match what
    the tab shows it against. min_date bounds how far back to pull (e.g. 2 years) -
    required, not optional, since an unbounded pull only grows more expensive over time."""
    sql = """
        WITH rev AS (
          SELECT sku, order_date::date AS date,
            SUM(net_revenue / vat_divisor(shipping_country))::numeric(12,2) AS revenue,
            SUM(quantity)::int AS units
          FROM v_sku_revenue
          WHERE sku IS NOT NULL AND order_date::date >= %(min_date)s
          GROUP BY sku, order_date::date
        ),
        ref AS (
          SELECT sku, refund_date::date AS date,
            SUM(amount_refunded / vat_divisor(shipping_country))::numeric(12,2) AS refunded
          FROM v_refunds_by_date
          WHERE sku IS NOT NULL AND refund_date::date >= %(min_date)s
          GROUP BY sku, refund_date::date
        )
        SELECT COALESCE(rev.sku, ref.sku) AS sku, COALESCE(rev.date, ref.date) AS date,
          (COALESCE(rev.revenue, 0) - COALESCE(ref.refunded, 0))::numeric(12,2) AS revenue,
          COALESCE(rev.units, 0) AS units
        FROM rev FULL OUTER JOIN ref ON ref.sku = rev.sku AND ref.date = rev.date
        ORDER BY 1, 2
    """
    return pd.read_sql(sql, conn, params={"min_date": min_date})


def fetch_sku_config(conn):
    """Manual overrides from the Sales Forecast tab - always take precedence over
    auto-classification/auto-fitting for the SKU they're set on."""
    sql = "SELECT sku, stage_override, is_end_of_life FROM sku_forecast_config"
    return pd.read_sql(sql, conn)


def fetch_eol_inputs(conn):
    """Everything the end-of-life depletion model needs, in one query: current FBA
    sellable stock, and the same seasonal-velocity inputs /api/inventory uses for its
    "days of inventory left" figure - so a SKU's forecast tapering to zero lines up with
    what the Inventory tab already says about it, instead of computing a second,
    disagreeing answer. Amazon/FBA-only, matching the Inventory tab's own scope."""
    sql = """
        WITH latest_inv AS (
          SELECT DISTINCT ON (sku) sku, fulfillable_quantity::int AS sellable
          FROM amazon_inventory_snapshots
          ORDER BY sku, snapshot_date DESC
        ),
        cy_trailing AS (
          SELECT aol.sku, SUM(aol.quantity)::int AS units, SUM(aol.quantity * COALESCE(NULLIF(aol.unit_price,0), lp.last_price, 0))::numeric AS revenue
          FROM amazon_order_lines aol
          JOIN amazon_orders ao ON ao.amazon_order_id = aol.amazon_order_id
          LEFT JOIN v_sku_last_price lp ON lp.sku = aol.sku
          WHERE ao.status != 'Canceled' AND ao.order_date::date >= CURRENT_DATE - 89
          GROUP BY aol.sku
        ),
        py_trailing AS (
          SELECT aol.sku, SUM(aol.quantity)::int AS units
          FROM amazon_order_lines aol
          JOIN amazon_orders ao ON ao.amazon_order_id = aol.amazon_order_id
          WHERE ao.status != 'Canceled'
            AND ao.order_date::date BETWEEN (CURRENT_DATE - INTERVAL '1 year' - INTERVAL '89 days')::date AND (CURRENT_DATE - INTERVAL '1 year')::date
          GROUP BY aol.sku
        ),
        py_forward AS (
          SELECT aol.sku, SUM(aol.quantity)::int AS units
          FROM amazon_order_lines aol
          JOIN amazon_orders ao ON ao.amazon_order_id = aol.amazon_order_id
          WHERE ao.status != 'Canceled'
            AND ao.order_date::date BETWEEN (CURRENT_DATE - INTERVAL '1 year')::date AND (CURRENT_DATE - INTERVAL '1 year' + INTERVAL '89 days')::date
          GROUP BY aol.sku
        )
        SELECT li.sku, li.sellable,
          COALESCE(cy.units, 0) AS cy_trailing_units, COALESCE(cy.revenue, 0) AS cy_trailing_revenue,
          COALESCE(pyt.units, 0) AS py_trailing_units, COALESCE(pyf.units, 0) AS py_forward_units
        FROM latest_inv li
        LEFT JOIN cy_trailing cy ON cy.sku = li.sku
        LEFT JOIN py_trailing pyt ON pyt.sku = li.sku
        LEFT JOIN py_forward pyf ON pyf.sku = li.sku
    """
    return pd.read_sql(sql, conn)


def write_forecast(conn, rows):
    """rows: list of dicts with sku, forecast_date, forecast_revenue, low_revenue,
    high_revenue, stage_used, model_used. Replaces this run's horizon per SKU rather than
    appending, so a re-run doesn't leave stale rows behind an now-shorter forecast."""
    if not rows:
        return
    skus = list({r["sku"] for r in rows})
    with conn.cursor() as cur:
        cur.execute("DELETE FROM sales_forecast WHERE sku = ANY(%s)", (skus,))
        psycopg2.extras.execute_values(
            cur,
            """INSERT INTO sales_forecast
               (sku, forecast_date, forecast_revenue, low_revenue, high_revenue, stage_used, model_used, generated_at)
               VALUES %s""",
            [(r["sku"], r["forecast_date"], r["forecast_revenue"], r["low_revenue"], r["high_revenue"],
              r["stage_used"], r["model_used"], r["generated_at"]) for r in rows],
        )
    conn.commit()


def write_exclusions(conn, rows):
    """rows: list of dicts with sku, excluded_date, reason. Same replace-per-SKU approach
    as write_forecast - an outlier that drops out of the rolling detection window on a
    later run shouldn't stay flagged forever."""
    if not rows:
        return
    skus = list({r["sku"] for r in rows})
    with conn.cursor() as cur:
        cur.execute("DELETE FROM sales_forecast_exclusions WHERE sku = ANY(%s)", (skus,))
        psycopg2.extras.execute_values(
            cur,
            "INSERT INTO sales_forecast_exclusions (sku, excluded_date, reason) VALUES %s",
            [(r["sku"], r["excluded_date"], r["reason"]) for r in rows],
        )
    conn.commit()
