"""Сборка пользовательских таблиц из событий T-ECD.

Окно «теста» — последние 28 дней (1281–1308), предпериод для CUPED — 28 дней до него (1253–1280).
В выборку попадает каждый, кто хоть раз зашёл в сервис в окне теста: так в A/B-тест попадают все,
кто увидел изменение. Пользователь без активности в предпериоде получает нули.
"""
import duckdb

DAY_US = 86_400_000_000
TEST = (1281, 1308)
PRE = (1253, 1280)

con = duckdb.connect()
con.execute("SET threads=8")

RETAIL = f"""
create or replace view r as
select user_id, timestamp // {DAY_US} as d, action_type
from read_parquet('data/raw/retail/events/*.pq')
where timestamp // {DAY_US} between {PRE[0]} and {TEST[1]}
"""
MARKET = f"""
create or replace view m as
select user_id, timestamp // {DAY_US} as d, action_type
from read_parquet('data/raw/marketplace/events/*.pq')
where timestamp // {DAY_US} between {PRE[0]} and {TEST[1]}
"""
con.execute(RETAIL)
con.execute(MARKET)


def period(col, lo, hi, cond):
    return f"count(*) filter (where d between {lo} and {hi} and {cond}) as {col}"


def days(col, lo, hi, cond):
    return f"count(distinct d) filter (where d between {lo} and {hi} and {cond}) as {col}"


# Ритейл: просмотры, корзина, купленные товары, дни с покупкой
con.execute(f"""
copy (
  select user_id,
         min(d) filter (where d >= {TEST[0]}) as first_day,
         {period('views', *TEST, "action_type = 'view'")},
         {period('carts', *TEST, "action_type = 'added_to_cart'")},
         {period('items', *TEST, "action_type = 'order'")},
         {days('buy_days', *TEST, "action_type = 'order'")},
         {days('active_days', *TEST, 'true')},
         {period('views_pre', *PRE, "action_type = 'view'")},
         {period('items_pre', *PRE, "action_type = 'order'")},
         {days('buy_days_pre', *PRE, "action_type = 'order'")},
         {days('active_days_pre', *PRE, 'true')}
  from r
  group by user_id
  having count(*) filter (where d >= {TEST[0]}) > 0
) to 'data/processed/retail_users.parquet' (format parquet)
""")

# Маркетплейс: показы, клики, переходы к партнёру
con.execute(f"""
copy (
  select user_id,
         min(d) filter (where d >= {TEST[0]}) as first_day,
         {period('views', *TEST, "action_type = 'view'")},
         {period('clicks', *TEST, "action_type = 'click'")},
         {period('clickouts', *TEST, "action_type = 'clickout'")},
         {days('active_days', *TEST, 'true')},
         {period('views_pre', *PRE, "action_type = 'view'")},
         {period('clicks_pre', *PRE, "action_type = 'click'")},
         {period('clickouts_pre', *PRE, "action_type = 'clickout'")},
         {days('active_days_pre', *PRE, 'true')}
  from m
  group by user_id
  having count(*) filter (where d >= {TEST[0]}) > 0
) to 'data/processed/market_users.parquet' (format parquet)
""")

# Накопленные метрики ритейла по дням теста — для симуляции подглядывания
con.execute(f"""
copy (
  select user_id, d,
         count(*) filter (where action_type = 'order') as items,
         (count(*) filter (where action_type = 'order') > 0)::int as buy_day
  from r
  where d >= {TEST[0]}
  group by user_id, d
) to 'data/processed/retail_daily.parquet' (format parquet)
""")

for t in ['retail_users', 'market_users', 'retail_daily']:
    n = con.execute(f"select count(*) from read_parquet('data/processed/{t}.parquet')").fetchone()[0]
    print(t, n)
