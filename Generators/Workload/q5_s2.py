#!/usr/bin/env python3
"""Portable Q5 S2 workload generator.

Run with --help for paths and options. Requires duckdb, numpy and pandas.
The database is opened read-only. Original SQL and sampling are preserved.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

DB_SCHEMA = "tpcds"

SEED = 42

# Para as escalas atuais, 500 por tipo chega:
# escala 50k usa 625 anomalias no teste.
# Com 5 tipos, isso dá 125 por tipo.
# 500 por tipo dá margem.
N_ANOMALY_PER_TYPE = 500

S2_SUBTYPES = [
    "q5_s2_join_add_table",
    "q5_s2_agg_no_group_by",
    "q5_s2_agg_group_by_channel",
    "q5_s2_limit_1",
    "q5_s2_limit_none",
]


# ============================================================
# SQL: Q5 S2 JOIN ADD TABLE
# Adds customer as a provenance-bearing joined table in sales branches.
# Returns branches are preserved as in the original query.
# ============================================================

SQL_Q5_S2_JOIN_ADD_TABLE = """
with ssr as
(
    select s_store_id,
           sum(sales_price) as sales,
           sum(profit) as profit,
           sum(return_amt) as returns,
           sum(net_loss) as profit_loss,
           array_agg(
               list_concat(
                   src_prov,
                   list_value(date_dim.prov, store.prov)
               )
           ) as prov
    from
    (
        select ss_store_sk as store_sk,
               ss_sold_date_sk as date_sk,
               ss_ext_sales_price as sales_price,
               ss_net_profit as profit,
               cast(0 as decimal(7,2)) as return_amt,
               cast(0 as decimal(7,2)) as net_loss,
               list_value(store_sales.prov, customer.prov) as src_prov
        from store_sales,
             customer
        where ss_customer_sk = c_customer_sk

        union all

        select sr_store_sk as store_sk,
               sr_returned_date_sk as date_sk,
               cast(0 as decimal(7,2)) as sales_price,
               cast(0 as decimal(7,2)) as profit,
               sr_return_amt as return_amt,
               sr_net_loss as net_loss,
               list_value(store_returns.prov) as src_prov
        from store_returns
    ) salesreturns,
      date_dim,
      store
    where date_sk = d_date_sk
      and d_date between cast('{sdate}' as date)
                     and cast('{sdate}' as date) + INTERVAL {window_days} DAY
      and store_sk = s_store_sk
    group by s_store_id
),
csr as
(
    select cp_catalog_page_id,
           sum(sales_price) as sales,
           sum(profit) as profit,
           sum(return_amt) as returns,
           sum(net_loss) as profit_loss,
           array_agg(
               list_concat(
                   src_prov,
                   list_value(date_dim.prov, catalog_page.prov)
               )
           ) as prov
    from
    (
        select cs_catalog_page_sk as page_sk,
               cs_sold_date_sk as date_sk,
               cs_ext_sales_price as sales_price,
               cs_net_profit as profit,
               cast(0 as decimal(7,2)) as return_amt,
               cast(0 as decimal(7,2)) as net_loss,
               list_value(catalog_sales.prov, customer.prov) as src_prov
        from catalog_sales,
             customer
        where cs_bill_customer_sk = c_customer_sk

        union all

        select cr_catalog_page_sk as page_sk,
               cr_returned_date_sk as date_sk,
               cast(0 as decimal(7,2)) as sales_price,
               cast(0 as decimal(7,2)) as profit,
               cr_return_amount as return_amt,
               cr_net_loss as net_loss,
               list_value(catalog_returns.prov) as src_prov
        from catalog_returns
    ) salesreturns,
      date_dim,
      catalog_page
    where date_sk = d_date_sk
      and d_date between cast('{sdate}' as date)
                     and cast('{sdate}' as date) + INTERVAL {window_days} DAY
      and page_sk = cp_catalog_page_sk
    group by cp_catalog_page_id
),
wsr as
(
    select web_site_id,
           sum(sales_price) as sales,
           sum(profit) as profit,
           sum(return_amt) as returns,
           sum(net_loss) as profit_loss,
           array_agg(
               list_concat(
                   src_prov,
                   list_value(date_dim.prov, web_site.prov)
               )
           ) as prov
    from
    (
        select ws_web_site_sk as wsr_web_site_sk,
               ws_sold_date_sk as date_sk,
               ws_ext_sales_price as sales_price,
               ws_net_profit as profit,
               cast(0 as decimal(7,2)) as return_amt,
               cast(0 as decimal(7,2)) as net_loss,
               list_value(web_sales.prov, customer.prov) as src_prov
        from web_sales,
             customer
        where ws_bill_customer_sk = c_customer_sk

        union all

        select ws_web_site_sk as wsr_web_site_sk,
               wr_returned_date_sk as date_sk,
               cast(0 as decimal(7,2)) as sales_price,
               cast(0 as decimal(7,2)) as profit,
               wr_return_amt as return_amt,
               wr_net_loss as net_loss,
               case
                   when web_sales.prov is null then list_value(web_returns.prov)
                   else list_value(web_returns.prov, web_sales.prov)
               end as src_prov
        from web_returns
        left outer join web_sales
          on wr_item_sk = ws_item_sk
         and wr_order_number = ws_order_number
    ) salesreturns,
      date_dim,
      web_site
    where date_sk = d_date_sk
      and d_date between cast('{sdate}' as date)
                     and cast('{sdate}' as date) + INTERVAL {window_days} DAY
      and wsr_web_site_sk = web_site_sk
    group by web_site_id
)
select channel,
       id,
       sum(sales) as sales,
       sum(returns) as returns,
       sum(profit) as profit,
       array_agg(prov) as prov
from
(
    select 'store channel' as channel,
           'store' || s_store_id as id,
           sales,
           returns,
           profit - profit_loss as profit,
           prov
    from ssr

    union all

    select 'catalog channel' as channel,
           'catalog_page' || cp_catalog_page_id as id,
           sales,
           returns,
           profit - profit_loss as profit,
           prov
    from csr

    union all

    select 'web channel' as channel,
           'web_site' || web_site_id as id,
           sales,
           returns,
           profit - profit_loss as profit,
           prov
    from wsr
) x
group by rollup(channel, id)
order by channel, id
limit 100
"""


COUNT_Q5_S2_JOIN_ADD_TABLE = """
select count(*) as n_rows
from (
""" + SQL_Q5_S2_JOIN_ADD_TABLE.strip().rstrip(";") + """
) t
"""


# ============================================================
# SQL: Q5 S2 AGG NO GROUP BY
# Final global aggregation only.
# Internal channel CTEs remain unchanged.
# ============================================================

SQL_Q5_S2_AGG_NO_GROUP_BY = """
with ssr as
(
    select s_store_id,
           sum(sales_price) as sales,
           sum(profit) as profit,
           sum(return_amt) as returns,
           sum(net_loss) as profit_loss,
           array_agg(
               list_concat(
                   src_prov,
                   list_value(date_dim.prov, store.prov)
               )
           ) as prov
    from
    (
        select ss_store_sk as store_sk,
               ss_sold_date_sk as date_sk,
               ss_ext_sales_price as sales_price,
               ss_net_profit as profit,
               cast(0 as decimal(7,2)) as return_amt,
               cast(0 as decimal(7,2)) as net_loss,
               list_value(store_sales.prov) as src_prov
        from store_sales

        union all

        select sr_store_sk as store_sk,
               sr_returned_date_sk as date_sk,
               cast(0 as decimal(7,2)) as sales_price,
               cast(0 as decimal(7,2)) as profit,
               sr_return_amt as return_amt,
               sr_net_loss as net_loss,
               list_value(store_returns.prov) as src_prov
        from store_returns
    ) salesreturns,
      date_dim,
      store
    where date_sk = d_date_sk
      and d_date between cast('{sdate}' as date)
                     and cast('{sdate}' as date) + INTERVAL {window_days} DAY
      and store_sk = s_store_sk
    group by s_store_id
),
csr as
(
    select cp_catalog_page_id,
           sum(sales_price) as sales,
           sum(profit) as profit,
           sum(return_amt) as returns,
           sum(net_loss) as profit_loss,
           array_agg(
               list_concat(
                   src_prov,
                   list_value(date_dim.prov, catalog_page.prov)
               )
           ) as prov
    from
    (
        select cs_catalog_page_sk as page_sk,
               cs_sold_date_sk as date_sk,
               cs_ext_sales_price as sales_price,
               cs_net_profit as profit,
               cast(0 as decimal(7,2)) as return_amt,
               cast(0 as decimal(7,2)) as net_loss,
               list_value(catalog_sales.prov) as src_prov
        from catalog_sales

        union all

        select cr_catalog_page_sk as page_sk,
               cr_returned_date_sk as date_sk,
               cast(0 as decimal(7,2)) as sales_price,
               cast(0 as decimal(7,2)) as profit,
               cr_return_amount as return_amt,
               cr_net_loss as net_loss,
               list_value(catalog_returns.prov) as src_prov
        from catalog_returns
    ) salesreturns,
      date_dim,
      catalog_page
    where date_sk = d_date_sk
      and d_date between cast('{sdate}' as date)
                     and cast('{sdate}' as date) + INTERVAL {window_days} DAY
      and page_sk = cp_catalog_page_sk
    group by cp_catalog_page_id
),
wsr as
(
    select web_site_id,
           sum(sales_price) as sales,
           sum(profit) as profit,
           sum(return_amt) as returns,
           sum(net_loss) as profit_loss,
           array_agg(
               list_concat(
                   src_prov,
                   list_value(date_dim.prov, web_site.prov)
               )
           ) as prov
    from
    (
        select ws_web_site_sk as wsr_web_site_sk,
               ws_sold_date_sk as date_sk,
               ws_ext_sales_price as sales_price,
               ws_net_profit as profit,
               cast(0 as decimal(7,2)) as return_amt,
               cast(0 as decimal(7,2)) as net_loss,
               list_value(web_sales.prov) as src_prov
        from web_sales

        union all

        select ws_web_site_sk as wsr_web_site_sk,
               wr_returned_date_sk as date_sk,
               cast(0 as decimal(7,2)) as sales_price,
               cast(0 as decimal(7,2)) as profit,
               wr_return_amt as return_amt,
               wr_net_loss as net_loss,
               case
                   when web_sales.prov is null then list_value(web_returns.prov)
                   else list_value(web_returns.prov, web_sales.prov)
               end as src_prov
        from web_returns
        left outer join web_sales
          on wr_item_sk = ws_item_sk
         and wr_order_number = ws_order_number
    ) salesreturns,
      date_dim,
      web_site
    where date_sk = d_date_sk
      and d_date between cast('{sdate}' as date)
                     and cast('{sdate}' as date) + INTERVAL {window_days} DAY
      and wsr_web_site_sk = web_site_sk
    group by web_site_id
)
select
       sum(sales) as sales,
       sum(returns) as returns,
       sum(profit) as profit,
       array_agg(prov) as prov
from
(
    select 'store channel' as channel,
           'store' || s_store_id as id,
           sales,
           returns,
           profit - profit_loss as profit,
           prov
    from ssr

    union all

    select 'catalog channel' as channel,
           'catalog_page' || cp_catalog_page_id as id,
           sales,
           returns,
           profit - profit_loss as profit,
           prov
    from csr

    union all

    select 'web channel' as channel,
           'web_site' || web_site_id as id,
           sales,
           returns,
           profit - profit_loss as profit,
           prov
    from wsr
) x
limit 100
"""


COUNT_Q5_S2_AGG_NO_GROUP_BY = """
select count(*) as n_rows
from (
""" + SQL_Q5_S2_AGG_NO_GROUP_BY.strip().rstrip(";") + """
) t
where sales is not null
"""


# ============================================================
# SQL: Q5 S2 AGG GROUP BY CHANNEL
# Coarser final aggregation.
# ============================================================

SQL_Q5_S2_AGG_GROUP_BY_CHANNEL = """
with ssr as
(
    select s_store_id,
           sum(sales_price) as sales,
           sum(profit) as profit,
           sum(return_amt) as returns,
           sum(net_loss) as profit_loss,
           array_agg(
               list_concat(
                   src_prov,
                   list_value(date_dim.prov, store.prov)
               )
           ) as prov
    from
    (
        select ss_store_sk as store_sk,
               ss_sold_date_sk as date_sk,
               ss_ext_sales_price as sales_price,
               ss_net_profit as profit,
               cast(0 as decimal(7,2)) as return_amt,
               cast(0 as decimal(7,2)) as net_loss,
               list_value(store_sales.prov) as src_prov
        from store_sales

        union all

        select sr_store_sk as store_sk,
               sr_returned_date_sk as date_sk,
               cast(0 as decimal(7,2)) as sales_price,
               cast(0 as decimal(7,2)) as profit,
               sr_return_amt as return_amt,
               sr_net_loss as net_loss,
               list_value(store_returns.prov) as src_prov
        from store_returns
    ) salesreturns,
      date_dim,
      store
    where date_sk = d_date_sk
      and d_date between cast('{sdate}' as date)
                     and cast('{sdate}' as date) + INTERVAL {window_days} DAY
      and store_sk = s_store_sk
    group by s_store_id
),
csr as
(
    select cp_catalog_page_id,
           sum(sales_price) as sales,
           sum(profit) as profit,
           sum(return_amt) as returns,
           sum(net_loss) as profit_loss,
           array_agg(
               list_concat(
                   src_prov,
                   list_value(date_dim.prov, catalog_page.prov)
               )
           ) as prov
    from
    (
        select cs_catalog_page_sk as page_sk,
               cs_sold_date_sk as date_sk,
               cs_ext_sales_price as sales_price,
               cs_net_profit as profit,
               cast(0 as decimal(7,2)) as return_amt,
               cast(0 as decimal(7,2)) as net_loss,
               list_value(catalog_sales.prov) as src_prov
        from catalog_sales

        union all

        select cr_catalog_page_sk as page_sk,
               cr_returned_date_sk as date_sk,
               cast(0 as decimal(7,2)) as sales_price,
               cast(0 as decimal(7,2)) as profit,
               cr_return_amount as return_amt,
               cr_net_loss as net_loss,
               list_value(catalog_returns.prov) as src_prov
        from catalog_returns
    ) salesreturns,
      date_dim,
      catalog_page
    where date_sk = d_date_sk
      and d_date between cast('{sdate}' as date)
                     and cast('{sdate}' as date) + INTERVAL {window_days} DAY
      and page_sk = cp_catalog_page_sk
    group by cp_catalog_page_id
),
wsr as
(
    select web_site_id,
           sum(sales_price) as sales,
           sum(profit) as profit,
           sum(return_amt) as returns,
           sum(net_loss) as profit_loss,
           array_agg(
               list_concat(
                   src_prov,
                   list_value(date_dim.prov, web_site.prov)
               )
           ) as prov
    from
    (
        select ws_web_site_sk as wsr_web_site_sk,
               ws_sold_date_sk as date_sk,
               ws_ext_sales_price as sales_price,
               ws_net_profit as profit,
               cast(0 as decimal(7,2)) as return_amt,
               cast(0 as decimal(7,2)) as net_loss,
               list_value(web_sales.prov) as src_prov
        from web_sales

        union all

        select ws_web_site_sk as wsr_web_site_sk,
               wr_returned_date_sk as date_sk,
               cast(0 as decimal(7,2)) as sales_price,
               cast(0 as decimal(7,2)) as profit,
               wr_return_amt as return_amt,
               wr_net_loss as net_loss,
               case
                   when web_sales.prov is null then list_value(web_returns.prov)
                   else list_value(web_returns.prov, web_sales.prov)
               end as src_prov
        from web_returns
        left outer join web_sales
          on wr_item_sk = ws_item_sk
         and wr_order_number = ws_order_number
    ) salesreturns,
      date_dim,
      web_site
    where date_sk = d_date_sk
      and d_date between cast('{sdate}' as date)
                     and cast('{sdate}' as date) + INTERVAL {window_days} DAY
      and wsr_web_site_sk = web_site_sk
    group by web_site_id
)
select channel,
       sum(sales) as sales,
       sum(returns) as returns,
       sum(profit) as profit,
       array_agg(prov) as prov
from
(
    select 'store channel' as channel,
           'store' || s_store_id as id,
           sales,
           returns,
           profit - profit_loss as profit,
           prov
    from ssr

    union all

    select 'catalog channel' as channel,
           'catalog_page' || cp_catalog_page_id as id,
           sales,
           returns,
           profit - profit_loss as profit,
           prov
    from csr

    union all

    select 'web channel' as channel,
           'web_site' || web_site_id as id,
           sales,
           returns,
           profit - profit_loss as profit,
           prov
    from wsr
) x
group by channel
order by channel
limit 100
"""


COUNT_Q5_S2_AGG_GROUP_BY_CHANNEL = """
select count(*) as n_rows
from (
""" + SQL_Q5_S2_AGG_GROUP_BY_CHANNEL.strip().rstrip(";") + """
) t
"""


# ============================================================
# SQL: Q5 S2 LIMIT 1
# ============================================================

SQL_Q5_S2_LIMIT_1 = """
with ssr as
(
    select s_store_id,
           sum(sales_price) as sales,
           sum(profit) as profit,
           sum(return_amt) as returns,
           sum(net_loss) as profit_loss,
           array_agg(
               list_concat(
                   src_prov,
                   list_value(date_dim.prov, store.prov)
               )
           ) as prov
    from
    (
        select ss_store_sk as store_sk,
               ss_sold_date_sk as date_sk,
               ss_ext_sales_price as sales_price,
               ss_net_profit as profit,
               cast(0 as decimal(7,2)) as return_amt,
               cast(0 as decimal(7,2)) as net_loss,
               list_value(store_sales.prov) as src_prov
        from store_sales

        union all

        select sr_store_sk as store_sk,
               sr_returned_date_sk as date_sk,
               cast(0 as decimal(7,2)) as sales_price,
               cast(0 as decimal(7,2)) as profit,
               sr_return_amt as return_amt,
               sr_net_loss as net_loss,
               list_value(store_returns.prov) as src_prov
        from store_returns
    ) salesreturns,
      date_dim,
      store
    where date_sk = d_date_sk
      and d_date between cast('{sdate}' as date)
                     and cast('{sdate}' as date) + INTERVAL {window_days} DAY
      and store_sk = s_store_sk
    group by s_store_id
),
csr as
(
    select cp_catalog_page_id,
           sum(sales_price) as sales,
           sum(profit) as profit,
           sum(return_amt) as returns,
           sum(net_loss) as profit_loss,
           array_agg(
               list_concat(
                   src_prov,
                   list_value(date_dim.prov, catalog_page.prov)
               )
           ) as prov
    from
    (
        select cs_catalog_page_sk as page_sk,
               cs_sold_date_sk as date_sk,
               cs_ext_sales_price as sales_price,
               cs_net_profit as profit,
               cast(0 as decimal(7,2)) as return_amt,
               cast(0 as decimal(7,2)) as net_loss,
               list_value(catalog_sales.prov) as src_prov
        from catalog_sales

        union all

        select cr_catalog_page_sk as page_sk,
               cr_returned_date_sk as date_sk,
               cast(0 as decimal(7,2)) as sales_price,
               cast(0 as decimal(7,2)) as profit,
               cr_return_amount as return_amt,
               cr_net_loss as net_loss,
               list_value(catalog_returns.prov) as src_prov
        from catalog_returns
    ) salesreturns,
      date_dim,
      catalog_page
    where date_sk = d_date_sk
      and d_date between cast('{sdate}' as date)
                     and cast('{sdate}' as date) + INTERVAL {window_days} DAY
      and page_sk = cp_catalog_page_sk
    group by cp_catalog_page_id
),
wsr as
(
    select web_site_id,
           sum(sales_price) as sales,
           sum(profit) as profit,
           sum(return_amt) as returns,
           sum(net_loss) as profit_loss,
           array_agg(
               list_concat(
                   src_prov,
                   list_value(date_dim.prov, web_site.prov)
               )
           ) as prov
    from
    (
        select ws_web_site_sk as wsr_web_site_sk,
               ws_sold_date_sk as date_sk,
               ws_ext_sales_price as sales_price,
               ws_net_profit as profit,
               cast(0 as decimal(7,2)) as return_amt,
               cast(0 as decimal(7,2)) as net_loss,
               list_value(web_sales.prov) as src_prov
        from web_sales

        union all

        select ws_web_site_sk as wsr_web_site_sk,
               wr_returned_date_sk as date_sk,
               cast(0 as decimal(7,2)) as sales_price,
               cast(0 as decimal(7,2)) as profit,
               wr_return_amt as return_amt,
               wr_net_loss as net_loss,
               case
                   when web_sales.prov is null then list_value(web_returns.prov)
                   else list_value(web_returns.prov, web_sales.prov)
               end as src_prov
        from web_returns
        left outer join web_sales
          on wr_item_sk = ws_item_sk
         and wr_order_number = ws_order_number
    ) salesreturns,
      date_dim,
      web_site
    where date_sk = d_date_sk
      and d_date between cast('{sdate}' as date)
                     and cast('{sdate}' as date) + INTERVAL {window_days} DAY
      and wsr_web_site_sk = web_site_sk
    group by web_site_id
)
select channel,
       id,
       sum(sales) as sales,
       sum(returns) as returns,
       sum(profit) as profit,
       array_agg(prov) as prov
from
(
    select 'store channel' as channel,
           'store' || s_store_id as id,
           sales,
           returns,
           profit - profit_loss as profit,
           prov
    from ssr

    union all

    select 'catalog channel' as channel,
           'catalog_page' || cp_catalog_page_id as id,
           sales,
           returns,
           profit - profit_loss as profit,
           prov
    from csr

    union all

    select 'web channel' as channel,
           'web_site' || web_site_id as id,
           sales,
           returns,
           profit - profit_loss as profit,
           prov
    from wsr
) x
group by rollup(channel, id)
order by channel, id
limit 1
"""


COUNT_Q5_S2_LIMIT_1 = """
select count(*) as n_rows
from (
""" + SQL_Q5_S2_LIMIT_1.strip().rstrip(";") + """
) t
"""


# ============================================================
# SQL: Q5 S2 LIMIT NONE
# Removes the final LIMIT 100.
# This is valid for Q5 because the unbounded grouped result is usually
# much larger than 100 rows, unlike Q3 where no-limit was equivalent.
# ============================================================

SQL_Q5_S2_LIMIT_NONE = """
with ssr as
(
    select s_store_id,
           sum(sales_price) as sales,
           sum(profit) as profit,
           sum(return_amt) as returns,
           sum(net_loss) as profit_loss,
           array_agg(
               list_concat(
                   src_prov,
                   list_value(date_dim.prov, store.prov)
               )
           ) as prov
    from
    (
        select ss_store_sk as store_sk,
               ss_sold_date_sk as date_sk,
               ss_ext_sales_price as sales_price,
               ss_net_profit as profit,
               cast(0 as decimal(7,2)) as return_amt,
               cast(0 as decimal(7,2)) as net_loss,
               list_value(store_sales.prov) as src_prov
        from store_sales

        union all

        select sr_store_sk as store_sk,
               sr_returned_date_sk as date_sk,
               cast(0 as decimal(7,2)) as sales_price,
               cast(0 as decimal(7,2)) as profit,
               sr_return_amt as return_amt,
               sr_net_loss as net_loss,
               list_value(store_returns.prov) as src_prov
        from store_returns
    ) salesreturns,
      date_dim,
      store
    where date_sk = d_date_sk
      and d_date between cast('{sdate}' as date)
                     and cast('{sdate}' as date) + INTERVAL {window_days} DAY
      and store_sk = s_store_sk
    group by s_store_id
),
csr as
(
    select cp_catalog_page_id,
           sum(sales_price) as sales,
           sum(profit) as profit,
           sum(return_amt) as returns,
           sum(net_loss) as profit_loss,
           array_agg(
               list_concat(
                   src_prov,
                   list_value(date_dim.prov, catalog_page.prov)
               )
           ) as prov
    from
    (
        select cs_catalog_page_sk as page_sk,
               cs_sold_date_sk as date_sk,
               cs_ext_sales_price as sales_price,
               cs_net_profit as profit,
               cast(0 as decimal(7,2)) as return_amt,
               cast(0 as decimal(7,2)) as net_loss,
               list_value(catalog_sales.prov) as src_prov
        from catalog_sales

        union all

        select cr_catalog_page_sk as page_sk,
               cr_returned_date_sk as date_sk,
               cast(0 as decimal(7,2)) as sales_price,
               cast(0 as decimal(7,2)) as profit,
               cr_return_amount as return_amt,
               cr_net_loss as net_loss,
               list_value(catalog_returns.prov) as src_prov
        from catalog_returns
    ) salesreturns,
      date_dim,
      catalog_page
    where date_sk = d_date_sk
      and d_date between cast('{sdate}' as date)
                     and cast('{sdate}' as date) + INTERVAL {window_days} DAY
      and page_sk = cp_catalog_page_sk
    group by cp_catalog_page_id
),
wsr as
(
    select web_site_id,
           sum(sales_price) as sales,
           sum(profit) as profit,
           sum(return_amt) as returns,
           sum(net_loss) as profit_loss,
           array_agg(
               list_concat(
                   src_prov,
                   list_value(date_dim.prov, web_site.prov)
               )
           ) as prov
    from
    (
        select ws_web_site_sk as wsr_web_site_sk,
               ws_sold_date_sk as date_sk,
               ws_ext_sales_price as sales_price,
               ws_net_profit as profit,
               cast(0 as decimal(7,2)) as return_amt,
               cast(0 as decimal(7,2)) as net_loss,
               list_value(web_sales.prov) as src_prov
        from web_sales

        union all

        select ws_web_site_sk as wsr_web_site_sk,
               wr_returned_date_sk as date_sk,
               cast(0 as decimal(7,2)) as sales_price,
               cast(0 as decimal(7,2)) as profit,
               wr_return_amt as return_amt,
               wr_net_loss as net_loss,
               case
                   when web_sales.prov is null then list_value(web_returns.prov)
                   else list_value(web_returns.prov, web_sales.prov)
               end as src_prov
        from web_returns
        left outer join web_sales
          on wr_item_sk = ws_item_sk
         and wr_order_number = ws_order_number
    ) salesreturns,
      date_dim,
      web_site
    where date_sk = d_date_sk
      and d_date between cast('{sdate}' as date)
                     and cast('{sdate}' as date) + INTERVAL {window_days} DAY
      and wsr_web_site_sk = web_site_sk
    group by web_site_id
)
select channel,
       id,
       sum(sales) as sales,
       sum(returns) as returns,
       sum(profit) as profit,
       array_agg(prov) as prov
from
(
    select 'store channel' as channel,
           'store' || s_store_id as id,
           sales,
           returns,
           profit - profit_loss as profit,
           prov
    from ssr

    union all

    select 'catalog channel' as channel,
           'catalog_page' || cp_catalog_page_id as id,
           sales,
           returns,
           profit - profit_loss as profit,
           prov
    from csr

    union all

    select 'web channel' as channel,
           'web_site' || web_site_id as id,
           sales,
           returns,
           profit - profit_loss as profit,
           prov
    from wsr
) x
group by rollup(channel, id)
order by channel, id
"""


COUNT_Q5_S2_LIMIT_NONE = """
select count(*) as n_rows
from (
""" + SQL_Q5_S2_LIMIT_NONE.strip().rstrip(";") + """
) t
"""


SQL_BY_SUBTYPE = {
    "q5_s2_join_add_table": SQL_Q5_S2_JOIN_ADD_TABLE,
    "q5_s2_agg_no_group_by": SQL_Q5_S2_AGG_NO_GROUP_BY,
    "q5_s2_agg_group_by_channel": SQL_Q5_S2_AGG_GROUP_BY_CHANNEL,
    "q5_s2_limit_1": SQL_Q5_S2_LIMIT_1,
    "q5_s2_limit_none": SQL_Q5_S2_LIMIT_NONE,
}


COUNT_BY_SUBTYPE = {
    "q5_s2_join_add_table": COUNT_Q5_S2_JOIN_ADD_TABLE,
    "q5_s2_agg_no_group_by": COUNT_Q5_S2_AGG_NO_GROUP_BY,
    "q5_s2_agg_group_by_channel": COUNT_Q5_S2_AGG_GROUP_BY_CHANNEL,
    "q5_s2_limit_1": COUNT_Q5_S2_LIMIT_1,
    "q5_s2_limit_none": COUNT_Q5_S2_LIMIT_NONE,
}


# ============================================================
# HELPERS
# ============================================================

def render(template: str, params: dict) -> str:
    return template.format(**params)


def write_metadata(path: Path, rows: list[dict]) -> None:
    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def load_normal_metadata(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Normal metadata not found: {path}")

    df = pd.read_csv(path)

    required = {"sdate", "window_days"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Normal metadata must contain {sorted(required)}. "
            f"Missing: {sorted(missing)}"
        )

    df = df.copy()
    df["sdate"] = df["sdate"].astype(str)
    df["window_days"] = df["window_days"].astype(int)

    if len(df) == 0:
        raise ValueError("Normal metadata is empty.")

    return df


def remove_old_s2_files(out_dir: Path) -> None:
    """
    Remove only old Q5 S2 files generated by this script.
    It does NOT remove S1 files and does NOT delete q5_anomalies.
    """

    if not out_dir.exists():
        return

    for subtype in S2_SUBTYPES:
        for p in out_dir.glob(f"{subtype}_*.sql"):
            p.unlink()

    for name in [
        "metadata_s2.csv",
        "q5_s2_metadata.csv",
        "q5_s2_summary.csv",
    ]:
        p = out_dir / name
        if p.exists():
            p.unlink()


def sample_normal_params(
    normal_df: pd.DataFrame,
    rng: np.random.Generator,
) -> dict:
    row = normal_df.iloc[int(rng.integers(0, len(normal_df)))]

    params = {
        "sdate": str(row["sdate"]),
        "window_days": int(row["window_days"]),
    }

    if "file" in normal_df.columns:
        params["source_normal_file"] = str(row["file"])

    return params


def count_rows(
    con: duckdb.DuckDBPyConnection,
    template: str,
    params: dict,
) -> int:
    return int(con.execute(render(template, params)).fetchone()[0])


# ============================================================
# GENERATION
# ============================================================

def write_s2_queries(
    out_dir: Path,
    subtype: str,
    n: int,
    normal_df: pd.DataFrame,
    con: duckdb.DuckDBPyConnection,
    rng: np.random.Generator,
) -> list[dict]:

    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    i = 0
    attempts = 0
    # Do NOT enforce unique (sdate, window_days) pairs.
    # Q5 normal metadata has many instances but only a small number of unique dates.
    # This matches the S1 generator behaviour: queries may share parameters,
    # but remain separate generated instances/files.
    max_attempts = max(100_000, n * 100)

    sql_template = SQL_BY_SUBTYPE[subtype]
    count_template = COUNT_BY_SUBTYPE[subtype]

    while i < n:
        attempts += 1

        if attempts > max_attempts:
            raise RuntimeError(
                f"Too many attempts while generating {subtype}: "
                f"{attempts} attempts for {i}/{n} queries. "
                "The sampled normal metadata may be too restrictive or counts may keep returning zero."
            )

        params = sample_normal_params(normal_df, rng)

        try:
            s2_n_rows = count_rows(con, count_template, params)
        except Exception as e:
            print(f"[ERROR] count failed for {subtype} {params}: {e}")
            continue

        if s2_n_rows == 0:
            continue

        filename = f"{subtype}_{i:05d}.sql"

        (out_dir / filename).write_text(
            render(sql_template, params).strip() + "\n",
            encoding="utf-8",
        )

        row = {
            "file": filename,
            "label": 1,
            "scenario": subtype,
            "sdate": params["sdate"],
            "window_days": int(params["window_days"]),
            "s2_n_rows": s2_n_rows,
        }

        if "source_normal_file" in params:
            row["source_normal_file"] = params["source_normal_file"]

        rows.append(row)

        i += 1

        if i % 100 == 0:
            print(f"  [GEN] {subtype}: {i}/{n}")

    print(f"  [OK] {subtype}: {len(rows)} queries, attempts={attempts}")

    return rows


def merge_metadata(
    out_dir: Path,
    s2_rows: list[dict],
) -> None:
    """
    Keeps old S1 metadata if it exists, removes old S2 rows, then appends new S2 rows.
    """

    metadata_path = out_dir / "metadata.csv"

    if metadata_path.exists():
        old = pd.read_csv(metadata_path)

        if "scenario" in old.columns:
            old = old[~old["scenario"].astype(str).isin(S2_SUBTYPES)].copy()
        else:
            old = old.copy()

        new = pd.DataFrame(s2_rows)

        cols = list(old.columns)
        for c in new.columns:
            if c not in cols:
                cols.append(c)

        merged = pd.concat(
            [
                old.reindex(columns=cols),
                new.reindex(columns=cols),
            ],
            ignore_index=True,
        )
    else:
        merged = pd.DataFrame(s2_rows)

    merged.to_csv(metadata_path, index=False)
    print(f"[WRITE] {metadata_path} rows={len(merged)}")


# ============================================================
# DIAGNOSTICS
# ============================================================

def print_diagnostics(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)

    print("\n" + "=" * 80)
    print("[Q5 S2 GENERATION DIAGNOSTICS]")
    print("=" * 80)

    print(f"\n[TOTAL S2] n={len(df)}")
    print(f"  scenarios={df['scenario'].value_counts().to_dict()}")

    print("\n[PARAMS FROM NORMAL DISTRIBUTION]")
    print(f"  sdate unique={df['sdate'].nunique()}")
    print(f"  sdate min={df['sdate'].min()} max={df['sdate'].max()}")
    print(f"  window_days unique={sorted(df['window_days'].unique().astype(int).tolist())}")

    print("\n[ROWS]")
    summary = (
        df.groupby("scenario")
        .agg(
            n=("file", "count"),
            s2_n_rows_min=("s2_n_rows", "min"),
            s2_n_rows_median=("s2_n_rows", "median"),
            s2_n_rows_max=("s2_n_rows", "max"),
            unique_sdate=("sdate", "nunique"),
        )
        .reset_index()
    )

    print(summary.to_string(index=False))
    print("=" * 80)


# ============================================================
# MAIN
# ============================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generate Q5 S2 anomalies only. "
            "Read normal metadata and write SQL files and S2 metadata."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--db", "--db-path", dest="db_path", type=Path, required=True, help="Existing DuckDB database.")
    p.add_argument("--schema", default=DB_SCHEMA, help="Database search path.")
    p.add_argument("--normal-metadata", type=Path, required=True, help="Metadata CSV of the normal Q5 workload.")

    output = p.add_mutually_exclusive_group(required=True)
    output.add_argument(
        "--output-dir", type=Path,
        help="Parent directory; output is written to its q5_s2_anomalies subfolder.",
    )
    output.add_argument(
        "--out-anom",
        type=Path,
        help="Legacy alternative: specify the S2 output pool directly.",
    )

    p.add_argument("--seed", type=int, default=SEED, help="Random seed.")
    p.add_argument("--n-per-type", type=int, default=N_ANOMALY_PER_TYPE, help="Number of queries per S2 subtype.")

    p.add_argument(
        "--clean-s2",
        action="store_true",
        help=(
            "Explicitly regenerate S2 outputs in an existing pool. "
            "Remove old generated S2 files; do not delete unrelated SQL files."
        ),
    )

    p.add_argument(
        "--no-merge-metadata",
        action="store_true",
        help=(
            "Do not update metadata.csv in the output pool. "
            "The other three original CSV outputs are still written."
        ),
    )

    args = p.parse_args(argv)
    args.db_path = args.db_path.expanduser().resolve()
    args.normal_metadata = args.normal_metadata.expanduser().resolve()
    if args.output_dir is not None:
        args.output_dir = args.output_dir.expanduser().resolve()
        if args.output_dir.exists() and not args.output_dir.is_dir():
            p.error(f"Output path is not a directory: {args.output_dir}")
        args.out_anom = args.output_dir / "q5_s2_anomalies"
    else:
        args.out_anom = args.out_anom.expanduser().resolve()
    if not args.db_path.is_file():
        p.error(f"Database file does not exist: {args.db_path}")
    if not args.normal_metadata.is_file():
        p.error(f"Normal metadata file does not exist: {args.normal_metadata}")
    if args.seed < 0:
        p.error("--seed must be non-negative.")
    if args.n_per_type <= 0:
        p.error("--n-per-type must be positive.")
    if not args.schema.strip():
        p.error("--schema must not be empty.")
    if args.normal_metadata.is_relative_to(args.out_anom):
        p.error("Normal input metadata must be outside the S2 output pool.")
    if args.out_anom.exists():
        if not args.out_anom.is_dir():
            p.error(f"Output pool is not a directory: {args.out_anom}")
        if not args.clean_s2 and next(args.out_anom.iterdir(), None) is not None:
            p.error(f"Output pool is not empty: {args.out_anom}. Use a new output directory or explicitly pass --clean-s2.")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    print("=" * 80)
    print("[Q5 S2 QUERY GENERATOR]")
    print("=" * 80)
    print(f"DB_PATH={args.db_path}")
    print(f"DB_SCHEMA={args.schema}")
    print(f"NORMAL_METADATA={args.normal_metadata}")
    print(f"OUT_ANOM={args.out_anom}")
    print(f"SEED={args.seed}")
    print(f"N_ANOMALY_PER_TYPE={args.n_per_type}")
    print(f"S2_SUBTYPES={S2_SUBTYPES}")
    print("=" * 80)

    normal_df = load_normal_metadata(args.normal_metadata)

    print(f"\n[NORMAL METADATA] rows={len(normal_df)}")
    print(f"  sdate min={normal_df['sdate'].min()} max={normal_df['sdate'].max()}")
    print(f"  sdate unique={normal_df['sdate'].nunique()}")
    print(f"  window_days unique={sorted(normal_df['window_days'].unique().astype(int).tolist())}")

    con = duckdb.connect(str(args.db_path), read_only=True)
    try:
        con.execute("SET search_path = ?", [args.schema])

        if args.clean_s2:
            remove_old_s2_files(args.out_anom)

        rng = np.random.default_rng(args.seed)

        all_rows: list[dict] = []

        for subtype in S2_SUBTYPES:
            print(f"\n[GEN] {subtype}")

            rows = write_s2_queries(
                out_dir=args.out_anom,
                subtype=subtype,
                n=args.n_per_type,
                normal_df=normal_df,
                con=con,
                rng=rng,
            )

            all_rows.extend(rows)

    finally:
        con.close()

    metadata_s2 = args.out_anom / "metadata_s2.csv"
    write_metadata(metadata_s2, all_rows)
    print(f"\n[WRITE] {metadata_s2} rows={len(all_rows)}")

    q5_s2_metadata = args.out_anom / "q5_s2_metadata.csv"
    write_metadata(q5_s2_metadata, all_rows)
    print(f"[WRITE] {q5_s2_metadata} rows={len(all_rows)}")

    summary = (
        pd.DataFrame(all_rows)
        .groupby("scenario")
        .agg(
            n=("file", "count"),
            s2_n_rows_min=("s2_n_rows", "min"),
            s2_n_rows_median=("s2_n_rows", "median"),
            s2_n_rows_max=("s2_n_rows", "max"),
            unique_sdate=("sdate", "nunique"),
        )
        .reset_index()
    )

    summary_path = args.out_anom / "q5_s2_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[WRITE] {summary_path} rows={len(summary)}")

    if not args.no_merge_metadata:
        merge_metadata(args.out_anom, all_rows)

    print_diagnostics(all_rows)

    print("\n[DONE]")


if __name__ == "__main__":
    main()

