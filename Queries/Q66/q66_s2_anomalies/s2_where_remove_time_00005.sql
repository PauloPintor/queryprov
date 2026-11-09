WITH base AS (
SELECT
    w.w_warehouse_name,
    w.w_warehouse_sq_ft,
    w.w_city,
    w.w_county,
    w.w_state,
    w.w_country,
    'RUPEKSA' || ',' || 'USPS' AS ship_carriers,
    d.d_year AS year,
    sum(case when d.d_moy = 1 then ws.ws_ext_sales_price * ws.ws_quantity else 0 end) AS jan_sales,
    sum(case when d.d_moy = 2 then ws.ws_ext_sales_price * ws.ws_quantity else 0 end) AS feb_sales,
    sum(case when d.d_moy = 3 then ws.ws_ext_sales_price * ws.ws_quantity else 0 end) AS mar_sales,
    sum(case when d.d_moy = 4 then ws.ws_ext_sales_price * ws.ws_quantity else 0 end) AS apr_sales,
    sum(case when d.d_moy = 5 then ws.ws_ext_sales_price * ws.ws_quantity else 0 end) AS may_sales,
    sum(case when d.d_moy = 6 then ws.ws_ext_sales_price * ws.ws_quantity else 0 end) AS jun_sales,
    sum(case when d.d_moy = 7 then ws.ws_ext_sales_price * ws.ws_quantity else 0 end) AS jul_sales,
    sum(case when d.d_moy = 8 then ws.ws_ext_sales_price * ws.ws_quantity else 0 end) AS aug_sales,
    sum(case when d.d_moy = 9 then ws.ws_ext_sales_price * ws.ws_quantity else 0 end) AS sep_sales,
    sum(case when d.d_moy = 10 then ws.ws_ext_sales_price * ws.ws_quantity else 0 end) AS oct_sales,
    sum(case when d.d_moy = 11 then ws.ws_ext_sales_price * ws.ws_quantity else 0 end) AS nov_sales,
    sum(case when d.d_moy = 12 then ws.ws_ext_sales_price * ws.ws_quantity else 0 end) AS dec_sales,
    sum(case when d.d_moy = 1 then ws.ws_net_paid_inc_ship * ws.ws_quantity else 0 end) AS jan_net,
    sum(case when d.d_moy = 2 then ws.ws_net_paid_inc_ship * ws.ws_quantity else 0 end) AS feb_net,
    sum(case when d.d_moy = 3 then ws.ws_net_paid_inc_ship * ws.ws_quantity else 0 end) AS mar_net,
    sum(case when d.d_moy = 4 then ws.ws_net_paid_inc_ship * ws.ws_quantity else 0 end) AS apr_net,
    sum(case when d.d_moy = 5 then ws.ws_net_paid_inc_ship * ws.ws_quantity else 0 end) AS may_net,
    sum(case when d.d_moy = 6 then ws.ws_net_paid_inc_ship * ws.ws_quantity else 0 end) AS jun_net,
    sum(case when d.d_moy = 7 then ws.ws_net_paid_inc_ship * ws.ws_quantity else 0 end) AS jul_net,
    sum(case when d.d_moy = 8 then ws.ws_net_paid_inc_ship * ws.ws_quantity else 0 end) AS aug_net,
    sum(case when d.d_moy = 9 then ws.ws_net_paid_inc_ship * ws.ws_quantity else 0 end) AS sep_net,
    sum(case when d.d_moy = 10 then ws.ws_net_paid_inc_ship * ws.ws_quantity else 0 end) AS oct_net,
    sum(case when d.d_moy = 11 then ws.ws_net_paid_inc_ship * ws.ws_quantity else 0 end) AS nov_net,
    sum(case when d.d_moy = 12 then ws.ws_net_paid_inc_ship * ws.ws_quantity else 0 end) AS dec_net,
    array_agg(list_value(ws.prov, w.prov, d.prov, t.prov, sm.prov)) AS prov
FROM web_sales ws
JOIN warehouse w ON ws.ws_warehouse_sk = w.w_warehouse_sk
JOIN date_dim d ON ws.ws_sold_date_sk = d.d_date_sk
JOIN time_dim t ON ws.ws_sold_time_sk = t.t_time_sk
JOIN ship_mode sm ON ws.ws_ship_mode_sk = sm.sm_ship_mode_sk
WHERE d.d_year = 2002
  AND sm.sm_carrier IN ('RUPEKSA', 'USPS')
GROUP BY
    w.w_warehouse_name,
    w.w_warehouse_sq_ft,
    w.w_city,
    w.w_county,
    w.w_state,
    w.w_country,
    d.d_year

UNION ALL

SELECT
    w.w_warehouse_name,
    w.w_warehouse_sq_ft,
    w.w_city,
    w.w_county,
    w.w_state,
    w.w_country,
    'RUPEKSA' || ',' || 'USPS' AS ship_carriers,
    d.d_year AS year,
    sum(case when d.d_moy = 1 then cs.cs_sales_price * cs.cs_quantity else 0 end) AS jan_sales,
    sum(case when d.d_moy = 2 then cs.cs_sales_price * cs.cs_quantity else 0 end) AS feb_sales,
    sum(case when d.d_moy = 3 then cs.cs_sales_price * cs.cs_quantity else 0 end) AS mar_sales,
    sum(case when d.d_moy = 4 then cs.cs_sales_price * cs.cs_quantity else 0 end) AS apr_sales,
    sum(case when d.d_moy = 5 then cs.cs_sales_price * cs.cs_quantity else 0 end) AS may_sales,
    sum(case when d.d_moy = 6 then cs.cs_sales_price * cs.cs_quantity else 0 end) AS jun_sales,
    sum(case when d.d_moy = 7 then cs.cs_sales_price * cs.cs_quantity else 0 end) AS jul_sales,
    sum(case when d.d_moy = 8 then cs.cs_sales_price * cs.cs_quantity else 0 end) AS aug_sales,
    sum(case when d.d_moy = 9 then cs.cs_sales_price * cs.cs_quantity else 0 end) AS sep_sales,
    sum(case when d.d_moy = 10 then cs.cs_sales_price * cs.cs_quantity else 0 end) AS oct_sales,
    sum(case when d.d_moy = 11 then cs.cs_sales_price * cs.cs_quantity else 0 end) AS nov_sales,
    sum(case when d.d_moy = 12 then cs.cs_sales_price * cs.cs_quantity else 0 end) AS dec_sales,
    sum(case when d.d_moy = 1 then cs.cs_net_paid_inc_tax * cs.cs_quantity else 0 end) AS jan_net,
    sum(case when d.d_moy = 2 then cs.cs_net_paid_inc_tax * cs.cs_quantity else 0 end) AS feb_net,
    sum(case when d.d_moy = 3 then cs.cs_net_paid_inc_tax * cs.cs_quantity else 0 end) AS mar_net,
    sum(case when d.d_moy = 4 then cs.cs_net_paid_inc_tax * cs.cs_quantity else 0 end) AS apr_net,
    sum(case when d.d_moy = 5 then cs.cs_net_paid_inc_tax * cs.cs_quantity else 0 end) AS may_net,
    sum(case when d.d_moy = 6 then cs.cs_net_paid_inc_tax * cs.cs_quantity else 0 end) AS jun_net,
    sum(case when d.d_moy = 7 then cs.cs_net_paid_inc_tax * cs.cs_quantity else 0 end) AS jul_net,
    sum(case when d.d_moy = 8 then cs.cs_net_paid_inc_tax * cs.cs_quantity else 0 end) AS aug_net,
    sum(case when d.d_moy = 9 then cs.cs_net_paid_inc_tax * cs.cs_quantity else 0 end) AS sep_net,
    sum(case when d.d_moy = 10 then cs.cs_net_paid_inc_tax * cs.cs_quantity else 0 end) AS oct_net,
    sum(case when d.d_moy = 11 then cs.cs_net_paid_inc_tax * cs.cs_quantity else 0 end) AS nov_net,
    sum(case when d.d_moy = 12 then cs.cs_net_paid_inc_tax * cs.cs_quantity else 0 end) AS dec_net,
    array_agg(list_value(cs.prov, w.prov, d.prov, t.prov, sm.prov)) AS prov
FROM catalog_sales cs
JOIN warehouse w ON cs.cs_warehouse_sk = w.w_warehouse_sk
JOIN date_dim d ON cs.cs_sold_date_sk = d.d_date_sk
JOIN time_dim t ON cs.cs_sold_time_sk = t.t_time_sk
JOIN ship_mode sm ON cs.cs_ship_mode_sk = sm.sm_ship_mode_sk
WHERE d.d_year = 2002
  AND sm.sm_carrier IN ('RUPEKSA', 'USPS')
GROUP BY
    w.w_warehouse_name,
    w.w_warehouse_sq_ft,
    w.w_city,
    w.w_county,
    w.w_state,
    w.w_country,
    d.d_year
)
SELECT
    w_warehouse_name,
    w_warehouse_sq_ft,
    w_city,
    w_county,
    w_state,
    w_country,
    ship_carriers,
    year,
    sum(jan_sales) AS jan_sales,
    sum(feb_sales) AS feb_sales,
    sum(mar_sales) AS mar_sales,
    sum(apr_sales) AS apr_sales,
    sum(may_sales) AS may_sales,
    sum(jun_sales) AS jun_sales,
    sum(jul_sales) AS jul_sales,
    sum(aug_sales) AS aug_sales,
    sum(sep_sales) AS sep_sales,
    sum(oct_sales) AS oct_sales,
    sum(nov_sales) AS nov_sales,
    sum(dec_sales) AS dec_sales,
    sum(jan_sales / nullif(w_warehouse_sq_ft, 0)) AS jan_sales_per_sq_foot,
    sum(feb_sales / nullif(w_warehouse_sq_ft, 0)) AS feb_sales_per_sq_foot,
    sum(mar_sales / nullif(w_warehouse_sq_ft, 0)) AS mar_sales_per_sq_foot,
    sum(apr_sales / nullif(w_warehouse_sq_ft, 0)) AS apr_sales_per_sq_foot,
    sum(may_sales / nullif(w_warehouse_sq_ft, 0)) AS may_sales_per_sq_foot,
    sum(jun_sales / nullif(w_warehouse_sq_ft, 0)) AS jun_sales_per_sq_foot,
    sum(jul_sales / nullif(w_warehouse_sq_ft, 0)) AS jul_sales_per_sq_foot,
    sum(aug_sales / nullif(w_warehouse_sq_ft, 0)) AS aug_sales_per_sq_foot,
    sum(sep_sales / nullif(w_warehouse_sq_ft, 0)) AS sep_sales_per_sq_foot,
    sum(oct_sales / nullif(w_warehouse_sq_ft, 0)) AS oct_sales_per_sq_foot,
    sum(nov_sales / nullif(w_warehouse_sq_ft, 0)) AS nov_sales_per_sq_foot,
    sum(dec_sales / nullif(w_warehouse_sq_ft, 0)) AS dec_sales_per_sq_foot,
    sum(jan_net) AS jan_net,
    sum(feb_net) AS feb_net,
    sum(mar_net) AS mar_net,
    sum(apr_net) AS apr_net,
    sum(may_net) AS may_net,
    sum(jun_net) AS jun_net,
    sum(jul_net) AS jul_net,
    sum(aug_net) AS aug_net,
    sum(sep_net) AS sep_net,
    sum(oct_net) AS oct_net,
    sum(nov_net) AS nov_net,
    sum(dec_net) AS dec_net,
    list_reduce(array_agg(prov), (x, y) -> list_concat(x, y)) AS prov
FROM base
GROUP BY
    w_warehouse_name,
    w_warehouse_sq_ft,
    w_city,
    w_county,
    w_state,
    w_country,
    ship_carriers,
    year
ORDER BY w_warehouse_name
LIMIT 100
