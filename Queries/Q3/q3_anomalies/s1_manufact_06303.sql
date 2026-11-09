select
    dt.d_year,
    item.i_brand_id as brand_id,
    item.i_brand as brand,
    sum(ss_ext_sales_price) as sum_agg,
    array_agg(
        list_value(
            dt.prov,
            store_sales.prov,
            item.prov
        )
    ) as prov
from date_dim dt,
     store_sales,
     item
where dt.d_date_sk = store_sales.ss_sold_date_sk
  and store_sales.ss_item_sk = item.i_item_sk
  and item.i_manufact_id = 110
  and dt.d_moy = 9
group by
    dt.d_year,
    item.i_brand,
    item.i_brand_id
order by
    dt.d_year,
    sum_agg desc,
    brand_id
limit 100
