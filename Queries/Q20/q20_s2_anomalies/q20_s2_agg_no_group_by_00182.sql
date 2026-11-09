select
       sum(cs_ext_sales_price) as itemrevenue
      ,cast(100.0 as double) as revenueratio
      ,array_agg(list_value(catalog_sales.prov, item.prov, date_dim.prov)) as prov
from catalog_sales
    ,item
    ,date_dim
where cs_item_sk = i_item_sk
  and i_category in ('Books', 'Music', 'Men')
  and cs_sold_date_sk = d_date_sk
  and d_date between cast('2002-04-17' as date)
                 and cast('2002-04-17' as date) + INTERVAL 30 DAY
