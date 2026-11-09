select i_item_id
      ,i_item_desc
      ,i_category
      ,i_class
      ,i_current_price
      ,sum(cs_ext_sales_price) as itemrevenue
      ,sum(cs_ext_sales_price)*100/sum(sum(cs_ext_sales_price)) over
          (partition by i_class) as revenueratio
      ,array_agg(list_value(catalog_sales.prov, catalog_page.prov, item.prov, date_dim.prov)) as prov
from catalog_sales
    ,catalog_page
    ,item
    ,date_dim
where cs_item_sk = i_item_sk
  and cs_catalog_page_sk = cp_catalog_page_sk
  and i_category in ('Home', 'Books', 'Children')
  and cs_sold_date_sk = d_date_sk
  and d_date between cast('2002-05-28' as date)
                 and cast('2002-05-28' as date) + INTERVAL 30 DAY
group by i_item_id
        ,i_item_desc
        ,i_category
        ,i_class
        ,i_current_price
order by i_category
        ,i_class
        ,i_item_id
        ,i_item_desc
        ,revenueratio
limit 100
