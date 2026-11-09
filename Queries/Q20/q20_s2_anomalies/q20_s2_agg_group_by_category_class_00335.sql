select i_category
      ,i_class
      ,sum(cs_ext_sales_price) as itemrevenue
      ,sum(cs_ext_sales_price)*100/sum(sum(cs_ext_sales_price)) over
          (partition by i_class) as revenueratio
      ,array_agg(list_value(catalog_sales.prov, item.prov, date_dim.prov)) as prov
from catalog_sales
    ,item
    ,date_dim
where cs_item_sk = i_item_sk
  and i_category in ('Children', 'Home', 'Men')
  and cs_sold_date_sk = d_date_sk
  and d_date between cast('2002-01-28' as date)
                 and cast('2002-01-28' as date) + INTERVAL 30 DAY
group by i_category
        ,i_class
order by i_category
        ,i_class
limit 100
