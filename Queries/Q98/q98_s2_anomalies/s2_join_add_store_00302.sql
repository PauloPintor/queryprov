select
    i.i_item_id as i_item_id,
    i.i_item_desc as i_item_desc,
    i.i_category as i_category,
    i.i_class as i_class,
    i.i_current_price as i_current_price,
    sum(ss.ss_ext_sales_price) as itemrevenue,
    sum(ss.ss_ext_sales_price) * 100.0 / sum(sum(ss.ss_ext_sales_price)) over (partition by i.i_class) as revenueratio,
    array_agg(
        list_value(
                ss.prov,
                i.prov,
                d.prov,
                s.prov
        )
    ) as prov
from store_sales ss
    join item i
      on ss.ss_item_sk = i.i_item_sk
    join date_dim d
      on ss.ss_sold_date_sk = d.d_date_sk
    join store s
      on ss.ss_store_sk = s.s_store_sk
where i.i_category in (
        'Sports',
        'Children',
        'Books'
      )
  and d.d_date between cast('2002-04-09' as date)
                   and cast('2002-04-09' as date)
                       + interval 30 day
group by
    i.i_item_id,
    i.i_item_desc,
    i.i_category,
    i.i_class,
    i.i_current_price
order by
    i_category,
    i_class,
    i_item_id,
    i_item_desc,
    revenueratio;
