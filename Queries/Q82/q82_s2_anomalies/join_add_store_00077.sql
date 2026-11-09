select
    i_item_id,
    i_item_desc,
    i_current_price,
    array_agg(
           list_value(
               item.prov,
            inventory.prov,
            date_dim.prov,
            store_sales.prov,
               s.prov
           )
       ) as prov
from item,
     inventory,
     date_dim,
     store_sales,
     store s
where i_current_price between 51 and 51+30
  and inv_item_sk = i_item_sk
  and d_date_sk = inv_date_sk
  and d_date between cast('2002-04-27' as date)
                 and cast('2002-04-27' as date) + interval 60 day
  and i_manufact_id in (568, 190, 55, 49)
  and inv_quantity_on_hand between 100 and 500
  and ss_item_sk = i_item_sk

  and store_sales.ss_store_sk = s.s_store_sk
group by
    i_item_id,
    i_item_desc,
    i_current_price
order by
    i_item_id
limit 100;
