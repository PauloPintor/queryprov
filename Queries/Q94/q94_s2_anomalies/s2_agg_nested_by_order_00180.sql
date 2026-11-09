with order_warehouses as (
    select distinct
        ws_order_number,
        ws_warehouse_sk
    from web_sales
),
nested as (
    select
        ow.ws_order_number as ws_order_number
        ,
        array_agg(ws2.prov) as nested_prov
    from web_sales ws2
    join order_warehouses ow
      on ws2.ws_order_number = ow.ws_order_number
     and ws2.ws_warehouse_sk <> ow.ws_warehouse_sk
    group by
        ow.ws_order_number
),
qualified as (
    select
        ws1.ws_order_number,
        ws1.ws_ext_ship_cost,
        ws1.ws_net_profit
        ,
        list_value(
            list_value(ws1.prov),
            list_value(d.prov),
            list_value(ca.prov),
            list_value(web.prov),
            nested.nested_prov
        ) as factor_prov
    from web_sales ws1
    join nested
      on nested.ws_order_number = ws1.ws_order_number
    join date_dim d
      on ws1.ws_ship_date_sk = d.d_date_sk
    join customer_address ca
      on ws1.ws_ship_addr_sk = ca.ca_address_sk
    join web_site web
      on ws1.ws_web_site_sk = web.web_site_sk
    where d.d_date between cast('2002-03-16' as date) and cast('2002-03-16' as date) + interval 60 day
      and ca.ca_state = 'NJ'
      and web.web_company_name = 'pri'
      and not exists (
        select 1
        from web_returns wr1
        where wr1.wr_order_number = ws1.ws_order_number
      )
)

select
    count(
        distinct ws_order_number
    ) as "order count",
    sum(ws_ext_ship_cost) as "total shipping cost",
    sum(ws_net_profit) as "total net profit"
    ,
    array_agg(factor_prov) as prov
from qualified
order by
    count(distinct ws_order_number)
limit 100;
