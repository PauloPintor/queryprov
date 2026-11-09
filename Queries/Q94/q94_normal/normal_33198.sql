select
   count(distinct ws1.ws_order_number) as "order count",
   sum(ws1.ws_ext_ship_cost) as "total shipping cost",
   sum(ws1.ws_net_profit) as "total net profit",
   array_agg(
        list_value(
            list_value(ws1.prov),
            list_value(date_dim.prov),
            list_value(customer_address.prov),
            list_value(web_site.prov),
            nested.prov
        )
    ) as prov
from
   web_sales ws1
   inner join (
        select
            ws3.ws_order_number,
            ws3.ws_warehouse_sk,
            array_agg(ws2.prov) as prov
        from web_sales ws2
        inner join (
            select distinct
                ws_order_number,
                ws_warehouse_sk
            from web_sales
        ) ws3
          on ws2.ws_order_number = ws3.ws_order_number
         and ws2.ws_warehouse_sk <> ws3.ws_warehouse_sk
        group by
            ws3.ws_order_number,
            ws3.ws_warehouse_sk
    ) nested
      on nested.ws_order_number = ws1.ws_order_number
     and nested.ws_warehouse_sk = ws1.ws_warehouse_sk,
   date_dim,
   customer_address,
   web_site
where d_date between cast('2002-02-09' as date)
                 and cast('2002-02-09' as date) + interval 60 day
  and ws1.ws_ship_date_sk = d_date_sk
  and ws1.ws_ship_addr_sk = ca_address_sk
  and ca_state = 'MS'
  and ws1.ws_web_site_sk = web_site_sk
  and web_company_name = 'pri'
  and not exists (
        select 1
        from web_returns wr1
        where ws1.ws_order_number = wr1.wr_order_number
  )
order by count(distinct ws1.ws_order_number)
limit 100
