select
    substr(r.r_reason_desc, 1, 20) as reason_desc,
    avg(ws.ws_quantity) as avg_ws_quantity,
    avg(wr.wr_refunded_cash) as avg_wr_refunded_cash,
    avg(wr.wr_fee) as avg_wr_fee,
    array_agg(
            list_value(
                ws.prov,
                wr.prov,
                wp.prov,
                cd1.prov,
                cd2.prov,
                ca.prov,
                d.prov,
                r.prov
            )
        ) as prov
from web_sales ws,
     web_returns wr,
     web_page wp,
     customer_demographics cd1,
     customer_demographics cd2,
     customer_address ca,
     date_dim d,
     reason r
where ws.ws_item_sk = wr.wr_item_sk
  and ws.ws_order_number = wr.wr_order_number
  and ws.ws_web_page_sk = wp.wp_web_page_sk
  and ws.ws_sold_date_sk = d.d_date_sk
  and d.d_year = 2002
  and cd1.cd_demo_sk = wr.wr_refunded_cdemo_sk
  and cd2.cd_demo_sk = wr.wr_returning_cdemo_sk
  and ca.ca_address_sk = wr.wr_refunded_addr_sk
  and r.r_reason_sk = wr.wr_reason_sk
  and (
      (
          cd1.cd_marital_status = 'S'
      and cd1.cd_marital_status = cd2.cd_marital_status
      and cd1.cd_education_status = 'Unknown'
      and cd1.cd_education_status = cd2.cd_education_status
      and ws.ws_sales_price between 100.00 and 150.00
      )
   or (
          cd1.cd_marital_status = 'M'
      and cd1.cd_marital_status = cd2.cd_marital_status
      and cd1.cd_education_status = 'College'
      and cd1.cd_education_status = cd2.cd_education_status
      and ws.ws_sales_price between 50.00 and 100.00
      )
   or (
          cd1.cd_marital_status = 'D'
      and cd1.cd_marital_status = cd2.cd_marital_status
      and cd1.cd_education_status = 'College'
      and cd1.cd_education_status = cd2.cd_education_status
      and ws.ws_sales_price between 150.00 and 200.00
      )
  )
  and (
      (
          ca.ca_country = 'United States'
      and ca.ca_state in ('ID', 'WI', 'FL')
      )
   or (
          ca.ca_country = 'United States'
      and ca.ca_state in ('GA', 'IN', 'TX')
      )
   or (
          ca.ca_country = 'United States'
      and ca.ca_state in ('MO', 'SD', 'IL')
      )
  )
group by
    r.r_reason_desc
order by
    substr(r.r_reason_desc, 1, 20),
    avg(ws.ws_quantity),
    avg(wr.wr_refunded_cash),
    avg(wr.wr_fee)

limit 100;
