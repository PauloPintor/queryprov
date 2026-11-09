with avg_discount as (
    select
        i_avg.i_manufact_id as threshold_key,
        1.3 * avg(
            ws_avg.ws_ext_discount_amt
        ) as discount_threshold
        ,
        array_agg(
            list_value(
                    ws_avg.prov,
                    d_avg.prov,
                    i_avg.prov
            )
        ) as avg_prov
    from web_sales ws_avg
    join date_dim d_avg
      on d_avg.d_date_sk = ws_avg.ws_sold_date_sk
    join item i_avg
      on i_avg.i_item_sk = ws_avg.ws_item_sk
    
    where d_avg.d_date between cast('2002-03-26' as date)
                           and cast('2002-03-26' as date)
                               + interval 90 day

    
    group by
        i_avg.i_manufact_id

),
qualified as (
    select
        ws.ws_ext_discount_amt,
        list_value(
                ws.prov,
                i.prov,
                d.prov
        ) as outer_prov,
        ad.avg_prov as avg_prov
    from web_sales ws
    join item i
      on i.i_item_sk = ws.ws_item_sk
    join date_dim d
      on d.d_date_sk = ws.ws_sold_date_sk
    join avg_discount ad
      on ad.threshold_key = i.i_manufact_id
    where i.i_manufact_id = 443
      and d.d_date between cast('2002-03-26' as date) and cast('2002-03-26' as date) + interval 90 day
      and ws.ws_ext_discount_amt > ad.discount_threshold
)

select
    sum(ws_ext_discount_amt) as "Excess Discount Amount",
    array_agg(
        list_value(
            list_value(outer_prov),
            avg_prov
        )
    ) as prov
from qualified
order by
    sum(ws_ext_discount_amt)
limit 100;
