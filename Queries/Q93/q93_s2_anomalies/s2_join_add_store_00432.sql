select
    ss_customer_sk,
    sum(act_sales) as sumsales
    ,
    array_agg(prov) as prov
from (

select
        ss.ss_item_sk,
        ss.ss_ticket_number,
        ss.ss_customer_sk,
        case
            when sr.sr_return_quantity is not null then
                (ss.ss_quantity - sr.sr_return_quantity)
                * ss.ss_sales_price
            else
                ss.ss_quantity * ss.ss_sales_price
        end as act_sales,
        list_value(
                ss.prov,
                sr.prov,
                r.prov,
                s.prov
        ) as prov
from store_sales ss
    left join store_returns sr
      on sr.sr_item_sk = ss.ss_item_sk
     and sr.sr_ticket_number = ss.ss_ticket_number
    join reason r
      on r.r_reason_sk = sr.sr_reason_sk
    join store s
      on s.s_store_sk = ss.ss_store_sk
where r.r_reason_desc = 'Not the product that was ordred'

) q93_inner
group by
    ss_customer_sk
order by
    sumsales,
    ss_customer_sk

limit 100;
