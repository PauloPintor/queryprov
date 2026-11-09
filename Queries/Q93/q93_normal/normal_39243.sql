select
    ss_customer_sk,
    sum(act_sales) sumsales,
    array_agg(prov) as prov
from (
    select
        ss_item_sk,
        ss_ticket_number,
        ss_customer_sk,
        case
            when sr_return_quantity is not null then
                (ss_quantity - sr_return_quantity) * ss_sales_price
            else
                ss_quantity * ss_sales_price
        end act_sales,
        list_value(
            store_sales.prov,
            store_returns.prov,
            reason.prov
        ) as prov
    from store_sales
         left outer join store_returns
           on (sr_item_sk = ss_item_sk
               and sr_ticket_number = ss_ticket_number),
         reason
    where sr_reason_sk = r_reason_sk
      and r_reason_desc = 'Did not like the make'
) t
group by ss_customer_sk
order by
    sumsales,
    ss_customer_sk
limit 100
