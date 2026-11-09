select
       i_item_id,
       ca_country,
       ca_state,
       ca_county,
       avg(cast(cs_quantity as decimal(12,2))) as agg1,
       avg(cast(cs_list_price as decimal(12,2))) as agg2,
       avg(cast(cs_coupon_amt as decimal(12,2))) as agg3,
       avg(cast(cs_sales_price as decimal(12,2))) as agg4,
       avg(cast(cs_net_profit as decimal(12,2))) as agg5,
       avg(cast(c_birth_year as decimal(12,2))) as agg6,
       avg(cast(cd1.cd_dep_count as decimal(12,2))) as agg7,
       array_agg(
           list_value(
               catalog_sales.prov,
               cd1.prov,
               cd2.prov,
               customer.prov,
               customer_address.prov,
               date_dim.prov,
               item.prov
           )
       ) as prov
from catalog_sales,
     customer_demographics cd1,
     customer_demographics cd2,
     customer,
     customer_address,
     date_dim,
     item
where cs_sold_date_sk = d_date_sk
  and cs_item_sk = i_item_sk
  and cs_bill_cdemo_sk = cd1.cd_demo_sk
  and cs_bill_customer_sk = c_customer_sk

  and cd1.cd_gender = 'M'
  and cd1.cd_education_status = '2 yr Degree'

  and c_current_cdemo_sk = cd2.cd_demo_sk
  and c_current_addr_sk = ca_address_sk

  and c_birth_month in (4, 5, 9, 10, 11, 12)

  and d_year = 2002

  and ca_state in ('NE', 'VA', 'IN', 'TX', 'CO', 'ID', 'IA')

group by rollup (
    i_item_id,
    ca_country,
    ca_state,
    ca_county
)

order by
    ca_country,
    ca_state,
    ca_county,
    i_item_id
