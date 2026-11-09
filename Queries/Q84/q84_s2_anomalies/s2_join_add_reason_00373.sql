select
    c.c_customer_id as customer_id,
    coalesce(c.c_last_name, '') || ', ' ||
        coalesce(c.c_first_name, '') as customername
    ,
    list_value(
        list_value(
            c.prov,
        ca.prov,
        cd.prov,
        hd.prov,
        ib.prov,
        sr.prov,
        r.prov
        )
    ) as prov
from customer c,
     customer_address ca,
     customer_demographics cd,
     household_demographics hd,
     income_band ib,
     store_returns sr,
     reason r
where ca.ca_city = 'Fairview'
  and c.c_current_addr_sk = ca.ca_address_sk
  and ib.ib_lower_bound >= 25133
  and ib.ib_upper_bound <= 25133 + 50000
  and ib.ib_income_band_sk = hd.hd_income_band_sk
  and cd.cd_demo_sk = c.c_current_cdemo_sk
  and hd.hd_demo_sk = c.c_current_hdemo_sk
  and sr.sr_cdemo_sk = cd.cd_demo_sk
  and sr.sr_reason_sk = r.r_reason_sk
order by
    c.c_customer_id

limit 100;
