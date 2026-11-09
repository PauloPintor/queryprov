select *
from (
    select
            i.i_category as i_category,
            i.i_class as i_class,
            i.i_brand as i_brand,
            s.s_store_name as s_store_name,
            s.s_company_name as s_company_name,
            d.d_moy as d_moy,
            sum(ss.ss_sales_price) as sum_sales,
            avg(sum(ss.ss_sales_price)) over (partition by i.i_category, i.i_brand, s.s_store_name, s.s_company_name) as avg_monthly_sales,
            array_agg(
                list_value(
                    i.prov,
                    ss.prov,
                    d.prov,
                    s.prov
                )
            ) as prov
    from item i,
         store_sales ss,
         date_dim d,
         store s
    where ss.ss_item_sk = i.i_item_sk
      and ss.ss_sold_date_sk = d.d_date_sk
      and ss.ss_store_sk = s.s_store_sk
      and (
      (
          i.i_category = 'Music'
      and i.i_class = 'country'
      )
   or (
          i.i_category = 'Music'
      and i.i_class = 'classical'
      )
   or (
          i.i_category = 'Books'
      and i.i_class = 'self-help'
      )
   or (
          i.i_category = 'Women'
      and i.i_class = 'maternity'
      )
   or (
          i.i_category = 'Children'
      and i.i_class = 'newborn'
      )
   or (
          i.i_category = 'Jewelry'
      and i.i_class = 'birdal'
      )
  )
    group by
        i.i_category,
        i.i_class,
        i.i_brand,
        s.s_store_name,
        s.s_company_name,
        d.d_moy
) tmp1
where case
        when avg_monthly_sales <> 0
        then abs(sum_sales - avg_monthly_sales) / avg_monthly_sales
        else null
      end > 0.1
order by
    sum_sales - avg_monthly_sales,
    s_store_name

limit 100;
