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
                    s.prov,
                    p.prov
                )
            ) as prov
    from item i,
         store_sales ss,
         date_dim d,
         store s,
         promotion p
    where ss.ss_item_sk = i.i_item_sk
      and ss.ss_sold_date_sk = d.d_date_sk
      and ss.ss_store_sk = s.s_store_sk
      and d.d_year = 2002
      and ss.ss_promo_sk = p.p_promo_sk
      and (
      (
          i.i_category = 'Electronics'
      and i.i_class = 'camcorders'
      )
   or (
          i.i_category = 'Home'
      and i.i_class = 'accent'
      )
   or (
          i.i_category = 'Home'
      and i.i_class = 'rugs'
      )
   or (
          i.i_category = 'Men'
      and i.i_class = 'shirts'
      )
   or (
          i.i_category = 'Children'
      and i.i_class = 'school-uniforms'
      )
   or (
          i.i_category = 'Books'
      and i.i_class = 'home repair'
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
