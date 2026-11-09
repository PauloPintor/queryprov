select *
from (
    select
        i_category,
        i_class,
        i_brand,
        s_store_name,
        s_company_name,
        d_moy,
        sum(ss_sales_price) as sum_sales,
        avg(sum(ss_sales_price)) over
          (partition by i_category, i_brand, s_store_name, s_company_name)
          as avg_monthly_sales,
        array_agg(
            list_value(
                item.prov,
                store_sales.prov,
                date_dim.prov,
                store.prov
            )
        ) as prov
    from item,
         store_sales,
         date_dim,
         store
    where ss_item_sk = i_item_sk
      and ss_sold_date_sk = d_date_sk
      and ss_store_sk = s_store_sk
      and d_year = 2002

          and (
                 (i_category = 'Children' and i_class = 'toddlers')
              or (i_category = 'Music' and i_class = 'pop')
              or (i_category = 'Children' and i_class = 'school-uniforms')
              or (i_category = 'Jewelry' and i_class = 'semi-precious')
              or (i_category = 'Shoes' and i_class = 'kids')
              or (i_category = 'Sports' and i_class = 'outdoor')
          )

    group by
        i_category,
        i_class,
        i_brand,
        s_store_name,
        s_company_name,
        d_moy
) tmp1
where case
        when avg_monthly_sales <> 0
        then abs(sum_sales - avg_monthly_sales) / avg_monthly_sales
        else null
      end > 0.1
order by
    sum_sales - avg_monthly_sales,
    s_store_name
limit 100
