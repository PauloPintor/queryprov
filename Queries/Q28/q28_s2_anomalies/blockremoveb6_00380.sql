select B1.B1_LP, B1.B1_CNT, B1.B1_CNTD, B2.B2_LP, B2.B2_CNT, B2.B2_CNTD, B3.B3_LP, B3.B3_CNT, B3.B3_CNTD, B4.B4_LP, B4.B4_CNT, B4.B4_CNTD, B5.B5_LP, B5.B5_CNT, B5.B5_CNTD,
       list_value(B1.prov, B2.prov, B3.prov, B4.prov, B5.prov) as prov
from

     (select avg(ss_list_price) B1_LP
            ,count(ss_list_price) B1_CNT
            ,count(distinct ss_list_price) B1_CNTD
            ,array_agg(store_sales.prov) prov
      from store_sales

      where ss_quantity between 0 and 5
        and (ss_list_price between 102 and 102+10 or ss_coupon_amt between 9344 and 9344+1000 or ss_wholesale_cost between 42 and 42+20)
) B1
,
     (select avg(ss_list_price) B2_LP
            ,count(ss_list_price) B2_CNT
            ,count(distinct ss_list_price) B2_CNTD
            ,array_agg(store_sales.prov) prov
      from store_sales

      where ss_quantity between 6 and 10
        and (ss_list_price between 82 and 82+10 or ss_coupon_amt between 7240 and 7240+1000 or ss_wholesale_cost between 43 and 43+20)
) B2
,
     (select avg(ss_list_price) B3_LP
            ,count(ss_list_price) B3_CNT
            ,count(distinct ss_list_price) B3_CNTD
            ,array_agg(store_sales.prov) prov
      from store_sales

      where ss_quantity between 11 and 15
        and (ss_list_price between 96 and 96+10 or ss_coupon_amt between 7997 and 7997+1000 or ss_wholesale_cost between 42 and 42+20)
) B3
,
     (select avg(ss_list_price) B4_LP
            ,count(ss_list_price) B4_CNT
            ,count(distinct ss_list_price) B4_CNTD
            ,array_agg(store_sales.prov) prov
      from store_sales

      where ss_quantity between 16 and 20
        and (ss_list_price between 70 and 70+10 or ss_coupon_amt between 8338 and 8338+1000 or ss_wholesale_cost between 43 and 43+20)
) B4
,
     (select avg(ss_list_price) B5_LP
            ,count(ss_list_price) B5_CNT
            ,count(distinct ss_list_price) B5_CNTD
            ,array_agg(store_sales.prov) prov
      from store_sales

      where ss_quantity between 21 and 25
        and (ss_list_price between 109 and 109+10 or ss_coupon_amt between 7403 and 7403+1000 or ss_wholesale_cost between 50 and 50+20)
) B5

limit 100
