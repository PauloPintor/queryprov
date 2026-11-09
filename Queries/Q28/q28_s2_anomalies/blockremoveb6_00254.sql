select B1.B1_LP, B1.B1_CNT, B1.B1_CNTD, B2.B2_LP, B2.B2_CNT, B2.B2_CNTD, B3.B3_LP, B3.B3_CNT, B3.B3_CNTD, B4.B4_LP, B4.B4_CNT, B4.B4_CNTD, B5.B5_LP, B5.B5_CNT, B5.B5_CNTD,
       list_value(B1.prov, B2.prov, B3.prov, B4.prov, B5.prov) as prov
from

     (select avg(ss_list_price) B1_LP
            ,count(ss_list_price) B1_CNT
            ,count(distinct ss_list_price) B1_CNTD
            ,array_agg(store_sales.prov) prov
      from store_sales

      where ss_quantity between 0 and 5
        and (ss_list_price between 85 and 85+10 or ss_coupon_amt between 9666 and 9666+1000 or ss_wholesale_cost between 41 and 41+20)
) B1
,
     (select avg(ss_list_price) B2_LP
            ,count(ss_list_price) B2_CNT
            ,count(distinct ss_list_price) B2_CNTD
            ,array_agg(store_sales.prov) prov
      from store_sales

      where ss_quantity between 6 and 10
        and (ss_list_price between 80 and 80+10 or ss_coupon_amt between 8649 and 8649+1000 or ss_wholesale_cost between 38 and 38+20)
) B2
,
     (select avg(ss_list_price) B3_LP
            ,count(ss_list_price) B3_CNT
            ,count(distinct ss_list_price) B3_CNTD
            ,array_agg(store_sales.prov) prov
      from store_sales

      where ss_quantity between 11 and 15
        and (ss_list_price between 89 and 89+10 or ss_coupon_amt between 8461 and 8461+1000 or ss_wholesale_cost between 37 and 37+20)
) B3
,
     (select avg(ss_list_price) B4_LP
            ,count(ss_list_price) B4_CNT
            ,count(distinct ss_list_price) B4_CNTD
            ,array_agg(store_sales.prov) prov
      from store_sales

      where ss_quantity between 16 and 20
        and (ss_list_price between 94 and 94+10 or ss_coupon_amt between 9123 and 9123+1000 or ss_wholesale_cost between 35 and 35+20)
) B4
,
     (select avg(ss_list_price) B5_LP
            ,count(ss_list_price) B5_CNT
            ,count(distinct ss_list_price) B5_CNTD
            ,array_agg(store_sales.prov) prov
      from store_sales

      where ss_quantity between 21 and 25
        and (ss_list_price between 113 and 113+10 or ss_coupon_amt between 8941 and 8941+1000 or ss_wholesale_cost between 35 and 35+20)
) B5

limit 100
