select B1.B1_LP, B1.B1_CNT, B1.B1_CNTD, B2.B2_LP, B2.B2_CNT, B2.B2_CNTD, B3.B3_LP, B3.B3_CNT, B3.B3_CNTD, B4.B4_LP, B4.B4_CNT, B4.B4_CNTD, B5.B5_LP, B5.B5_CNT, B5.B5_CNTD,
       list_value(B1.prov, B2.prov, B3.prov, B4.prov, B5.prov) as prov
from

     (select avg(ss_list_price) B1_LP
            ,count(ss_list_price) B1_CNT
            ,count(distinct ss_list_price) B1_CNTD
            ,array_agg(store_sales.prov) prov
      from store_sales

      where ss_quantity between 0 and 5
        and (ss_list_price between 93 and 93+10 or ss_coupon_amt between 9495 and 9495+1000 or ss_wholesale_cost between 40 and 40+20)
) B1
,
     (select avg(ss_list_price) B2_LP
            ,count(ss_list_price) B2_CNT
            ,count(distinct ss_list_price) B2_CNTD
            ,array_agg(store_sales.prov) prov
      from store_sales

      where ss_quantity between 6 and 10
        and (ss_list_price between 94 and 94+10 or ss_coupon_amt between 9399 and 9399+1000 or ss_wholesale_cost between 50 and 50+20)
) B2
,
     (select avg(ss_list_price) B3_LP
            ,count(ss_list_price) B3_CNT
            ,count(distinct ss_list_price) B3_CNTD
            ,array_agg(store_sales.prov) prov
      from store_sales

      where ss_quantity between 11 and 15
        and (ss_list_price between 97 and 97+10 or ss_coupon_amt between 9046 and 9046+1000 or ss_wholesale_cost between 46 and 46+20)
) B3
,
     (select avg(ss_list_price) B4_LP
            ,count(ss_list_price) B4_CNT
            ,count(distinct ss_list_price) B4_CNTD
            ,array_agg(store_sales.prov) prov
      from store_sales

      where ss_quantity between 16 and 20
        and (ss_list_price between 78 and 78+10 or ss_coupon_amt between 8683 and 8683+1000 or ss_wholesale_cost between 37 and 37+20)
) B4
,
     (select avg(ss_list_price) B5_LP
            ,count(ss_list_price) B5_CNT
            ,count(distinct ss_list_price) B5_CNTD
            ,array_agg(store_sales.prov) prov
      from store_sales

      where ss_quantity between 21 and 25
        and (ss_list_price between 100 and 100+10 or ss_coupon_amt between 7883 and 7883+1000 or ss_wholesale_cost between 39 and 39+20)
) B5

limit 100
