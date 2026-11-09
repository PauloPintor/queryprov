select B1.B1_LP, B1.B1_CNT, B1.B1_CNTD, B2.B2_LP, B2.B2_CNT, B2.B2_CNTD, B3.B3_LP, B3.B3_CNT, B3.B3_CNTD, B4.B4_LP, B4.B4_CNT, B4.B4_CNTD, B5.B5_LP, B5.B5_CNT, B5.B5_CNTD,
       list_value(B1.prov, B2.prov, B3.prov, B4.prov, B5.prov) as prov
from

     (select avg(ss_list_price) B1_LP
            ,count(ss_list_price) B1_CNT
            ,count(distinct ss_list_price) B1_CNTD
            ,array_agg(store_sales.prov) prov
      from store_sales

      where ss_quantity between 0 and 5
        and (ss_list_price between 79 and 79+10 or ss_coupon_amt between 8138 and 8138+1000 or ss_wholesale_cost between 42 and 42+20)
) B1
,
     (select avg(ss_list_price) B2_LP
            ,count(ss_list_price) B2_CNT
            ,count(distinct ss_list_price) B2_CNTD
            ,array_agg(store_sales.prov) prov
      from store_sales

      where ss_quantity between 6 and 10
        and (ss_list_price between 73 and 73+10 or ss_coupon_amt between 8335 and 8335+1000 or ss_wholesale_cost between 37 and 37+20)
) B2
,
     (select avg(ss_list_price) B3_LP
            ,count(ss_list_price) B3_CNT
            ,count(distinct ss_list_price) B3_CNTD
            ,array_agg(store_sales.prov) prov
      from store_sales

      where ss_quantity between 11 and 15
        and (ss_list_price between 90 and 90+10 or ss_coupon_amt between 8642 and 8642+1000 or ss_wholesale_cost between 48 and 48+20)
) B3
,
     (select avg(ss_list_price) B4_LP
            ,count(ss_list_price) B4_CNT
            ,count(distinct ss_list_price) B4_CNTD
            ,array_agg(store_sales.prov) prov
      from store_sales

      where ss_quantity between 16 and 20
        and (ss_list_price between 109 and 109+10 or ss_coupon_amt between 8667 and 8667+1000 or ss_wholesale_cost between 42 and 42+20)
) B4
,
     (select avg(ss_list_price) B5_LP
            ,count(ss_list_price) B5_CNT
            ,count(distinct ss_list_price) B5_CNTD
            ,array_agg(store_sales.prov) prov
      from store_sales

      where ss_quantity between 21 and 25
        and (ss_list_price between 106 and 106+10 or ss_coupon_amt between 8541 and 8541+1000 or ss_wholesale_cost between 40 and 40+20)
) B5

limit 100
