select B1.B1_LP, B1.B1_CNT, B1.B1_CNTD, B2.B2_LP, B2.B2_CNT, B2.B2_CNTD, B3.B3_LP, B3.B3_CNT, B3.B3_CNTD, B4.B4_LP, B4.B4_CNT, B4.B4_CNTD, B5.B5_LP, B5.B5_CNT, B5.B5_CNTD,
       list_value(B1.prov, B2.prov, B3.prov, B4.prov, B5.prov) as prov
from

     (select avg(ss_list_price) B1_LP
            ,count(ss_list_price) B1_CNT
            ,count(distinct ss_list_price) B1_CNTD
            ,array_agg(store_sales.prov) prov
      from store_sales

      where ss_quantity between 0 and 5
        and (ss_list_price between 98 and 98+10 or ss_coupon_amt between 8570 and 8570+1000 or ss_wholesale_cost between 40 and 40+20)
) B1
,
     (select avg(ss_list_price) B2_LP
            ,count(ss_list_price) B2_CNT
            ,count(distinct ss_list_price) B2_CNTD
            ,array_agg(store_sales.prov) prov
      from store_sales

      where ss_quantity between 6 and 10
        and (ss_list_price between 107 and 107+10 or ss_coupon_amt between 8393 and 8393+1000 or ss_wholesale_cost between 42 and 42+20)
) B2
,
     (select avg(ss_list_price) B3_LP
            ,count(ss_list_price) B3_CNT
            ,count(distinct ss_list_price) B3_CNTD
            ,array_agg(store_sales.prov) prov
      from store_sales

      where ss_quantity between 11 and 15
        and (ss_list_price between 99 and 99+10 or ss_coupon_amt between 8718 and 8718+1000 or ss_wholesale_cost between 43 and 43+20)
) B3
,
     (select avg(ss_list_price) B4_LP
            ,count(ss_list_price) B4_CNT
            ,count(distinct ss_list_price) B4_CNTD
            ,array_agg(store_sales.prov) prov
      from store_sales

      where ss_quantity between 16 and 20
        and (ss_list_price between 102 and 102+10 or ss_coupon_amt between 8578 and 8578+1000 or ss_wholesale_cost between 47 and 47+20)
) B4
,
     (select avg(ss_list_price) B5_LP
            ,count(ss_list_price) B5_CNT
            ,count(distinct ss_list_price) B5_CNTD
            ,array_agg(store_sales.prov) prov
      from store_sales

      where ss_quantity between 21 and 25
        and (ss_list_price between 89 and 89+10 or ss_coupon_amt between 10776 and 10776+1000 or ss_wholesale_cost between 46 and 46+20)
) B5

limit 100
