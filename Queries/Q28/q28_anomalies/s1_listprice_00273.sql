select B1.B1_LP, B1.B1_CNT, B1.B1_CNTD,
       B2.B2_LP, B2.B2_CNT, B2.B2_CNTD,
       B3.B3_LP, B3.B3_CNT, B3.B3_CNTD,
       B4.B4_LP, B4.B4_CNT, B4.B4_CNTD,
       B5.B5_LP, B5.B5_CNT, B5.B5_CNTD,
       B6.B6_LP, B6.B6_CNT, B6.B6_CNTD,
       list_value(B1.prov, B2.prov, B3.prov, B4.prov, B5.prov, B6.prov) as prov
from
     (select avg(ss_list_price) B1_LP
            ,count(ss_list_price) B1_CNT
            ,count(distinct ss_list_price) B1_CNTD
            ,array_agg(store_sales.prov) prov
      from store_sales
      where ss_quantity between 0 and 5
        and (ss_list_price between 137 and 137+10
             or ss_coupon_amt between 9031 and 9031+1000
             or ss_wholesale_cost between 44 and 44+20)) B1,
     (select avg(ss_list_price) B2_LP
            ,count(ss_list_price) B2_CNT
            ,count(distinct ss_list_price) B2_CNTD
            ,array_agg(store_sales.prov) prov
      from store_sales
      where ss_quantity between 6 and 10
        and (ss_list_price between 53 and 53+10
          or ss_coupon_amt between 7794 and 7794+1000
          or ss_wholesale_cost between 47 and 47+20)) B2,
     (select avg(ss_list_price) B3_LP
            ,count(ss_list_price) B3_CNT
            ,count(distinct ss_list_price) B3_CNTD
            ,array_agg(store_sales.prov) prov
      from store_sales
      where ss_quantity between 11 and 15
        and (ss_list_price between 156 and 156+10
          or ss_coupon_amt between 9070 and 9070+1000
          or ss_wholesale_cost between 43 and 43+20)) B3,
     (select avg(ss_list_price) B4_LP
            ,count(ss_list_price) B4_CNT
            ,count(distinct ss_list_price) B4_CNTD
            ,array_agg(store_sales.prov) prov
      from store_sales
      where ss_quantity between 16 and 20
        and (ss_list_price between 163 and 163+10
          or ss_coupon_amt between 10835 and 10835+1000
          or ss_wholesale_cost between 35 and 35+20)) B4,
     (select avg(ss_list_price) B5_LP
            ,count(ss_list_price) B5_CNT
            ,count(distinct ss_list_price) B5_CNTD
            ,array_agg(store_sales.prov) prov
      from store_sales
      where ss_quantity between 21 and 25
        and (ss_list_price between 141 and 141+10
          or ss_coupon_amt between 8805 and 8805+1000
          or ss_wholesale_cost between 38 and 38+20)) B5,
     (select avg(ss_list_price) B6_LP
            ,count(ss_list_price) B6_CNT
            ,count(distinct ss_list_price) B6_CNTD
            ,array_agg(store_sales.prov) prov
      from store_sales
      where ss_quantity between 26 and 30
        and (ss_list_price between 55 and 55+10
          or ss_coupon_amt between 8953 and 8953+1000
          or ss_wholesale_cost between 39 and 39+20)) B6
limit 100
