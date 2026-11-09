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
        and (ss_list_price between 95 and 95+10
             or ss_coupon_amt between 8224 and 8224+1000
             or ss_wholesale_cost between 35 and 35+20)) B1,
     (select avg(ss_list_price) B2_LP
            ,count(ss_list_price) B2_CNT
            ,count(distinct ss_list_price) B2_CNTD
            ,array_agg(store_sales.prov) prov
      from store_sales
      where ss_quantity between 6 and 10
        and (ss_list_price between 91 and 91+10
          or ss_coupon_amt between 9042 and 9042+1000
          or ss_wholesale_cost between 38 and 38+20)) B2,
     (select avg(ss_list_price) B3_LP
            ,count(ss_list_price) B3_CNT
            ,count(distinct ss_list_price) B3_CNTD
            ,array_agg(store_sales.prov) prov
      from store_sales
      where ss_quantity between 11 and 15
        and (ss_list_price between 69 and 69+10
          or ss_coupon_amt between 9123 and 9123+1000
          or ss_wholesale_cost between 48 and 48+20)) B3,
     (select avg(ss_list_price) B4_LP
            ,count(ss_list_price) B4_CNT
            ,count(distinct ss_list_price) B4_CNTD
            ,array_agg(store_sales.prov) prov
      from store_sales
      where ss_quantity between 16 and 20
        and (ss_list_price between 88 and 88+10
          or ss_coupon_amt between 9715 and 9715+1000
          or ss_wholesale_cost between 35 and 35+20)) B4,
     (select avg(ss_list_price) B5_LP
            ,count(ss_list_price) B5_CNT
            ,count(distinct ss_list_price) B5_CNTD
            ,array_agg(store_sales.prov) prov
      from store_sales
      where ss_quantity between 21 and 25
        and (ss_list_price between 99 and 99+10
          or ss_coupon_amt between 8358 and 8358+1000
          or ss_wholesale_cost between 37 and 37+20)) B5,
     (select avg(ss_list_price) B6_LP
            ,count(ss_list_price) B6_CNT
            ,count(distinct ss_list_price) B6_CNTD
            ,array_agg(store_sales.prov) prov
      from store_sales
      where ss_quantity between 26 and 30
        and (ss_list_price between 94 and 94+10
          or ss_coupon_amt between 9874 and 9874+1000
          or ss_wholesale_cost between 34 and 34+20)) B6
limit 100
