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
        and (ss_list_price between 97 and 97+10
             or ss_coupon_amt between 10374 and 10374+1000
             or ss_wholesale_cost between 44 and 44+20)) B1,
     (select avg(ss_list_price) B2_LP
            ,count(ss_list_price) B2_CNT
            ,count(distinct ss_list_price) B2_CNTD
            ,array_agg(store_sales.prov) prov
      from store_sales
      where ss_quantity between 6 and 10
        and (ss_list_price between 88 and 88+10
          or ss_coupon_amt between 8384 and 8384+1000
          or ss_wholesale_cost between 45 and 45+20)) B2,
     (select avg(ss_list_price) B3_LP
            ,count(ss_list_price) B3_CNT
            ,count(distinct ss_list_price) B3_CNTD
            ,array_agg(store_sales.prov) prov
      from store_sales
      where ss_quantity between 11 and 15
        and (ss_list_price between 102 and 102+10
          or ss_coupon_amt between 9404 and 9404+1000
          or ss_wholesale_cost between 39 and 39+20)) B3,
     (select avg(ss_list_price) B4_LP
            ,count(ss_list_price) B4_CNT
            ,count(distinct ss_list_price) B4_CNTD
            ,array_agg(store_sales.prov) prov
      from store_sales
      where ss_quantity between 16 and 20
        and (ss_list_price between 97 and 97+10
          or ss_coupon_amt between 8915 and 8915+1000
          or ss_wholesale_cost between 42 and 42+20)) B4,
     (select avg(ss_list_price) B5_LP
            ,count(ss_list_price) B5_CNT
            ,count(distinct ss_list_price) B5_CNTD
            ,array_agg(store_sales.prov) prov
      from store_sales
      where ss_quantity between 21 and 25
        and (ss_list_price between 76 and 76+10
          or ss_coupon_amt between 10297 and 10297+1000
          or ss_wholesale_cost between 36 and 36+20)) B5,
     (select avg(ss_list_price) B6_LP
            ,count(ss_list_price) B6_CNT
            ,count(distinct ss_list_price) B6_CNTD
            ,array_agg(store_sales.prov) prov
      from store_sales
      where ss_quantity between 26 and 30
        and (ss_list_price between 100 and 100+10
          or ss_coupon_amt between 7615 and 7615+1000
          or ss_wholesale_cost between 26 and 26+20)) B6
limit 100
