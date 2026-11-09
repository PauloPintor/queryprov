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
        and (ss_list_price between 149 and 149+10
             or ss_coupon_amt between 2724 and 2724+1000
             or ss_wholesale_cost between 58 and 58+20)) B1,
     (select avg(ss_list_price) B2_LP
            ,count(ss_list_price) B2_CNT
            ,count(distinct ss_list_price) B2_CNTD
            ,array_agg(store_sales.prov) prov
      from store_sales
      where ss_quantity between 6 and 10
        and (ss_list_price between 25 and 25+10
          or ss_coupon_amt between 15035 and 15035+1000
          or ss_wholesale_cost between 57 and 57+20)) B2,
     (select avg(ss_list_price) B3_LP
            ,count(ss_list_price) B3_CNT
            ,count(distinct ss_list_price) B3_CNTD
            ,array_agg(store_sales.prov) prov
      from store_sales
      where ss_quantity between 11 and 15
        and (ss_list_price between 26 and 26+10
          or ss_coupon_amt between 12117 and 12117+1000
          or ss_wholesale_cost between 19 and 19+20)) B3,
     (select avg(ss_list_price) B4_LP
            ,count(ss_list_price) B4_CNT
            ,count(distinct ss_list_price) B4_CNTD
            ,array_agg(store_sales.prov) prov
      from store_sales
      where ss_quantity between 16 and 20
        and (ss_list_price between 174 and 174+10
          or ss_coupon_amt between 5919 and 5919+1000
          or ss_wholesale_cost between 16 and 16+20)) B4,
     (select avg(ss_list_price) B5_LP
            ,count(ss_list_price) B5_CNT
            ,count(distinct ss_list_price) B5_CNTD
            ,array_agg(store_sales.prov) prov
      from store_sales
      where ss_quantity between 21 and 25
        and (ss_list_price between 53 and 53+10
          or ss_coupon_amt between 16021 and 16021+1000
          or ss_wholesale_cost between 25 and 25+20)) B5,
     (select avg(ss_list_price) B6_LP
            ,count(ss_list_price) B6_CNT
            ,count(distinct ss_list_price) B6_CNTD
            ,array_agg(store_sales.prov) prov
      from store_sales
      where ss_quantity between 26 and 30
        and (ss_list_price between 58 and 58+10
          or ss_coupon_amt between 4294 and 4294+1000
          or ss_wholesale_cost between 67 and 67+20)) B6
limit 100
