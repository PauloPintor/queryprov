select B1.B1_LP, B1.B1_CNT, B1.B1_CNTD, B2.B2_LP, B2.B2_CNT, B2.B2_CNTD, B3.B3_LP, B3.B3_CNT, B3.B3_CNTD, B4.B4_LP, B4.B4_CNT, B4.B4_CNTD, B5.B5_LP, B5.B5_CNT, B5.B5_CNTD, B6.B6_LP, B6.B6_CNT, B6.B6_CNTD,
       list_value(B1.prov, B2.prov, B3.prov, B4.prov, B5.prov, B6.prov) as prov
from

     (select avg(ss_list_price) B1_LP
            ,count(ss_list_price) B1_CNT
            ,count(distinct ss_list_price) B1_CNTD
            ,array_agg(list_value(store_sales.prov, item.prov)) prov
      from store_sales, item

      where store_sales.ss_item_sk = item.i_item_sk
        and ss_quantity between 0 and 5
        and (ss_list_price between 104 and 104+10 or ss_coupon_amt between 7744 and 7744+1000 or ss_wholesale_cost between 38 and 38+20)
) B1
,
     (select avg(ss_list_price) B2_LP
            ,count(ss_list_price) B2_CNT
            ,count(distinct ss_list_price) B2_CNTD
            ,array_agg(list_value(store_sales.prov, item.prov)) prov
      from store_sales, item

      where store_sales.ss_item_sk = item.i_item_sk
        and ss_quantity between 6 and 10
        and (ss_list_price between 82 and 82+10 or ss_coupon_amt between 8990 and 8990+1000 or ss_wholesale_cost between 37 and 37+20)
) B2
,
     (select avg(ss_list_price) B3_LP
            ,count(ss_list_price) B3_CNT
            ,count(distinct ss_list_price) B3_CNTD
            ,array_agg(list_value(store_sales.prov, item.prov)) prov
      from store_sales, item

      where store_sales.ss_item_sk = item.i_item_sk
        and ss_quantity between 11 and 15
        and (ss_list_price between 76 and 76+10 or ss_coupon_amt between 7658 and 7658+1000 or ss_wholesale_cost between 41 and 41+20)
) B3
,
     (select avg(ss_list_price) B4_LP
            ,count(ss_list_price) B4_CNT
            ,count(distinct ss_list_price) B4_CNTD
            ,array_agg(list_value(store_sales.prov, item.prov)) prov
      from store_sales, item

      where store_sales.ss_item_sk = item.i_item_sk
        and ss_quantity between 16 and 20
        and (ss_list_price between 110 and 110+10 or ss_coupon_amt between 8589 and 8589+1000 or ss_wholesale_cost between 47 and 47+20)
) B4
,
     (select avg(ss_list_price) B5_LP
            ,count(ss_list_price) B5_CNT
            ,count(distinct ss_list_price) B5_CNTD
            ,array_agg(list_value(store_sales.prov, item.prov)) prov
      from store_sales, item

      where store_sales.ss_item_sk = item.i_item_sk
        and ss_quantity between 21 and 25
        and (ss_list_price between 112 and 112+10 or ss_coupon_amt between 10179 and 10179+1000 or ss_wholesale_cost between 34 and 34+20)
) B5
,
     (select avg(ss_list_price) B6_LP
            ,count(ss_list_price) B6_CNT
            ,count(distinct ss_list_price) B6_CNTD
            ,array_agg(list_value(store_sales.prov, item.prov)) prov
      from store_sales, item

      where store_sales.ss_item_sk = item.i_item_sk
        and ss_quantity between 26 and 30
        and (ss_list_price between 88 and 88+10 or ss_coupon_amt between 9370 and 9370+1000 or ss_wholesale_cost between 41 and 41+20)
) B6

limit 100
