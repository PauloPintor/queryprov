WITH 
ss AS (

    SELECT
        i.i_item_id,
        sum(s.ss_ext_sales_price) AS total_sales,
        array_agg(
            list_value(
                s.prov, d.prov, ca.prov, i.prov
            )
        ) AS prov

    FROM store_sales s,
         date_dim d,
         customer_address ca,
         item i

    WHERE i.i_item_id IN (
            SELECT i_item_id
            FROM item
            WHERE i_category IN ('Men')
          )
      AND s.ss_item_sk = i.i_item_sk
      AND s.ss_sold_date_sk = d.d_date_sk
      AND d.d_year = 2002
      AND s.ss_addr_sk = ca.ca_address_sk
      AND ca.ca_gmt_offset = -5
    GROUP BY i.i_item_id
)
,

cs AS (

    SELECT
        i.i_item_id,
        sum(s.cs_ext_sales_price) AS total_sales,
        array_agg(
            list_value(
                s.prov, d.prov, ca.prov, i.prov
            )
        ) AS prov

    FROM catalog_sales s,
         date_dim d,
         customer_address ca,
         item i

    WHERE i.i_item_id IN (
            SELECT i_item_id
            FROM item
            WHERE i_category IN ('Men')
          )
      AND s.cs_item_sk = i.i_item_sk
      AND s.cs_sold_date_sk = d.d_date_sk
      AND d.d_year = 2002
      AND s.cs_bill_addr_sk = ca.ca_address_sk
      AND ca.ca_gmt_offset = -5
    GROUP BY i.i_item_id
)
,

ws AS (

    SELECT
        i.i_item_id,
        sum(s.ws_ext_sales_price) AS total_sales,
        array_agg(
            list_value(
                s.prov, d.prov, ca.prov, i.prov
            )
        ) AS prov

    FROM web_sales s,
         date_dim d,
         customer_address ca,
         item i

    WHERE i.i_item_id IN (
            SELECT i_item_id
            FROM item
            WHERE i_category IN ('Men')
          )
      AND s.ws_item_sk = i.i_item_sk
      AND s.ws_sold_date_sk = d.d_date_sk
      AND d.d_year = 2002
      AND s.ws_bill_addr_sk = ca.ca_address_sk
      AND ca.ca_gmt_offset = -5
    GROUP BY i.i_item_id
)


SELECT
    i_item_id,
    sum(total_sales) AS total_sales,
    list_reduce(
        array_agg(prov),
        (x, y) -> list_concat(x, y)
    ) AS prov
FROM (
    SELECT * FROM ss

    UNION ALL

    SELECT * FROM cs

    UNION ALL

    SELECT * FROM ws
) tmp1
GROUP BY i_item_id
ORDER BY
    i_item_id,
    total_sales
LIMIT 100
