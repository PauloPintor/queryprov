WITH ss AS (
    SELECT
        i_item_id,
        sum(ss_ext_sales_price) AS total_sales,
        array_agg(
            list_value(
                store_sales.prov,
                date_dim.prov,
                customer_address.prov,
                item.prov
            )
        ) AS prov
    FROM store_sales,
         date_dim,
         customer_address,
         item
    WHERE i_item_id IN (
            SELECT i_item_id
            FROM item
            WHERE i_category IN ('Music')
          )
      AND ss_item_sk = i_item_sk
      AND ss_sold_date_sk = d_date_sk
      AND d_year = 2002
      AND d_moy = 9
      AND ss_addr_sk = ca_address_sk
      AND ca_gmt_offset = -7
    GROUP BY i_item_id
),

cs AS (
    SELECT
        i_item_id,
        sum(cs_ext_sales_price) AS total_sales,
        array_agg(
            list_value(
                catalog_sales.prov,
                date_dim.prov,
                customer_address.prov,
                item.prov
            )
        ) AS prov
    FROM catalog_sales,
         date_dim,
         customer_address,
         item
    WHERE i_item_id IN (
            SELECT i_item_id
            FROM item
            WHERE i_category IN ('Music')
          )
      AND cs_item_sk = i_item_sk
      AND cs_sold_date_sk = d_date_sk
      AND d_year = 2002
      AND d_moy = 9
      AND cs_bill_addr_sk = ca_address_sk
      AND ca_gmt_offset = -7
    GROUP BY i_item_id
),

ws AS (
    SELECT
        i_item_id,
        sum(ws_ext_sales_price) AS total_sales,
        array_agg(
            list_value(
                web_sales.prov,
                date_dim.prov,
                customer_address.prov,
                item.prov
            )
        ) AS prov
    FROM web_sales,
         date_dim,
         customer_address,
         item
    WHERE i_item_id IN (
            SELECT i_item_id
            FROM item
            WHERE i_category IN ('Music')
          )
      AND ws_item_sk = i_item_sk
      AND ws_sold_date_sk = d_date_sk
      AND d_year = 2002
      AND d_moy = 9
      AND ws_bill_addr_sk = ca_address_sk
      AND ca_gmt_offset = -7
    GROUP BY i_item_id
)

SELECT
    i_item_id,
    sum(total_sales) AS total_sales,
    list_reduce(
        array_agg(prov),
        (x, y) -> list_concat(x, y)
    ) AS prov
FROM (
    SELECT *
    FROM ss

    UNION ALL

    SELECT *
    FROM cs

    UNION ALL

    SELECT *
    FROM ws
) tmp1
GROUP BY i_item_id
ORDER BY
    i_item_id,
    total_sales
LIMIT 100
