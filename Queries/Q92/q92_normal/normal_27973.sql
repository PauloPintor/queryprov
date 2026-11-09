WITH avg_discount AS (
    SELECT
        ws_avg.ws_item_sk AS item_sk,
        1.3 * avg(ws_avg.ws_ext_discount_amt) AS discount_threshold,
        array_agg(
            list_value(
                ws_avg.prov,
                d_avg.prov
            )
        ) AS avg_prov
    FROM web_sales ws_avg
    JOIN date_dim d_avg
      ON d_avg.d_date_sk = ws_avg.ws_sold_date_sk
    WHERE d_avg.d_date BETWEEN cast('2002-02-19' AS date)
                           AND cast('2002-02-19' AS date) + INTERVAL 90 DAY
    GROUP BY
        ws_avg.ws_item_sk
),

qualified AS (
    SELECT
        ws.ws_ext_discount_amt,
        list_value(
            ws.prov,
            item.prov,
            d.prov
        ) AS outer_prov,
        ad.avg_prov AS avg_prov
    FROM web_sales ws
    JOIN item
      ON item.i_item_sk = ws.ws_item_sk
    JOIN date_dim d
      ON d.d_date_sk = ws.ws_sold_date_sk
    JOIN avg_discount ad
      ON ad.item_sk = item.i_item_sk
    WHERE item.i_manufact_id = 530
      AND d.d_date BETWEEN cast('2002-02-19' AS date)
                       AND cast('2002-02-19' AS date) + INTERVAL 90 DAY
      AND ws.ws_ext_discount_amt > ad.discount_threshold
)

SELECT
    sum(ws_ext_discount_amt) AS "Excess Discount Amount",

    -- Factorised provenance:
    -- each element is [outer witness factor, avg witness factor]
    array_agg(
        list_value(
            list_value(outer_prov),
            avg_prov
        )
    ) AS prov

FROM qualified
ORDER BY
    sum(ws_ext_discount_amt)
LIMIT 100
