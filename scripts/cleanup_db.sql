-- =============================================================================
-- Shimmi DB Cleanup Script
-- Run on the server:  sqlite3 /opt/shimmi/data/shimmi.sqlite < cleanup_db.sql
-- ALWAYS take a backup first:  cp shimmi.sqlite shimmi.sqlite.bak
-- =============================================================================

-- Show current state before cleanup
SELECT '=== BEFORE CLEANUP ===' AS step;
SELECT COUNT(*) AS total_facts FROM user_memory;

-- ---------------------------------------------------------------------------
-- STEP 1: Remove dangerous / sensitive data
-- These should never have been stored.
-- ---------------------------------------------------------------------------
SELECT '--- Step 1: Remove sensitive data ---' AS step;

DELETE FROM user_memory WHERE fact_key = 'social_security_number';
SELECT changes() AS ssn_rows_deleted;

DELETE FROM user_memory WHERE fact_key IN ('bank_account', 'bank_account_number',
    'credit_card', 'credit_card_number', 'password', 'pin', 'otp',
    'passport_number', 'aadhaar', 'aadhaar_number', 'pan', 'pan_number');
SELECT changes() AS other_sensitive_rows_deleted;

-- ---------------------------------------------------------------------------
-- STEP 2: Fix SMS flood pollution
-- 'TEJA EDUCATIONAL SOCIETY' was ingested from an SMS and saved as city
-- AND education. Both are wrong.
-- ---------------------------------------------------------------------------
SELECT '--- Step 2: Fix SMS flood pollution ---' AS step;

DELETE FROM user_memory
WHERE fact_key = 'city'
  AND fact_value = 'TEJA EDUCATIONAL SOCIETY';
SELECT changes() AS bad_city_rows_deleted;

DELETE FROM user_memory
WHERE fact_key = 'education'
  AND fact_value = 'TEJA EDUCATIONAL SOCIETY';
SELECT changes() AS bad_education_rows_deleted;

-- ---------------------------------------------------------------------------
-- STEP 3: Remove ephemeral / session noise keys
-- These have no long-term value and burn tokens on every prompt.
-- ---------------------------------------------------------------------------
SELECT '--- Step 3: Remove session noise ---' AS step;

DELETE FROM user_memory WHERE fact_key IN (
    'greeting',
    'arrival_time',
    'destination',
    'next_meeting_team',
    'next_meeting_time',
    'next_trip_start_date',
    'recent_query',
    'recent_search',
    'conversation_since_morning',
    'lists',
    'favorite_video',
    'recent_article',
    'work_experience'
);
SELECT changes() AS noise_rows_deleted;

-- ---------------------------------------------------------------------------
-- STEP 4: Remove verbose text blob keys
-- These are 200-800 char LLM-generated descriptions, not facts.
-- ---------------------------------------------------------------------------
SELECT '--- Step 4: Remove verbose blobs ---' AS step;

DELETE FROM user_memory WHERE fact_key IN (
    'recent_article_details',
    'favorite_news_source_details',
    'last_summary',
    'conversation_summary'    -- will be regenerated fresh; current one is stale/inconsistent
);
SELECT changes() AS blob_rows_deleted;

-- ---------------------------------------------------------------------------
-- STEP 5: Fix the name inconsistency
-- facts say 'Phani Adabala' but conversation_summary says 'Pranati Naidu Adabala'.
-- The user confirmed their name is Phani — keep that, remove the wrong one.
-- ---------------------------------------------------------------------------
SELECT '--- Step 5: Verify name ---' AS step;

SELECT fact_key, fact_value, source FROM user_memory
WHERE fact_key = 'name';

-- Only delete if the value is the wrong name
-- (adjust if the correct name stored differs)
DELETE FROM user_memory
WHERE fact_key = 'name'
  AND fact_value LIKE '%Pranati%';
SELECT changes() AS wrong_name_rows_deleted;

-- ---------------------------------------------------------------------------
-- STEP 6: Remove consolidation corruption
-- shopping_list was absorbed into recent_book — recent_book now has grocery
-- values. Fix: if recent_book looks like a grocery list, delete it.
-- ---------------------------------------------------------------------------
SELECT '--- Step 6: Fix consolidation corruption ---' AS step;

-- Show current state of these keys for inspection
SELECT fact_key, fact_value FROM user_memory
WHERE fact_key IN ('recent_book', 'shopping_list', 'grocery_list', 'lists')
ORDER BY fact_key;

-- Delete recent_book only if it contains grocery/shopping content
-- (safe: it will be re-populated when user next mentions a book)
DELETE FROM user_memory
WHERE fact_key = 'recent_book'
  AND (fact_value LIKE '%milk%'
    OR fact_value LIKE '%bread%'
    OR fact_value LIKE '%eggs%'
    OR fact_value LIKE '%shopping%'
    OR fact_value LIKE '%grocery%');
SELECT changes() AS corrupted_recent_book_deleted;

-- ---------------------------------------------------------------------------
-- STEP 7: Show final state
-- ---------------------------------------------------------------------------
SELECT '=== AFTER CLEANUP ===' AS step;
SELECT COUNT(*) AS total_facts_remaining FROM user_memory;

SELECT '--- Remaining facts ---' AS step;
SELECT fact_key, substr(fact_value, 1, 60) AS value_preview, source
FROM user_memory
ORDER BY fact_key;

-- ---------------------------------------------------------------------------
-- STEP 8: Fix consolidated portfolio corruption (post v3.15.1 DB residue)
-- Consolidation merged per-stock keys into meaningless single keys.
-- portfolio_purchase_price = '2150' (PAYTM price, ACMESOLAR's 289 was overwritten)
-- portfolio_quantity = '89' (combined shares, meaningless)
-- Delete these wrong merged keys. portfolio_holdings JSON is the correct source.
-- ---------------------------------------------------------------------------
SELECT '--- Step 8: Fix portfolio consolidation corruption ---' AS step;

DELETE FROM user_memory WHERE fact_key IN (
    'portfolio_purchase_price',
    'portfolio_quantity',
    'portfolio_status',
    'portfolio_review',
    'portfolio_summary',
    'stock_paytm_price',
    'stock_paytm_quantity',
    'stock_acmesolar_price',
    'stock_acmesolar_quantity',
    'portfolio_stocks_paytm',
    'portfolio_stocks_acmesolar',
    'portfolio_purchase_price_paytm',
    'portfolio_purchase_price_acmesolar',
    'favorite_stock'
);
SELECT changes() AS portfolio_noise_rows_deleted;

-- Show current portfolio_holdings to verify it's intact
SELECT '--- Current portfolio_holdings ---' AS step;
SELECT fact_key, fact_value FROM user_memory
WHERE fact_key = 'portfolio_holdings';
