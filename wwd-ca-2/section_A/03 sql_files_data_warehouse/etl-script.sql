-- SECTION A: Data Warehouse Modelling
-- EXTRACT
 -- data was obtained from csv files and imported manually into oracle via TABLES>IMPORT DATA
 -- screenshots were created showing importing process.
 -- default import values were kept. Dates were corrected, 
 -- where apprarent different values were chosen:
 -- e.g duration was changed to float and dates were changed to DD-MM-YY HH24:Mi
 -- However individual sql import files were created for each csv file
 -- Manual import or sql import can be chosen whichever is quick and convenient. 
 
 
-- TRANSFORM
-- The following are changes made to csv files to ease loading in data
    -- Call Rates
        --add rates_id column with unique row numbers
            ALTER TABLE CALL_RATES_CSV add (rates_id integer);
            UPDATE CALL_RATES_CSV set rates_id = rownum;     
            
        -- change plan id TO text
            ALTER TABLE CALL_RATES_CSV ADD (PLAN_NAME VARCHAR2(26 BYTE));
            UPDATE CALL_RATES_CSV SET PLAN_NAME = 'standard' 
            WHERE PLAN_ID = 1;
             
            UPDATE CALL_RATES_CSV SET PLAN_NAME = 'off peak' 
            WHERE PLAN_ID = 2;
            
            UPDATE CALL_RATES_CSV SET PLAN_NAME = 'cosmopolitan' 
            WHERE PLAN_ID = 3;
        -- change call type id TO text  
            ALTER TABLE CALL_RATES_CSV ADD (CALL_TYPE_NAME VARCHAR2(26 BYTE));          
            UPDATE CALL_RATES_CSV SET CALL_TYPE_NAME = 'peak' 
            WHERE CALL_TYPE_ID = 1;
            
            UPDATE CALL_RATES_CSV SET CALL_TYPE_NAME = 'off-peak' 
            WHERE CALL_TYPE_ID = 2;
            
            UPDATE CALL_RATES_CSV SET CALL_TYPE_NAME = 'international' 
            WHERE CALL_TYPE_ID = 3;
            
            UPDATE CALL_RATES_CSV SET CALL_TYPE_NAME = 'roaming' 
            WHERE CALL_TYPE_ID = 4;
            
            UPDATE CALL_RATES_CSV SET CALL_TYPE_NAME = 'voice mail' 
            WHERE CALL_TYPE_ID = 5;
            
            UPDATE CALL_RATES_CSV SET CALL_TYPE_NAME = 'customer service' 
            WHERE CALL_TYPE_ID = 6;                 
    -- Calls
        --phone number
            --remove white spaces from 
            UPDATE CALLS_CSV SET phone_number = REPLACE(phone_number, ' ', ''); 
            
        -- peak/off-peak, is_international and is_roaming to call_type_id
            --needed to change id to text and to keep id
            ALTER TABLE CALLS_CSV ADD (CALL_TYPE_ID NUMBER(38,0));
            ALTER TABLE CALLS_CSV ADD (CALL_TYPE_NAME VARCHAR2(26 BYTE));
            
             --needed to work with dates
            ALTER SESSION SET nls_territory = AMERICA;
            
            -- peak calls = Calls between 9am and 6pm from Monday to Friday
            UPDATE CALLS_CSV SET CALL_TYPE_ID=1, CALL_TYPE_NAME = 'peak'
            WHERE IS_INTERNATIONAL = 'FALSE' 
                AND IS_ROAMING = 'FALSE' 
                AND (to_char(TO_DATE(CALL_TIME,'dd-MM-YY HH24:MI','NLS_DATE_LANGUAGE = American'), 'd')<6)
                OR (to_char(TO_DATE(CALL_TIME,'dd-MM-YY HH24:MI','NLS_DATE_LANGUAGE = American'), 'd')>1)
                AND (EXTRACT(HOUR FROM (TO_TIMESTAMP(CALL_TIME, 'DD-MM-YY HH24:MI'))) > 8.59) 
                AND (EXTRACT(HOUR FROM (TO_TIMESTAMP(CALL_TIME, 'DD-MM-YY HH24:MI'))) < 18.00);
                
            -- off-peak = Between 5am-8am and 7pm-11pm, or any time during the weekend             
            UPDATE CALLS_CSV SET CALL_TYPE_ID=2, CALL_TYPE_NAME = 'off-peak'
            WHERE IS_INTERNATIONAL = 'FALSE' 
                AND IS_ROAMING = 'FALSE' 
                AND (to_char(TO_DATE(CALL_TIME,'dd-MM-YY HH24:MI','NLS_DATE_LANGUAGE = American'), 'd')>5)
                OR (to_char(TO_DATE(CALL_TIME,'dd-MM-YY HH24:MI','NLS_DATE_LANGUAGE = American'), 'd')<1)
                OR (EXTRACT(HOUR FROM (TO_TIMESTAMP(CALL_TIME, 'DD-MM-YY HH24:MI'))) < 8.59) 
                AND (EXTRACT(HOUR FROM (TO_TIMESTAMP(CALL_TIME, 'DD-MM-YY HH24:MI'))) > 18.00);
            
            -- international calls
            UPDATE CALLS_CSV SET CALL_TYPE_ID=3, CALL_TYPE_NAME = 'international' 
            WHERE IS_INTERNATIONAL = 'TRUE' 
                AND IS_ROAMING = 'FALSE';

            -- roaming calls  
            UPDATE CALLS_CSV SET CALL_TYPE_ID=4, CALL_TYPE_NAME = 'roaming' 
            WHERE IS_INTERNATIONAL = 'FALSE' 
                AND IS_ROAMING = 'TRUE';
                
                
    -- Contract Plan data
        -- imported ok

    -- Customer Service Data
         --phone number
            --remove white spaces from 
            UPDATE CUSTOMER_SERVICE_CSV SET phone_number = REPLACE(phone_number, ' ', '');
        --call type
            ALTER TABLE CUSTOMER_SERVICE_CSV ADD (CALL_TYPE_NAME VARCHAR2(26 BYTE));
            UPDATE CUSTOMER_SERVICE_CSV SET CALL_TYPE_NAME = 'customer service' 
            WHERE CALL_TYPE_ID = 6;
            
    -- Customers Data
        --add customer_id column with unique row numbers
            ALTER TABLE CUSTOMERS_CSV add (customer_id integer);
            UPDATE CUSTOMERS_CSV set customer_id = rownum;       
        --phone number
            --remove white spaces from 
            UPDATE CUSTOMERS_CSV set phone_number = REPLACE(phone_number, ' ', ''); 
        --NRS
            -- change the heading
            ALTER TABLE CUSTOMERS_CSV ADD (SOCIAL_GRADE VARCHAR2(26 BYTE)); --we need to add first 
            UPDATE CUSTOMERS_CSV SET SOCIAL_GRADE = 'Upper middle class' 
            WHERE NRS = 'A';
             
            UPDATE CUSTOMERS_CSV SET SOCIAL_GRADE = 'Middle middle class' 
            WHERE NRS = 'B';
            
            UPDATE CUSTOMERS_CSV SET SOCIAL_GRADE = 'Lower middle class' 
            WHERE NRS = 'C1';
            
            UPDATE CUSTOMERS_CSV SET SOCIAL_GRADE = 'Skilled working class' 
            WHERE NRS = 'C2';
            
            UPDATE CUSTOMERS_CSV SET SOCIAL_GRADE = 'Working class' 
            WHERE NRS = 'D';
    
            UPDATE CUSTOMERS_CSV SET SOCIAL_GRADE = 'Non-working' 
            WHERE NRS = 'E';
        --DOB
            -- convert years from YY to YYYY to avoid 2000 and 1950 order list issue?
        --contract plan
            ALTER TABLE CUSTOMERS_CSV ADD (PLAN_NAME VARCHAR2(26 BYTE)); --we need to add first 
            UPDATE CUSTOMERS_CSV SET PLAN_NAME = 'standard' 
            WHERE PLAN_ID = 1;
            UPDATE CUSTOMERS_CSV SET PLAN_NAME = 'off peak' 
            WHERE PLAN_ID = 2;
            UPDATE CUSTOMERS_CSV SET PLAN_NAME = 'cosmopolitan' 
            WHERE PLAN_ID = 3;
            
        --CONTRACT_START_DATE and CONTRACT_END_DATE 
            -- 48 rows are the wrong way around -- was unable to fix
             
    -- Voicemails data
        --phone number
            --remove white spaces from 
            update VOICEMAILS_CSV set phone_number = REPLACE(phone_number, ' ', '');
        --call type
            ALTER TABLE VOICEMAILS_CSV ADD (CALL_TYPE_NAME VARCHAR2(26 BYTE)); --we need to add first 
            UPDATE VOICEMAILS_CSV SET CALL_TYPE_NAME = 'voice mail' 
            WHERE CALL_TYPE_ID = 5;

        
-- LOAD
-- generate_call_star_schema.sql required to run first
-- The following sql commands load csv data into our star schema
    -- calls dimension
        -- merge calls, customer service and voicemails into the one table using union
        INSERT INTO CALL_DIM (CALL_TIME, PHONE_NUMBER, CALL_TYPE_NAME, CALL_DURATION)
        SELECT CALL_TIME, PHONE_NUMBER, CALL_TYPE_NAME, DURATION FROM CALLS_CSV
        UNION
        SELECT CALL_TIME, PHONE_NUMBER, CALL_TYPE_NAME, DURATION FROM CUSTOMER_SERVICE_CSV
        UNION
        SELECT CALL_TIME, PHONE_NUMBER, CALL_TYPE_NAME, DURATION FROM VOICEMAILS_CSV
        ORDER BY CALL_TIME;
        
    -- Customer dimension
        -- plan_end date that are null, are assumed to mean customer is still active
        INSERT INTO CUS_DIM (DATE_OF_BIRTH, SOCIAL_GRADE_NAME, PHONE_NUMBER, PLAN_NAME, PLAN_START_DATE, PLAN_END_DATE)
        SELECT DOB, SOCIAL_GRADE, PHONE_NUMBER, PLAN_NAME, CONTRACT_START_DATE, CONTRACT_END_DATE FROM CUSTOMERS_CSV;
        
    -- Rates Dimension
        INSERT INTO RATES_DIM (CALL_TYPE, PLAN_NAME, COST_PER_MIN)
        SELECT CALL_TYPE_NAME, PLAN_NAME, COST_PER_MINUTE FROM CALL_RATES_CSV;
        
    -- Dates Dimension
        INSERT INTO DATES_DIM (DATE_TIME)
        SELECT CALL_TIME FROM CALLS_CSV
        UNION
        SELECT CALL_TIME FROM CUSTOMER_SERVICE_CSV
        UNION
        SELECT CALL_TIME FROM VOICEMAILS_CSV
        ORDER BY CALL_TIME;
 
        -- we compute date time info
        ALTER TABLE DATES_DIM
        DROP COLUMN FULL_DATE_DESCRIPTION;
        ALTER TABLE DATES_DIM
        ADD FULL_DATE_DESCRIPTION AS (to_char(TO_DATE(DATE_TIME,'dd-MM-YY HH24:MI','NLS_DATE_LANGUAGE = American'), 'FMDay, DDth FMMonthYYYY'));
        
        ALTER TABLE DATES_DIM
        DROP COLUMN DAY_OF_WEEK;
        ALTER TABLE DATES_DIM
        ADD DAY_OF_WEEK AS (to_char(TO_DATE(DATE_TIME,'dd-MM-YY HH24:MI','NLS_DATE_LANGUAGE = American'), 'd'));
        
        ALTER TABLE DATES_DIM
        DROP COLUMN MONTH;
        ALTER TABLE DATES_DIM
        ADD MONTH AS (to_char(TO_DATE(DATE_TIME,'dd-MM-YY HH24:MI','NLS_DATE_LANGUAGE = American'), 'Month'));
        
        ALTER TABLE DATES_DIM
        DROP COLUMN YEAR;
        ALTER TABLE DATES_DIM
        ADD YEAR AS (to_char(TO_DATE(DATE_TIME,'dd-MM-YY HH24:MI','NLS_DATE_LANGUAGE = American'), 'YYYY'));
        
        ALTER TABLE DATES_DIM
        DROP COLUMN WEEK_NUM;
        ALTER TABLE DATES_DIM
        ADD WEEK_NUM AS (to_char(TO_DATE(DATE_TIME,'dd-MM-YY HH24:MI','NLS_DATE_LANGUAGE = American'), 'WW'));
        
        ALTER TABLE DATES_DIM
        DROP COLUMN QUARTER;
        ALTER TABLE DATES_DIM
        ADD QUARTER AS (to_char(TO_DATE(DATE_TIME,'dd-MM-YY HH24:MI','NLS_DATE_LANGUAGE = American'), 'Q'));

    -- Call Facts (202,091 rows)
        -- insert initial facts
        INSERT INTO CALL_FACTS (DATES_DIM_DD_ID, CUS_DIM_CUSD_ID, CALL_DIM_CD_ID,  RATES_DIM_RD_ID)
        SELECT DATES_DIM.DD_ID, CUS_DIM.CUSD_ID, CALL_DIM.CD_ID, RATES_DIM.RD_ID
        FROM DATES_DIM
        LEFT JOIN CALL_DIM
        ON DATES_DIM.DATE_TIME = CALL_DIM.CALL_TIME
        LEFT JOIN CUS_DIM
        ON CALL_DIM.PHONE_NUMBER = CUS_DIM.PHONE_NUMBER
        LEFT JOIN RATES_DIM
        ON (CALL_DIM.CALL_TYPE_NAME = RATES_DIM.CALL_TYPE) AND (CUS_DIM.PLAN_NAME = RATES_DIM.PLAN_NAME);
    
        -- customer's age
            --here we create a temp table, compute the age
            -- and then add it to our fact table
        CREATE GLOBAL TEMPORARY TABLE AGE_TEMP_TABLE (
            CUS_ID NUMBER,
            AGE  NUMBER
        );
        
        INSERT INTO AGE_TEMP_TABLE
        SELECT CUS_DIM.CUSD_ID, to_number(to_char(TO_DATE(sysdate,'dd-MM-RR HH24:MI'), 'RRRR'), '9999') - (to_number(to_char(TO_DATE(DATE_OF_BIRTH,'dd-MM-RR HH24:MI'), 'RRRR'), '9999'))
        FROM CUS_DIM;
        
            --SELECT * FROM AGE_TEMP_TABLE --use to check if ok
        
        UPDATE CALL_FACTS
        SET CUSTOMER_AGE = (SELECT AGE 
            FROM AGE_TEMP_TABLE 
            WHERE CALL_FACTS.CUS_DIM_CUSD_ID = AGE_TEMP_TABLE.CUS_ID);
        
        DROP TABLE AGE_TEMP_TABLE;

        -- customer's social grade
        UPDATE CALL_FACTS
        SET CUSTOMER_SOCIAL_GRADE = (SELECT SOCIAL_GRADE_NAME 
            FROM CUS_DIM 
            WHERE CALL_FACTS.CUS_DIM_CUSD_ID = CUS_DIM.CUSD_ID);
            
        -- customer's contract plan
        UPDATE CALL_FACTS
        SET CUSTOMER_CONTRACT_PLAN = (SELECT PLAN_NAME 
            FROM CUS_DIM 
            WHERE CALL_FACTS.CUS_DIM_CUSD_ID = CUS_DIM.CUSD_ID);
            
        -- days on plan 
            --(how many days customer was on plan b4 they quit
        CREATE GLOBAL TEMPORARY TABLE PDUR_TEMP_TABLE (
            CUS_ID NUMBER,
            PLAN_DURATION  NUMBER
        );
        
        INSERT INTO PDUR_TEMP_TABLE
        SELECT CUS_DIM.CUSD_ID, (TO_DATE(PLAN_END_DATE,'dd-MM-RR HH24:MI')) - (TO_DATE(PLAN_START_DATE,'dd-MM-RR HH24:MI')) AS PLAN_DURATION
        FROM CUS_DIM;
        
        SELECT COUNT(PLAN_DURATION) FROM PDUR_TEMP_TABLE WHERE PLAN_DURATION <0;
        --48 rows have start/end date swapped wrong
        
       -- UPDATE CALL_FACTS
        SET DAYS_ON_PLAN = (SELECT PLAN_DURATION
            FROM PDUR_TEMP_TABLE 
            WHERE CALL_FACTS.CUS_DIM_CUSD_ID = PDUR_TEMP_TABLE.CUS_ID);  
        
        DROP TABLE PDUR_TEMP_TABLE;
     
        --to fix wrong dates, the following was tried. but did not work
        --CREATE GLOBAL TEMPORARY TABLE PDUR_TEMP_TABLE (
        --    CUS_ID NUMBER,
        --    PLAN_DURATION  NUMBER
        --);
        
        --INSERT INTO PDUR_TEMP_TABLE
        --SELECT CUS_DIM.CUSD_ID, (TO_DATE(PLAN_START_DATE,'dd-MM-RR HH24:MI')) - (TO_DATE(PLAN_END_DATE,'dd-MM-RR HH24:MI')) AS PLAN_DURATION
        --FROM CUS_DIM
        --WHERE ((TO_DATE(PLAN_START_DATE,'dd-MM-RR HH24:MI')) - (TO_DATE(PLAN_END_DATE,'dd-MM-RR HH24:MI')))>1;    

        UPDATE CALL_FACTS
        SET DAYS_ON_PLAN = (SELECT PLAN_DURATION
            FROM PDUR_TEMP_TABLE 
            WHERE (CALL_FACTS.CUS_DIM_CUSD_ID = PDUR_TEMP_TABLE.CUS_ID) AND (DAYS_ON_PLAN<0));  
        
        DROP TABLE PDUR_TEMP_TABLE;  
        --call_revenue_generated
        CREATE GLOBAL TEMPORARY TABLE REV_TEMP_TABLE (
            FACT_ID NUMBER,
            CALL_COST  FLOAT
        );
        
        --we assume calls are in secs and cost of call is in minutes
        -- so we convert duration into minutes and work out call cost
        INSERT INTO REV_TEMP_TABLE
        SELECT CALL_FACTS.CF_ID, ((CALL_DIM.CALL_DURATION/60)*RATES_DIM.COST_PER_MIN) as CALL_COST
        FROM CALL_FACTS
        LEFT JOIN CALL_DIM
        ON CALL_FACTS.CALL_DIM_CD_ID = CALL_DIM.CD_ID
        LEFT JOIN RATES_DIM
        ON CALL_FACTS.RATES_DIM_RD_ID = RATES_DIM.RD_ID;
        
        --SELECT sum(CALL_COST) FROM REV_TEMP_TABLE
        --SELECT FACT_ID FROM REV_TEMP_TABLE WHERE CALL_COST=2594.88959559;
        
        UPDATE CALL_FACTS
        SET CALL_REVENUE_GENERATED = (SELECT CALL_COST 
            FROM REV_TEMP_TABLE 
            WHERE CALL_FACTS.CF_ID = REV_TEMP_TABLE.FACT_ID);
            
        DROP TABLE REV_TEMP_TABLE;
        
        --cus_service_time
            -- customer by max duration of call type "customer service"
        
        CREATE GLOBAL TEMPORARY TABLE CS_TEMP_TABLE (
            FACT_ID NUMBER,
            CS_TIME  FLOAT
        );        
        
        INSERT INTO CS_TEMP_TABLE
        SELECT CALL_FACTS.CF_ID, CALL_DIM.CALL_DURATION
        FROM CALL_FACTS
        LEFT JOIN CALL_DIM
        ON CALL_FACTS.CALL_DIM_CD_ID = CALL_DIM.CD_ID
        WHERE CALL_DIM.CALL_TYPE_NAME = 'customer service';
        
        -- SELECT * FROM CS_TEMP_TABLE

        UPDATE CALL_FACTS
        SET CUS_SERVICE_TIME = (SELECT CS_TIME 
            FROM CS_TEMP_TABLE 
            WHERE CALL_FACTS.CF_ID = CS_TEMP_TABLE.FACT_ID);
        
        DROP TABLE CS_TEMP_TABLE;
        
         --check if end date is null and set active to true, else to false  
        UPDATE CALL_FACTS
        SET CUSTOMER_ACTIVE = 1
        WHERE DAYS_ON_PLAN IS NULL;
        
        UPDATE CALL_FACTS
        SET CUSTOMER_ACTIVE = 0
        WHERE DAYS_ON_PLAN IS NOT NULL;
        
        -- CHECK FACT TABLE
        SELECT * FROM CALL_FACTS WHERE CF_ID=10319;