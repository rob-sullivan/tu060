-- Section C: Machine Learning using SQL

-- CREATING CASE TABLE
    --DATA CLEANING: Fixed null dates and negative dates
    --DATA ENCODING: Changed labels to binary values (instead of letting oracle do it
    CREATE OR REPLACE VIEW CUSTOMER_CHURN_CASE_TABLE_VIEW AS
    SELECT 
        CALL_FACTS.CF_ID AS CASE_ID, 
        CALL_FACTS.CUS_DIM_CUSD_ID,
        CALL_FACTS.CUSTOMER_AGE,
        (LOWER(REPLACE(CUS_DIM.SOCIAL_GRADE_NAME, ' ', ''))) AS SOCIAL_GRADE,
        CALL_FACTS.CUSTOMER_CONTRACT_PLAN AS CONTRACT_PLAN,
        (CASE
             WHEN CALL_FACTS.DAYS_ON_PLAN IS NOT NULL THEN 
                (CASE 
                    WHEN CALL_FACTS.DAYS_ON_PLAN<0 THEN 
                        (SELECT (TO_DATE(PLAN_START_DATE,'dd-MM-RR HH24:MI')) - (TO_DATE(PLAN_END_DATE,'dd-MM-RR HH24:MI'))
                        FROM CUS_DIM WHERE CUS_DIM.CUSD_ID = CALL_FACTS.CUS_DIM_CUSD_ID)
                    ELSE CALL_FACTS.DAYS_ON_PLAN
                END)
             ELSE 
                (SELECT (TO_DATE(SYSDATE,'dd-MM-RR HH24:MI')) - (TO_DATE(PLAN_START_DATE,'dd-MM-RR HH24:MI'))
                FROM CUS_DIM WHERE CUS_DIM.CUSD_ID = CALL_FACTS.CUS_DIM_CUSD_ID)
        END) AS PLAN_DURATION,
        CALL_DIM.CALL_TYPE_NAME AS CALL_TYPE,
        CALL_DIM.CALL_DURATION,
        CALL_REVENUE_GENERATED,
        CALL_FACTS.CUSTOMER_ACTIVE
    FROM CALL_FACTS
    LEFT JOIN DATES_DIM
    ON CALL_FACTS.DATES_DIM_DD_ID = DATES_DIM.DD_ID
    LEFT JOIN CUS_DIM
    ON CALL_FACTS.CUS_DIM_CUSD_ID = CUS_DIM.CUSD_ID
    LEFT JOIN CALL_DIM
    ON CALL_FACTS.CALL_DIM_CD_ID = CALL_DIM.CD_ID;
    
    SELECT * FROM CUSTOMER_CHURN_CASE_TABLE_VIEW ORDER BY CASE_ID DESC;

-- CREATING TRAINING AND TESTING TABLES

    --TEST SET
    DROP TABLE CUSTOMER_CHURN_TEST_SET;
    CREATE TABLE CUSTOMER_CHURN_TEST_SET
    AS SELECT * FROM CUSTOMER_CHURN_CASE_TABLE_VIEW
    WHERE CASE_ID = -1;
    
    --APPLY SET
    DROP TABLE CUSTOMER_CHURN_APPLY_SET;
    CREATE TABLE CUSTOMER_CHURN_APPLY_SET
    AS SELECT * FROM CUSTOMER_CHURN_CASE_TABLE_VIEW
    WHERE CASE_ID = -1;

    --split data into training and testing data (80:20 rule)
    SELECT COUNT(CASE_ID) FROM CUSTOMER_CHURN_CASE_TABLE_VIEW;
    --we have 202091 rows so 80% = 161,673 and 20% = 40,418
    
    --INSERT 80% OF DATA INTO TEST SET (161,673 rows inserted.)
    INSERT INTO CUSTOMER_CHURN_TEST_SET
    SELECT * FROM CUSTOMER_CHURN_CASE_TABLE_VIEW
    WHERE CASE_ID< 161674;
    
    --INSERT 40% OF DATA INTO APPLY SET (40,418 rows inserted.)
    INSERT INTO CUSTOMER_CHURN_APPLY_SET
    SELECT * FROM CUSTOMER_CHURN_CASE_TABLE_VIEW
    WHERE CASE_ID> 161673; 
    
-- CREATE MODEL 1: DECISION TREE MODEL
    -- we create the settings table first
    DROP TABLE DECISION_TREE_MODEL_SETTINGS;
    CREATE TABLE DECISION_TREE_MODEL_SETTINGS (
    SETTING_NAME VARCHAR2(30),
    SETTING_VALUE VARCHAR2(30));
    -- Populate the settings table
    -- Specify DT. By default, Naive Bayes is used for classification.
    -- Specify ADP. By default, ADP is not used.
    
    BEGIN
       INSERT INTO DECISION_TREE_MODEL_SETTINGS (SETTING_NAME, SETTING_VALUE)
       VALUES (DBMS_DATA_MINING.ALGO_NAME, DBMS_DATA_MINING.ALGO_DECISION_TREE);
       
       INSERT INTO DECISION_TREE_MODEL_SETTINGS (SETTING_NAME, SETTING_VALUE)
       VALUES (DBMS_DATA_MINING.PREP_AUTO,DBMS_DATA_MINING.PREP_AUTO_ON);
       
       COMMIT;
    END;

    -- we give user modell building privileges, to avoid any issues.
    --login as SYS and run the following commands
    --GRANT CREATE MINING MODEL TO "C##ROB";
    --GRANT SELECT ANY MINING MODEL TO "C##ROB";
    --GRANT ALTER ANY MINING MODEL TO "C##ROB";
    --GRANT DROP ANY MINING MODEL TO "C##ROB";
    --GRANT COMMENT ANY MINING MODEL TO "C##ROB";
    
    BEGIN
        DBMS_DATA_MINING.CREATE_MODEL(
           MODEL_NAME => 'CUSTOMER_CHURN_DECISION_TREE_MODEL',
           MINING_FUNCTION => DBMS_DATA_MINING.CLASSIFICATION,
           DATA_TABLE_NAME => 'CUSTOMER_CHURN_CASE_TABLE_VIEW',
           CASE_ID_COLUMN_NAME => 'CASE_ID',
           TARGET_COLUMN_NAME => 'CUSTOMER_ACTIVE',
           SETTINGS_TABLE_NAME => 'DECISION_TREE_MODEL_SETTINGS');
    END;

    -- explore model meta data
    DESCRIBE USER_MINING_MODEL_SETTINGS
    
    --describe the model settings tables
    SELECT MODEL_NAME,
       MINING_FUNCTION,
       ALGORITHM,
       BUILD_DURATION,
       MODEL_SIZE
    FROM USER_MINING_MODELS;
    
    --check that model was created
    SELECT SETTING_NAME,
    SETTING_VALUE,
    SETTING_TYPE
    FROM USER_MINING_MODEL_SETTINGS
    WHERE MODEL_NAME in 'CUSTOMER_CHURN_DECISION_TREE_MODEL';

    --check algorithm settings
    SELECT ATTRIBUTE_NAME,
       ATTRIBUTE_TYPE,
       USAGE_TYPE, 
       TARGET
    FROM ALL_MINING_MODEL_ATTRIBUTES
    WHERE MODEL_NAME = 'CUSTOMER_CHURN_DECISION_TREE_MODEL';
    
-- TRAIN MODEL 1: DECISION TREE MODEL
    --Confusion Matrix
    CREATE OR REPLACE VIEW CUSTOMER_CHURN_DT_TEST_RESULTS
    AS
    SELECT CASE_ID, 
        PREDICTION(CUSTOMER_CHURN_DECISION_TREE_MODEL USING *) predicted_value, 
        PREDICTION_PROBABILITY(CUSTOMER_CHURN_DECISION_TREE_MODEL USING *) probability
    FROM CUSTOMER_CHURN_TEST_SET;
    
    SELECT * FROM CUSTOMER_CHURN_DT_TEST_RESULTS;
    
    
    DROP TABLE CUSTOMER_CHURN_DT_CONFUSION_MATRIX;
    DECLARE
       v_accuracy NUMBER;
    BEGIN
    DBMS_DATA_MINING.COMPUTE_CONFUSION_MATRIX (
       accuracy => v_accuracy,
       apply_result_table_name => 'DEMO_CLASS_DT_TEST_RESULTS',
       target_table_name => 'CUSTOMER_CHURN_TEST_SET',
       case_id_column_name => 'CASE_ID',
       target_column_name => 'CUSTOMER_ACTIVE',
       confusion_matrix_table_name => 'CUSTOMER_CHURN_DT_CONFUSION_MATRIX',
       score_column_name => 'PREDICTED_VALUE',
       score_criterion_column_name => 'PROBABILITY',
       cost_matrix_table_name => null,
       apply_result_schema_name => null,
       target_schema_name => null,
       cost_matrix_schema_name => null,
       score_criterion_type => 'PROBABILITY');
       DBMS_OUTPUT.PUT_LINE('**** MODEL ACCURACY ****: ' || ROUND(v_accuracy,4));
    END;
    
    SELECT * FROM CUSTOMER_CHURN_DT_CONFUSION_MATRIX;
    

-- TEST MODEL 1: DECISION TREE MODEL
    -- apply model to new data
    BEGIN
       dbms_data_mining.apply(
       model_name => 'CUSTOMER_CHURN_DECISION_TREE_MODEL',
       data_table_name => 'CUSTOMER_CHURN_APPLY_SET',
       case_id_column_name => 'CASE_ID',
       result_table_name => 'NEW_DATA_SCORED');
    END;
    
    --Inspect the results in the NEW_DATA_SCORED table.
    SELECT * FROM NEW_DATA_SCORED;
    
    -- apply model to real time data
    SELECT CASE_ID, PREDICTION(CUSTOMER_CHURN_DECISION_TREE_MODEL using *)
    FROM CUSTOMER_CHURN_APPLY_SET;
    
    
    SELECT CASE_ID, PREDICTION(CUSTOMER_CHURN_DECISION_TREE_MODEL using *), PREDICTION_PROBABILITY(CUSTOMER_CHURN_DECISION_TREE_MODEL using *)
    FROM CUSTOMER_CHURN_APPLY_SET
    WHERE rownum <=5;
    
    SELECT CASE_ID
    FROM CUSTOMER_CHURN_APPLY_SET
    WHERE PREDICTION(CUSTOMER_CHURN_DECISION_TREE_MODEL using *) = 1
    AND rownum <=10;
    
-- PREDICT WITH MODEL 1: DECISION TREE MODEL
    --what if analysis on dummy data
    Select PREDICTION_PROBABILITY (CUSTOMER_CHURN_DECISION_TREE_MODEL, 0
    USING 
    46 as CUSTOMER_AGE,
    'non-working' as SOCIAL_GRADE,
    'cosmopolitan' as CONTRACT_PLAN,
    432 as PLAN_DURATION,
    'off-peak' as CALL_TYPE,
    561.37 as CALL_DURATION,
    0.84 AS CALL_REVENUE_GENERATED) Pred_Prob
    from dual;
    
-- CREATE MODEL 2: NEURAL NETWORK MODEL
    -- we create the settings table first
    DROP TABLE NEURAL_NETWORK_MODEL_SETTINGS;
    CREATE TABLE NEURAL_NETWORK_MODEL_SETTINGS (
    SETTING_NAME VARCHAR2(30),
    SETTING_VALUE VARCHAR2(30));
    -- Populate the settings table
    -- Specify NN. uses ALGO_NEURAL_NETWORK 
    -- ref: https://developer.oracle.com/databases/neural-network-machine-learning.html
    -- ref: https://docs.oracle.com/en/database/oracle/oracle-database/19/dmcon/neural-network.html#GUID-08B8B820-078E-440B-93E7-A3F2ADC30054
    
    BEGIN
        INSERT INTO NEURAL_NETWORK_MODEL_SETTINGS (SETTING_NAME, SETTING_VALUE)
        VALUES ('ALGO_NAME', 'ALGO_NEURAL_NETWORK');
        
        INSERT INTO NEURAL_NETWORK_MODEL_SETTINGS (SETTING_NAME, SETTING_VALUE)
        VALUES (DBMS_DATA_MINING.PREP_AUTO, DBMS_DATA_MINING.PREP_AUTO_ON);
        
        --INSERT INTO NEURAL_NETWORK_MODEL_SETTINGS (SETTING_NAME, SETTING_VALUE)
        --VALUES ('NNET_NODES_PER_LAYER', '2,3');
       --INSERT INTO NEURAL_NETWORK_MODEL_SETTINGS (SETTING_NAME, SETTING_VALUE)
       --VALUES ('NNET_ACTIVATIONS', 'NNET_ACTIVATIONS_TANH'); --NNET_ACTIVATIONS_LOG_SIG
    
       COMMIT;
    END;

    -- we may need to give user model building privileges, to avoid any issues.
    --login as SYS and run the following commands
    --GRANT CREATE MINING MODEL TO "C##ROB";
    --GRANT SELECT ANY MINING MODEL TO "C##ROB";
    --GRANT ALTER ANY MINING MODEL TO "C##ROB";
    --GRANT DROP ANY MINING MODEL TO "C##ROB";
    --GRANT COMMENT ANY MINING MODEL TO "C##ROB";
    
    BEGIN --CUSTOMER_CHURN_NEURAL_NETWORK_MODEL was incorrect _2 uses auto
        DBMS_DATA_MINING.CREATE_MODEL(
           MODEL_NAME => 'CUSTOMER_CHURN_NEURAL_NETWORK_MODEL_2',
           MINING_FUNCTION => DBMS_DATA_MINING.CLASSIFICATION,
           DATA_TABLE_NAME => 'CUSTOMER_CHURN_CASE_TABLE_VIEW',
           CASE_ID_COLUMN_NAME => 'CASE_ID',
           TARGET_COLUMN_NAME => 'CUSTOMER_ACTIVE',
           SETTINGS_TABLE_NAME => 'NEURAL_NETWORK_MODEL_SETTINGS');
    END;

    -- explore model meta data
    DESCRIBE USER_MINING_MODEL_SETTINGS
    
    --describe the model settings tables
    SELECT MODEL_NAME,
       MINING_FUNCTION,
       ALGORITHM,
       BUILD_DURATION,
       MODEL_SIZE
    FROM USER_MINING_MODELS;
    
    --check that model was created
    SELECT SETTING_NAME,
    SETTING_VALUE,
    SETTING_TYPE
    FROM USER_MINING_MODEL_SETTINGS
    WHERE MODEL_NAME in 'CUSTOMER_CHURN_NEURAL_NETWORK_MODEL_2';

    --check algorithm settings
    SELECT ATTRIBUTE_NAME,
       ATTRIBUTE_TYPE,
       USAGE_TYPE, 
       TARGET
    FROM ALL_MINING_MODEL_ATTRIBUTES
    WHERE MODEL_NAME = 'CUSTOMER_CHURN_NEURAL_NETWORK_MODEL_2';
    
-- TRAIN MODEL 2: NEURAL NETWORK MODEL
    --Confusion Matrix
    CREATE OR REPLACE VIEW CUSTOMER_CHURN_NN_TEST_RESULTS
    AS
    SELECT CASE_ID, 
        PREDICTION(CUSTOMER_CHURN_NEURAL_NETWORK_MODEL_2 USING *) predicted_value, 
        PREDICTION_PROBABILITY(CUSTOMER_CHURN_NEURAL_NETWORK_MODEL_2 USING *) probability
    FROM CUSTOMER_CHURN_TEST_SET;
    
    SELECT * FROM CUSTOMER_CHURN_NN_TEST_RESULTS;
    
    
    DROP TABLE CUSTOMER_CHURN_NN_CONFUSION_MATRIX;
    DECLARE
       v_accuracy NUMBER;
    BEGIN
    DBMS_DATA_MINING.COMPUTE_CONFUSION_MATRIX (
       accuracy => v_accuracy,
       apply_result_table_name => 'CUSTOMER_CHURN_NN_TEST_RESULTS',
       target_table_name => 'CUSTOMER_CHURN_TEST_SET',
       case_id_column_name => 'CASE_ID',
       target_column_name => 'CUSTOMER_ACTIVE',
       confusion_matrix_table_name => 'CUSTOMER_CHURN_NN_CONFUSION_MATRIX',
       score_column_name => 'PREDICTED_VALUE',
       score_criterion_column_name => 'PROBABILITY',
       cost_matrix_table_name => null,
       apply_result_schema_name => null,
       target_schema_name => null,
       cost_matrix_schema_name => null,
       score_criterion_type => 'PROBABILITY');
       DBMS_OUTPUT.PUT_LINE('**** MODEL ACCURACY ****: ' || ROUND(v_accuracy,4));
    END;
    
    SELECT * FROM CUSTOMER_CHURN_NN_CONFUSION_MATRIX;
    

-- TEST MODEL 2: NEURAL NETWORK MODEL
    -- apply model to new data
    BEGIN
       dbms_data_mining.apply(
       model_name => 'CUSTOMER_CHURN_NEURAL_NETWORK_MODEL_2',
       data_table_name => 'CUSTOMER_CHURN_APPLY_SET',
       case_id_column_name => 'CASE_ID',
       result_table_name => 'NEW_DATA_SCORED_NN_2');
    END;
    
    --Inspect the results in the NEW_DATA_SCORED table.
    SELECT * FROM NEW_DATA_SCORED_NN_2;
    
    -- apply model to real time data
    SELECT CASE_ID, PREDICTION(CUSTOMER_CHURN_NEURAL_NETWORK_MODEL_2 using *)
    FROM CUSTOMER_CHURN_APPLY_SET;
    
    
    SELECT CASE_ID, PREDICTION(CUSTOMER_CHURN_NEURAL_NETWORK_MODEL_2 using *), PREDICTION_PROBABILITY(CUSTOMER_CHURN_NEURAL_NETWORK_MODEL_2 using *)
    FROM CUSTOMER_CHURN_APPLY_SET
    WHERE rownum <=5;
    
    SELECT CASE_ID
    FROM CUSTOMER_CHURN_APPLY_SET
    WHERE PREDICTION(CUSTOMER_CHURN_NEURAL_NETWORK_MODEL_2 using *) = 1
    AND rownum <=10;
    
-- PREDICT WITH MODEL 2: NEURAL NETWORK MODEL
    --what if analysis on dummy data
    Select PREDICTION_PROBABILITY (CUSTOMER_CHURN_NEURAL_NETWORK_MODEL_2, 0
    USING 
    46 as CUSTOMER_AGE,
    'non-working' as SOCIAL_GRADE,
    'cosmopolitan' as CONTRACT_PLAN,
    432 as PLAN_DURATION,
    'off-peak' as CALL_TYPE,
    561.37 as CALL_DURATION,
    0.84 AS CALL_REVENUE_GENERATED) Pred_Prob
    from dual;


-- PL/SQL used to combine the accuracy measures of the two models
    --BEGIN
        SELECT * FROM CUSTOMER_CHURN_DT_CONFUSION_MATRIX
        UNION
        SELECT * FROM CUSTOMER_CHURN_NN_CONFUSION_MATRIX;
    --END;