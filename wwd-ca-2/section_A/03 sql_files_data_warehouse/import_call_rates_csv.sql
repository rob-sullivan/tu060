SET DEFINE OFF

CREATE TABLE call_rates_csv ( call_type_id NUMBER(38),
plan_id NUMBER(38),
cost_per_minute FLOAT);



INSERT INTO call_rates_csv (call_type_id, plan_id, cost_per_minute) 
VALUES (1, 1, 0.09);

INSERT INTO call_rates_csv (call_type_id, plan_id, cost_per_minute) 
VALUES (1, 2, 0.12);

INSERT INTO call_rates_csv (call_type_id, plan_id, cost_per_minute) 
VALUES (1, 3, 0.12);

INSERT INTO call_rates_csv (call_type_id, plan_id, cost_per_minute) 
VALUES (2, 1, 0.09);

INSERT INTO call_rates_csv (call_type_id, plan_id, cost_per_minute) 
VALUES (2, 2, 0.03);

INSERT INTO call_rates_csv (call_type_id, plan_id, cost_per_minute) 
VALUES (2, 3, 0.09);

INSERT INTO call_rates_csv (call_type_id, plan_id, cost_per_minute) 
VALUES (3, 1, 0.26);

INSERT INTO call_rates_csv (call_type_id, plan_id, cost_per_minute) 
VALUES (3, 2, 0.26);

INSERT INTO call_rates_csv (call_type_id, plan_id, cost_per_minute) 
VALUES (3, 3, 0.13);

INSERT INTO call_rates_csv (call_type_id, plan_id, cost_per_minute) 
VALUES (4, 1, 0.57);

INSERT INTO call_rates_csv (call_type_id, plan_id, cost_per_minute) 
VALUES (4, 2, 0.57);

INSERT INTO call_rates_csv (call_type_id, plan_id, cost_per_minute) 
VALUES (4, 3, 0.37);

INSERT INTO call_rates_csv (call_type_id, plan_id, cost_per_minute) 
VALUES (5, 1, 0.1);

INSERT INTO call_rates_csv (call_type_id, plan_id, cost_per_minute) 
VALUES (5, 2, 0.1);

INSERT INTO call_rates_csv (call_type_id, plan_id, cost_per_minute) 
VALUES (5, 3, 0.1);

INSERT INTO call_rates_csv (call_type_id, plan_id, cost_per_minute) 
VALUES (6, 1, 0);

INSERT INTO call_rates_csv (call_type_id, plan_id, cost_per_minute) 
VALUES (6, 2, 0);

INSERT INTO call_rates_csv (call_type_id, plan_id, cost_per_minute) 
VALUES (6, 3, 0);

