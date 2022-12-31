SET DEFINE OFF

CREATE TABLE rate_types_csv ( id NUMBER(38),
name VARCHAR2(26));



INSERT INTO rate_type (id, name) 
VALUES (1, 'peak');

INSERT INTO rate_type (id, name) 
VALUES (2, 'off-peak');

INSERT INTO rate_type (id, name) 
VALUES (3, 'international');

INSERT INTO rate_type (id, name) 
VALUES (4, 'roaming');

INSERT INTO rate_type (id, name) 
VALUES (5, 'voice mail');

INSERT INTO rate_type (id, name) 
VALUES (6, 'customer service');

