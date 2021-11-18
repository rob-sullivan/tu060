drop table student cascade constraints;

create table STUDENT (
student_number VARCHAR2(20) PRIMARY KEY,
first_name VARCHAR2(20),
surname VARCHAR2(20),
dob DATE,
prog_code VARCHAR2(6));

insert into STUDENT (student_number, first_name, surname, dob, prog_code)
values ('D020150120', 'James', 'Smith', to_date('19/01/1995', 'DD/MM/YYYY'), 'TU256');

insert into STUDENT (student_number, first_name, surname, dob, prog_code)
values ('D020150121', 'John', 'Brown', to_date('18/09/1987', 'DD/MM/YYYY'), 'TU256');

insert into STUDENT (student_number, first_name, surname, dob, prog_code)
values ('D020150122', 'Patricia', 'Wilson', to_date('04/10/1973', 'DD/MM/YYYY'), 'TU256');

insert into STUDENT (student_number, first_name, surname, dob, prog_code)
values ('D020150123', 'Karen', 'Davies', to_date('28/12/2000', 'DD/MM/YYYY'), 'TU256');

select * from student;
select first_name from student where prog_code = 'TU256';
select first_name from student where first_name like 'D%';

select * from student where student_number = 'D020150123';

update STUDENT
set prog_code = 'TU059'
where student_number = 'D020150123';

select * from student where student_number = 'D020150123';


delete from STUDENT where student_number = 'D020150123';
delete from STUDENT where first_name like 'D%';
select * from STUDENT;