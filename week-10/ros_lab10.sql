--LAB 10
--What is the maximum and minimum salary across all employees?
SELECT MAX(SAL) AS MAX_SAL FROM EMP;
SELECT MIN(SAL) AS MIN_SAL FROM EMP;

--What is the difference between the maximum and the average salary
SELECT MAX(SAL) - MIN (SAL) AS DIFF FROM EMP;

--What is the average salary per department
SELECT EMPNO, DEPTNO, SAL, ROW_NUMBER() over (PARTITION BY DEPTNO ORDER BY SAL desc) SAL_RANK_RNUM
FROM EMP 
ORDER BY DEPTNO, SAL desc;