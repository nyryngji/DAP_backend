truncate TABLE chat_history; 
truncate TABLE chat_session; 
truncate TABLE saved_query; 
truncate TABLE ALL_QUERY_COUNT; 
COMMIT;

delete from chat_history
where CHAT_SESSION_ID in (select CHAT_SESSION_ID from CHAT_SESSION
where user_seq = 1);

commit;

SELECT COUNT(*), ROUND(AVG(CREATING_TIME),0) FROM SAVED_QUERY
WHERE CREATED_AT <= CURRENT_TIMESTAMP;

delete from chat_history
where user_seq = 1;
commit;

select * from ALL_QUERY_COUNT;

SELECT log_date, status
FROM user_scheduler_job_run_details
WHERE job_name = 'APP_HOURLY_CHECK_JOB'
ORDER BY log_date DESC;

SELECT * from ALL_QUERY_COUNT;

SELECT value
FROM v$sysstat
WHERE name = 'CPU used by this session';

SELECT SUM(st.value) AS CPU_CS
  FROM   v$session s
  JOIN   v$sesstat st ON s.sid = st.sid
  JOIN   v$statname sn ON st.statistic# = sn.statistic#
  WHERE  sn.name = 'CPU used by this session'
  AND    s.username = 'TEAM8';

UPDATE ALL_QUERY_COUNT
set over_30s = 23
where NOWTIME = 22;

COMMIT;

SELECT saved_query_id from SAVED_QUERY;

SELECT inputevents.subject_id 
FROM inputevents JOIN chartevents ON inputevents.subject_id = chartevents.subject_id 
WHERE ((inputevents.itemid IN (221906) AND chartevents.itemid IN (220052, 225312)) 
AND chartevents.valuenum < 65) AND 
NOT EXISTS(SELECT 1 FROM chartevents ce2 WHERE inputevents.subject_id = ce2.subject_id 
AND ce2.itemid IN (220051, 225310, 224643, 227242) AND ce2.valuenum >= 65);