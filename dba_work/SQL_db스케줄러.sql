-- [DBA가 설정해야 하는 권한]

-- GRANT CREATE JOB TO TEAM8;
-- GRANT SELECT_CATALOG_ROLE TO TEAM8; : CPU 사용량 조회를 위해

-- SELECT * FROM user_sys_privs;

-- 스케줄러가 SUCCESS됐는지 안됐는지
-- SELECT log_date, status
-- FROM user_scheduler_job_run_details
-- WHERE job_name = 'APP_HOURLY_CHECK_JOB'
-- ORDER BY log_date DESC;


-- 정각마다 쿼리 개수 업데이트
CREATE OR REPLACE PROCEDURE app_hourly_check AS
  v_count NUMBER;
  v_avg   NUMBER;
  v_hour  NUMBER;
  cpu_use  NUMBER;
BEGIN
  -- 현재 시간의 시(hour) 추출
  SELECT EXTRACT(HOUR FROM CAST(CURRENT_TIMESTAMP AT TIME ZONE 'Asia/Seoul' AS TIMESTAMP))
  INTO   v_hour
  FROM   dual;

  -- 집계 계산
  SELECT COUNT(*),
         NVL(ROUND(AVG(CREATING_TIME),0),0)
  INTO   v_count,
         v_avg
  FROM   SAVED_QUERY
  WHERE  CREATED_AT <= CURRENT_TIMESTAMP AT TIME ZONE 'Asia/Seoul';

  -- cpu 사용량 계산
  SELECT SUM(st.value) AS CPU_CS
  INTO   cpu_use
  FROM   v$session s
  JOIN   v$sesstat st ON s.sid = st.sid
  JOIN   v$statname sn ON st.statistic# = sn.statistic#
  WHERE  sn.name = 'CPU used by this session'
  AND    s.username = 'TEAM8';

  -- 시간별 로그 INSERT
  INSERT INTO ALL_QUERY_COUNT (
    NOWTIME,
    LAST_RUN_TIME,
    QUERY_COUNT,
    AVG_CREATING_TIME,
    CPU_CS
  )
  VALUES (
    v_hour,
    CURRENT_TIMESTAMP AT TIME ZONE 'Asia/Seoul',
    v_count,
    v_avg,
    cpu_use
  );

  COMMIT;
END;
/

BEGIN
  DBMS_SCHEDULER.CREATE_JOB (
    job_name        => 'APP_HOURLY_CHECK_JOB',
    job_type        => 'STORED_PROCEDURE',
    job_action      => 'APP_HOURLY_CHECK',
    repeat_interval => 'FREQ=HOURLY;BYMINUTE=0;BYSECOND=0',
    enabled         => TRUE
  );
END;
/

-- 자정마다 모든 유저의 채팅 내역 삭제 (즐겨찾기 제외)
CREATE OR REPLACE PROCEDURE app_midnight_cleanup AS
BEGIN
    EXECUTE IMMEDIATE 'TRUNCATE TABLE CHAT_HISTORY';
    EXECUTE IMMEDIATE 'TRUNCATE TABLE ALL_QUERY_COUNT';
    DELETE FROM saved_query
     WHERE IS_FAVORITE = 'N';

    COMMIT;

EXCEPTION
    WHEN OTHERS THEN
        ROLLBACK;
        RAISE;
END;
/

BEGIN
  DBMS_SCHEDULER.CREATE_JOB (
    job_name        => 'APP_MIDNIGHT_CLEANUP_JOB',
    job_type        => 'STORED_PROCEDURE',
    job_action      => 'APP_MIDNIGHT_CLEANUP',
    repeat_interval => 'FREQ=DAILY;BYHOUR=23;BYMINUTE=59;BYSECOND=59',
    enabled         => TRUE
  );
END;
/


BEGIN
  DBMS_SCHEDULER.RUN_JOB (
    job_name => 'APP_HOURLY_CHECK_JOB',
    use_current_session => TRUE
  );
END;
/



BEGIN
    DBMS_SCHEDULER.DROP_JOB (
        job_name => 'APP_MIDNIGHT_CLEANUP_JOB',
        force    => TRUE
    );
END;
/
DROP PROCEDURE APP_MIDNIGHT_CLEANUP;


BEGIN
  DBMS_SCHEDULER.RUN_JOB (
    job_name => 'APP_HOURLY_CHECK_JOB',
    force    => TRUE
  );
END;
/
DROP PROCEDURE APP_HOURLY_CHECK;


SELECT job_name 
FROM user_scheduler_jobs
WHERE job_name = 'APP_MIDNIGHT_CLEANUP_JOB';

