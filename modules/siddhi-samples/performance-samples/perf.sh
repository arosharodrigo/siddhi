#!/bin/bash

#ant SimpleFilterSingleQueryWithDisruptorAllCfgPerformance &> logs/pref.log
#ant SimpleFilterSingleQueryWithGpuAllCfgPerformance &> logs/SimpleFilterSingleQueryWithGpuAllCfgPerformance.log

JVM=java

AGENTLIB=-agentlib:hprof=cpu=samples

CLASSPATH=/home/prabodha/devel/research/cep/siddhi/siddhi-git-dev/modules/siddhi-samples/performance-samples/target/build/classes:/home/prabodha/devel/research/cep/siddhi/siddhi-git-dev/modules/siddhi-samples/target/lib/antlr4-runtime-4.3.jar:/home/prabodha/devel/research/cep/siddhi/siddhi-git-dev/modules/siddhi-samples/target/lib/disruptor-3.2.1.jar:/home/prabodha/devel/research/cep/siddhi/siddhi-git-dev/modules/siddhi-samples/target/lib/log4j-1.2.14.jar:/home/prabodha/devel/research/cep/siddhi/siddhi-git-dev/modules/siddhi-samples/target/lib/siddhi-core-3.0.0-SNAPSHOT.jar:/home/prabodha/devel/research/cep/siddhi/siddhi-git-dev/modules/siddhi-samples/target/lib/siddhi-query-api-3.0.0-SNAPSHOT.jar:/home/prabodha/devel/research/cep/siddhi/siddhi-git-dev/modules/siddhi-samples/target/lib/siddhi-query-compiler-3.0.0-SNAPSHOT.jar:/home/prabodha/devel/research/cep/siddhi/siddhi-git-dev/modules/siddhi-query-api/target/siddhi-query-api-3.0.0-SNAPSHOT.jar:/home/prabodha/devel/research/cep/siddhi/siddhi-git-dev/modules/siddhi-query-compiler/target/siddhi-query-compiler-3.0.0-SNAPSHOT.jar:/home/prabodha/devel/research/cep/siddhi/siddhi-git-dev/modules/siddhi-core/target/siddhi-core-3.0.0-SNAPSHOT.jar:/home/prabodha/devel/research/cep/siddhi/siddhi-git-dev/modules/siddhi-samples/performance-samples/src/main/resources/log4j.properties

APP=org.wso2.siddhi.performance.SimpleFilterSingleQueryWithDisruptorPerformance
LOG=logs/ComplexFilterSingleQueryAllCfgPerformance.log

${JVM} ${AGENTLIB} -classpath ${CLASSPATH} ${APP} #&>${LOG}
#ohup ${JVM} ${AGENTLIB} -classpath ${CLASSPATH} ${APP} &>${LOG} &
