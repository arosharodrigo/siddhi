#!/bin/bash

JVM=/usr/lib/jvm/java-6-sun-1.6.0.26/jre/bin/java
JDB=/usr/lib/jvm/java-6-sun/bin/jdb

#AGENTLIB=-agentlib:hprof=cpu=samples

CLASSPATH=/home/prabodha/project/siddhi-git-dev/modules/siddhi-samples/quick-start-samples/target/build/classes:/home/prabodha/project/siddhi-git-dev/modules/siddhi-samples/target/lib/antlr4-runtime-4.3.jar:/home/prabodha/project/siddhi-git-dev/modules/siddhi-samples/target/lib/disruptor-3.2.1.jar:/home/prabodha/project/siddhi-git-dev/modules/siddhi-samples/target/lib/log4j-1.2.14.jar:/home/prabodha/project/siddhi-git-dev/modules/siddhi-samples/target/lib/siddhi-core-3.0.0-SNAPSHOT.jar:/home/prabodha/project/siddhi-git-dev/modules/siddhi-samples/target/lib/siddhi-query-api-3.0.0-SNAPSHOT.jar:/home/prabodha/project/siddhi-git-dev/modules/siddhi-samples/target/lib/siddhi-query-compiler-3.0.0-SNAPSHOT.jar:/home/prabodha/project/siddhi-git-dev/modules/siddhi-query-api/target/siddhi-query-api-3.0.0-SNAPSHOT.jar:/home/prabodha/project/siddhi-git-dev/modules/siddhi-gpu/target/antlr4-annotations.jar:/home/prabodha/project/siddhi-git-dev/modules/siddhi-gpu/target/antlr4-runtime.jar:/home/prabodha/project/siddhi-git-dev/modules/siddhi-gpu/target/hamcrest-core.jar:/home/prabodha/project/siddhi-git-dev/modules/siddhi-gpu/target/javacpp-cudaclib-linux-x86_64.jar:/home/prabodha/project/siddhi-git-dev/modules/siddhi-gpu/target/javacpp-cudaclib.jar:/home/prabodha/project/siddhi-git-dev/modules/siddhi-gpu/target/javacpp.jar:/home/prabodha/project/siddhi-git-dev/modules/siddhi-gpu/target/junit.jar:/home/prabodha/project/siddhi-git-dev/modules/siddhi-gpu/target/log4j.jar:/home/prabodha/project/siddhi-git-dev/modules/siddhi-gpu/target/mvel2.jar:/home/prabodha/project/siddhi-git-dev/modules/siddhi-gpu/target/org.abego.treelayout.core.jar:/home/prabodha/project/siddhi-git-dev/modules/siddhi-gpu/target/siddhi-query-api.jar:/home/prabodha/project/siddhi-git-dev/modules/siddhi-gpu/target/siddhi-query-compiler.jar:/home/prabodha/project/siddhi-git-dev/modules/siddhi-query-compiler/target/siddhi-query-compiler-3.0.0-SNAPSHOT.jar:/home/prabodha/project/siddhi-git-dev/modules/siddhi-core/target/siddhi-core-3.0.0-SNAPSHOT.jar:/home/prabodha/project/siddhi-git-dev/modules/siddhi-samples/quick-start-samples/src/main/resources/log4j.properties

#APP=org.wso2.siddhi.sample.GpuFilterSample
APP=org.wso2.siddhi.sample.GpuJoinSample

ringbuffer_size=16384
threadpool_size=8
block_size=128

#${JVM} -classpath ${CLASSPATH} ${APP} ${ringbuffer_size} ${threadpool_size} ${block_size}
#${JDB} -classpath ${CLASSPATH} ${APP} ${ringbuffer_size} ${threadpool_size} ${block_size}
${JVM} ${AGENTLIB} -classpath ${CLASSPATH} ${APP}
#cuda-memcheck --leak-check full --save mcheck.save ${JVM} -classpath ${CLASSPATH} ${APP}
