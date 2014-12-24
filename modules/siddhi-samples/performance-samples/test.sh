#!/bin/bash

#ant SimpleFilterSingleQueryWithDisruptorAllCfgPerformance &> logs/pref.log
#ant SimpleFilterSingleQueryWithGpuAllCfgPerformance &> logs/SimpleFilterSingleQueryWithGpuAllCfgPerformance.log

JVM=/usr/lib/jvm/java-6-sun-1.6.0.26/jre/bin/java

#AGENTLIB=-agentlib:hprof=cpu=samples

CLASSPATH=/home/prabodha/project/siddhi-git-dev/modules/siddhi-samples/performance-samples/target/build/classes:/home/prabodha/project/siddhi-git-dev/modules/siddhi-samples/target/lib/antlr4-runtime-4.3.jar:/home/prabodha/project/siddhi-git-dev/modules/siddhi-samples/target/lib/commons-cli-1.2.jar:/home/prabodha/project/siddhi-git-dev/modules/siddhi-samples/target/lib/disruptor-3.2.1.jar:/home/prabodha/project/siddhi-git-dev/modules/siddhi-samples/target/lib/log4j-1.2.14.jar:/home/prabodha/project/siddhi-git-dev/modules/siddhi-samples/target/lib/siddhi-core-3.0.0-SNAPSHOT.jar:/home/prabodha/project/siddhi-git-dev/modules/siddhi-samples/target/lib/siddhi-query-api-3.0.0-SNAPSHOT.jar:/home/prabodha/project/siddhi-git-dev/modules/siddhi-samples/target/lib/siddhi-query-compiler-3.0.0-SNAPSHOT.jar:/home/prabodha/project/siddhi-git-dev/modules/siddhi-query-api/target/siddhi-query-api-3.0.0-SNAPSHOT.jar:/home/prabodha/project/siddhi-git-dev/modules/siddhi-query-compiler/target/siddhi-query-compiler-3.0.0-SNAPSHOT.jar:/home/prabodha/project/siddhi-git-dev/modules/siddhi-gpu/target/antlr4-annotations.jar:/home/prabodha/project/siddhi-git-dev/modules/siddhi-gpu/target/antlr4-runtime.jar:/home/prabodha/project/siddhi-git-dev/modules/siddhi-gpu/target/hamcrest-core.jar:/home/prabodha/project/siddhi-git-dev/modules/siddhi-gpu/target/javacpp-cudaclib-linux-x86_64.jar:/home/prabodha/project/siddhi-git-dev/modules/siddhi-gpu/target/javacpp-cudaclib.jar:/home/prabodha/project/siddhi-git-dev/modules/siddhi-gpu/target/javacpp.jar:/home/prabodha/project/siddhi-git-dev/modules/siddhi-gpu/target/junit.jar:/home/prabodha/project/siddhi-git-dev/modules/siddhi-gpu/target/log4j.jar:/home/prabodha/project/siddhi-git-dev/modules/siddhi-gpu/target/mvel2.jar:/home/prabodha/project/siddhi-git-dev/modules/siddhi-gpu/target/org.abego.treelayout.core.jar:/home/prabodha/project/siddhi-git-dev/modules/siddhi-gpu/target/siddhi-query-api.jar:/home/prabodha/project/siddhi-git-dev/modules/siddhi-gpu/target/siddhi-query-compiler.jar:/home/prabodha/project/siddhi-git-dev/modules/siddhi-core/target/siddhi-core-3.0.0-SNAPSHOT.jar:/home/prabodha/project/siddhi-git-dev/modules/siddhi-samples/performance-samples/src/main/resources/log4j.properties

APP=org.wso2.siddhi.performance.ComplexFilterSingleQueryPerformance
LOG=logs/ComplexFilterSingleQueryPerformance

a="false"
g="true"
r=2048
t=8
b=128
c=$((5000000 * r))
#${JVM} ${AGENTLIB} -classpath ${CLASSPATH} ${APP} --enable-gpu true  --event-count ${c} --ringbuffer-size ${r} --threadpool-size ${t} --events-per-tblock ${b} &>${LOG}_gpu_${r}_${t}_${b}.log
${JVM} ${AGENTLIB} -classpath ${CLASSPATH} ${APP} --enable-async false --enable-gpu false  --event-count ${c} --ringbuffer-size ${r} --threadpool-size ${t} --events-per-tblock ${b} &>${LOG}_cpu_singlethread_${r}_${t}_${b}.log
exit
#${JVM} ${AGENTLIB} -classpath ${CLASSPATH} ${APP} &>${LOG}
#nohup ${JVM} ${AGENTLIB} -classpath ${CLASSPATH} ${APP} &>${LOG} &

#for r in 512 1024 2048 4096 8192 16384; do
#	for t in 2 4 8 16; do
#		for b in 64 128 256 512 1024; do
#			c=$((5000000 * r)) 
#			echo "Executing --event-count ${c} --ringbuffer-size ${r} --threadpool-size ${t} --events-per-tblock ${b}"
#			${JVM} ${AGENTLIB} -classpath ${CLASSPATH} ${APP} --enable-gpu false --event-count ${c} --ringbuffer-size ${r} --threadpool-size ${t} --events-per-tblock ${b} &>${LOG}_cpu_${r}_${t}_${b}.log			
#			${JVM} ${AGENTLIB} -classpath ${CLASSPATH} ${APP} --enable-gpu true  --event-count ${c} --ringbuffer-size ${r} --threadpool-size ${t} --events-per-tblock ${b} &>${LOG}_gpu_${r}_${t}_${b}.log			
#		done
#	done
#done

for r in 2048 4096 8192 16384; do
	for t in 2 4 8 16; do
		for b in 64 128 256 512 1024; do
			c=$((5000000 * r)) 
			echo "Executing --event-count ${c} --ringbuffer-size ${r} --threadpool-size ${t} --events-per-tblock ${b}"
			${JVM} ${AGENTLIB} -classpath ${CLASSPATH} ${APP} --enable-gpu false --event-count ${c} --ringbuffer-size ${r} --threadpool-size ${t} --events-per-tblock ${b} &>${LOG}_cpu_${r}_${t}_${b}.log			
			${JVM} ${AGENTLIB} -classpath ${CLASSPATH} ${APP} --enable-gpu true  --event-count ${c} --ringbuffer-size ${r} --threadpool-size ${t} --events-per-tblock ${b} &>${LOG}_gpu_${r}_${t}_${b}.log			
		done
	done
done
